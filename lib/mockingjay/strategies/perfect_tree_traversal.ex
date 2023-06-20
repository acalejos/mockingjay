defmodule Mockingjay.Strategies.PerfectTreeTraversal do
  alias Mockingjay.Tree
  alias Mockingjay.DecisionTree
  import Nx.Defn
  @behaviour Mockingjay.Strategy

  # Derived from Binary Tree structure
  @factor 2

  @impl true
  def init(data, opts \\ []) do
    opts = Keyword.validate!(opts, [:forward, :aggregate, :post_transform])
    trees = DecisionTree.trees(data)
    condition = DecisionTree.condition(data)
    n_classes = DecisionTree.num_classes(data)
    num_trees = length(trees)
    # Number of classes each weak learner can predict
    # TODO: This is currently always 1, but could be more
    n_weak_learner_classes = 1

    max_tree_depth =
      Enum.reduce(trees, 0, fn tree, acc ->
        max(acc, Tree.depth(tree))
      end)

    perfect_trees = trees |> Enum.map(&make_tree_perfect(&1, 0, max_tree_depth))

    # num_internal_nodes = @factor ** max_tree_depth - 1
    # num_leaves = @factor ** max_tree_depth

    {features, thresholds, values} =
      perfect_trees
      |> Enum.reduce({[], [], []}, fn tree, {all_features, all_thresholds, all_values} ->
        {tf, tt, tv} =
          Enum.reduce(Tree.bfs(tree), {[], [], []}, fn node,
                                                       {tree_features, tree_thresholds,
                                                        tree_values} ->
            case node do
              %Tree{left: nil, right: nil} ->
                tree_values = tree_values ++ [[node.value]]
                {tree_features, tree_thresholds, tree_values}

              %Tree{left: _left, right: _right} ->
                tree_features = tree_features ++ [node.value.feature]
                tree_thresholds = tree_thresholds ++ [node.value.threshold]
                {tree_features, tree_thresholds, tree_values}
            end
          end)

        tf = Nx.tensor(tf)
        tt = Nx.tensor(tt)
        tv = Nx.tensor(tv)

        {[tf | all_features], [tt | all_thresholds], [tv | all_values]}
      end)

    # shape of {num_trees, 2 ** max_tree_depth - 1}
    features =
      Nx.stack(Enum.reverse(features))
      |> Nx.reshape({num_trees, @factor ** max_tree_depth - 1})
      |> Nx.as_type(:s64)

    # shape of {num_trees, 2 ** max_tree_depth - 1}
    thresholds =
      Nx.stack(Enum.reverse(thresholds))
      |> Nx.reshape({num_trees, @factor ** max_tree_depth - 1})
      |> Nx.as_type(:f64)

    # shape of {num_trees, 2 ** max_tree_depth}
    # TODO (Remove this) : Confirmed these leaves match the Hummingbird implementation
    values =
      Nx.stack(Enum.reverse(values))
      |> Nx.reshape({:auto, n_weak_learner_classes})
      |> Nx.as_type(:f64)

    root_features = features[[.., 0]] |> Nx.flatten() |> Nx.as_type(:s64)
    root_thresholds = thresholds[[.., 0]] |> Nx.flatten() |> Nx.as_type(:f64)

    {features, thresholds} =
      Enum.reduce(1..(max_tree_depth - 1), {[], []}, fn depth, {all_nodes, all_biases} ->
        start = @factor ** depth - 1
        stop = @factor ** (depth + 1) - 2
        n = features[[.., start..stop]] |> Nx.flatten() |> Nx.as_type(:s64)
        b = thresholds[[.., start..stop]] |> Nx.flatten() |> Nx.as_type(:f64)
        {[n | all_nodes], [b | all_biases]}
      end)

    # TODO (Remove this) : Confirmed these match the Hummingbird implementation
    features = Enum.reverse(features)
    thresholds = Enum.reverse(thresholds)

    nt = @factor * num_trees

    indices = 0..(nt - 1)//2 |> Enum.into([]) |> Nx.tensor(type: :s64)

    forward_args =
      if opts[:forward] do
        [custom_forward: opts[:forward]]
      else
        [
          indices: indices,
          num_trees: num_trees,
          max_tree_depth: max_tree_depth,
          features: features,
          thresholds: thresholds,
          root_features: root_features,
          root_thresholds: root_thresholds,
          values: values,
          condition: Mockingjay.Strategy.cond_to_fun(condition),
          n_classes: n_weak_learner_classes
        ]
      end

    post_transform_args =
      if opts[:post_transform] do
        [custom_post_transform: opts[:post_transform]]
      else
        [n_classes: n_classes]
      end

    aggregate_args =
      if opts[:aggregate] do
        [custom_aggregate: opts[:aggregate]]
      else
        [
          n_classes: n_classes,
          n_trees: num_trees
        ]
      end

    {forward_args, aggregate_args, post_transform_args}
  end

  @impl true
  def forward(x, opts \\ []) do
    prev_indices =
      x
      |> Nx.take(opts[:root_features], axis: 1)
      |> opts[:condition].(opts[:root_thresholds])
      |> Nx.add(opts[:indices])
      |> Nx.reshape({:auto})
      |> IO.inspect(label: "prev_indices before loop", limit: :infinity)

    prev_indices =
      Enum.zip(opts[:features], opts[:thresholds])
      |> Enum.with_index()
      |> Enum.reduce(prev_indices, fn {{nodes, biases}, index}, acc ->
        gather_indices = Nx.take(nodes, acc) |> Nx.reshape({:auto, opts[:num_trees]})
        features = Nx.take_along_axis(x, gather_indices, axis: 1) |> Nx.reshape({:auto})
        IO.puts("features at level #{inspect(index)}: #{inspect(features, limit: :infinity)}")

        acc
        |> Nx.multiply(@factor)
        |> Nx.add(opts[:condition].(features, Nx.take(biases, acc)))
      end)

    IO.inspect(prev_indices, label: "prev_indices after loop", limit: :infinity)

    Nx.take(opts[:values], prev_indices)
    |> Nx.reshape({:auto, opts[:num_trees], opts[:n_classes]})
  end

  @impl true
  def aggregate(x, opts \\ []) do
    opts = Keyword.validate!(opts, [:n_classes, :n_trees, :custom_aggregate])

    if opts[:custom_aggregate] do
      opts[:custom_aggregate].(x)
    else
      n_trees = opts[:n_trees]
      n_classes = opts[:n_classes]

      cond do
        n_classes > 1 and n_trees > 1 ->
          n_gbdt_classes = if n_classes > 2, do: n_classes, else: 1
          n_trees_per_class = trunc(n_trees / n_gbdt_classes)

          ensemble_aggregate(
            x,
            n_gbdt_classes,
            n_trees_per_class
          )

        n_classes > 1 and n_trees == 1 ->
          _aggregate(x)

        true ->
          raise "Unknown output type"
      end
    end
  end

  @impl true
  def post_transform(x, opts \\ []) do
    opts = Keyword.validate!(opts, [:custom_post_transform, :n_classes])

    if opts[:custom_post_transform] do
      opts[:custom_post_transform].(x)
    else
      transform =
        Mockingjay.Strategy.infer_post_transform(opts[:n_classes])
        |> Mockingjay.Strategy.post_transform_to_func()

      transform.(x)
    end
  end

  def make_tree_perfect(tree, current_depth, max_depth) do
    case tree do
      %Tree{left: nil, right: nil} ->
        if current_depth < max_depth do
          %Tree{
            id: make_ref(),
            # This can be anything since either path results in the same leaf
            value: %{feature: 0, threshold: 0},
            left: make_tree_perfect(tree, current_depth + 1, max_depth),
            right: make_tree_perfect(tree, current_depth + 1, max_depth)
          }
        else
          tree
        end

      %Tree{left: left, right: right} ->
        struct(tree,
          left: make_tree_perfect(left, current_depth + 1, max_depth),
          right: make_tree_perfect(right, current_depth + 1, max_depth)
        )
    end
  end

  deftransformp _aggregate(x) do
    x
    |> Nx.sum(axes: [1])
  end

  deftransformp ensemble_aggregate(x, n_gbdt_classes, n_trees_per_class) do
    x
    |> Nx.reshape({:auto, n_gbdt_classes, n_trees_per_class})
    |> Nx.sum(axes: [2])
  end
end
