defmodule Mockingjay.Strategies.PerfectTreeTraversal do
  @moduledoc false
  alias Mockingjay.Tree
  alias Mockingjay.DecisionTree
  import Nx.Defn
  @behaviour Mockingjay.Strategy

  # Derived from Binary Tree structure
  @factor 2

  @impl true
  def init(data, opts \\ []) do
    opts = Keyword.validate!(opts, [:forward, :aggregate, :post_transform, reorder_trees: true])
    trees = DecisionTree.trees(data)
    condition = DecisionTree.condition(data)
    n_classes = DecisionTree.num_classes(data)
    num_trees = length(trees)

    trees =
      if opts[:reorder_trees] do
        for j <- 0..(n_classes - 1),
            i <- 0..(Integer.floor_div(length(trees), n_classes) - 1),
            do: Enum.at(trees, i * n_classes + j)
      else
        trees
      end

    # Number of classes each weak learner can predict
    # We infer from the shape of a leaf's :value key
    n_weak_learner_classes =
      trees
      |> hd()
      |> Tree.get_decision_values()
      |> hd()
      |> case do
        value when is_list(value) ->
          length(value)

        _value ->
          1
      end

    [h | t] = trees

    max_tree_depth =
      Enum.reduce(t, Tree.depth(h), fn tree, acc ->
        max(acc, Tree.depth(tree))
      end)

    perfect_trees = Enum.map(trees, &make_tree_perfect(&1, 0, max_tree_depth))

    {features, tt_tv} =
      Enum.map(perfect_trees, fn tree ->
        {tf, tt, tv} =
          Enum.reduce(Tree.bfs(tree), {[], [], []}, fn
            node, {tree_features, tree_thresholds, tree_values} ->
              case node do
                %Tree{left: nil, right: nil} ->
                  tree_values = [[node.value] | tree_values]
                  {tree_features, tree_thresholds, tree_values}

                %Tree{left: _left, right: _right} ->
                  tree_features = [node.value.feature | tree_features]
                  tree_thresholds = [node.value.threshold | tree_thresholds]
                  {tree_features, tree_thresholds, tree_values}
              end
          end)

        tf = Enum.reverse(tf)
        tt = Enum.reverse(tt)
        tv = Enum.reverse(tv)

        {tf, {tt, tv}}
      end)
      |> Enum.unzip()

    {thresholds, values} = Enum.unzip(tt_tv)

    # shape of {num_trees, 2 ** max_tree_depth - 1}
    features =
      features
      |> Nx.tensor()
      |> Nx.reshape({num_trees, @factor ** max_tree_depth - 1})

    # shape of {num_trees, 2 ** max_tree_depth - 1}
    thresholds =
      thresholds
      |> Nx.tensor()
      |> Nx.reshape({num_trees, @factor ** max_tree_depth - 1})

    # shape of {num_trees, 2 ** max_tree_depth}
    values =
      values
      |> Nx.tensor()
      |> Nx.reshape({:auto, n_weak_learner_classes})

    root_features = Nx.flatten(features[[.., 0]])

    root_thresholds = Nx.flatten(thresholds[[.., 0]])

    {features, thresholds} =
      Enum.reduce(1..(max_tree_depth - 1), {[], []}, fn depth, {all_nodes, all_biases} ->
        start = @factor ** depth - 1
        stop = @factor ** (depth + 1) - 2

        n = Nx.flatten(features[[.., start..stop]])

        b = Nx.flatten(thresholds[[.., start..stop]])

        {[n | all_nodes], [b | all_biases]}
      end)

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
  deftransform forward(x, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :custom_forward,
        :root_features,
        :root_thresholds,
        :condition,
        :indices,
        :num_trees,
        :n_classes,
        :thresholds,
        :features,
        :max_tree_depth,
        :values
      ])

    case opts[:custom_forward] do
      value when value != nil ->
        opts[:custom_forward].(x, opts)

      _ ->
        prev_indices =
          x
          |> Nx.take(opts[:root_features], axis: 1)
          |> opts[:condition].(opts[:root_thresholds])
          |> Nx.add(opts[:indices])
          |> Nx.reshape({:auto})

        prev_indices =
          Enum.zip_reduce([opts[:features], opts[:thresholds]], prev_indices, fn elems, acc ->
            {nodes, biases} = elems |> List.to_tuple()
            gather_indices = Nx.take(nodes, acc) |> Nx.reshape({:auto, opts[:num_trees]})
            features = Nx.take_along_axis(x, gather_indices, axis: 1) |> Nx.reshape({:auto})

            acc
            |> Nx.multiply(@factor)
            |> Nx.add(opts[:condition].(features, Nx.take(biases, acc)))
          end)

        Nx.take(opts[:values], prev_indices)
        |> Nx.reshape({:auto, opts[:num_trees], opts[:n_classes]})
    end
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

  defp make_tree_perfect(tree, current_depth, max_depth) do
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
