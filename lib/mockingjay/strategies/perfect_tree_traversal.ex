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
    opts = Keyword.validate!(opts, reorder_trees: true)
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

    features = Enum.reverse(features) |> List.to_tuple()
    thresholds = Enum.reverse(thresholds) |> List.to_tuple()

    nt = @factor * num_trees

    indices = 0..(nt - 1)//2 |> Enum.into([]) |> Nx.tensor(type: :s64)

    arg = %{
      root_features: root_features,
      root_thresholds: root_thresholds,
      features: features,
      thresholds: thresholds,
      values: values,
      indices: indices
    }

    opts = [
      num_trees: num_trees,
      max_tree_depth: max_tree_depth,
      condition: Mockingjay.Strategy.cond_to_fun(condition),
      n_classes: n_classes
    ]

    {arg, opts}
  end

  @impl true
  deftransform forward(x, {arg, opts}) do
    opts =
      Keyword.validate!(opts, [
        :condition,
        :num_trees,
        :n_classes,
        :max_tree_depth
      ])

    _forward(x, arg, opts)
  end

  defnp _forward(x, arg, opts) do
    %{
      root_features: root_features,
      root_thresholds: root_thresholds,
      features: features,
      thresholds: thresholds,
      values: values,
      indices: indices
    } = arg

    prev_indices =
      x
      |> Nx.take(root_features, axis: 1)
      |> opts[:condition].(root_thresholds)
      |> Nx.add(indices)
      |> Nx.reshape({:auto})
      |> forward_reduce_features(x, features, thresholds, opts)

    Nx.take(values |> print_value(), prev_indices)
    |> Nx.reshape({:auto, opts[:num_trees], opts[:n_classes]})
  end

  deftransformp forward_reduce_features(prev_indices, x, features, thresholds, opts \\ []) do
    Enum.zip_reduce(
      Tuple.to_list(features),
      Tuple.to_list(thresholds),
      prev_indices,
      fn nodes, biases, acc ->
        _inner_reduce(x, nodes, biases, acc, opts)
      end
    )
  end

  defnp _inner_reduce(x, nodes, biases, acc, opts \\ []) do
    gather_indices =
      nodes |> print_value() |> Nx.take(acc) |> Nx.reshape({:auto, opts[:num_trees]})

    features = Nx.take_along_axis(x, gather_indices, axis: 1) |> Nx.reshape({:auto})

    acc
    |> print_value()
    |> Nx.multiply(@factor)
    |> Nx.add(opts[:condition].(features, Nx.take(biases |> print_value(), acc)))
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
end
