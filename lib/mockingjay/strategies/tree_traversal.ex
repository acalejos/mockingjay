defmodule Mockingjay.Strategies.TreeTraversal do
  @moduledoc false
  import Nx.Defn
  alias Mockingjay.Tree
  alias Mockingjay.DecisionTree
  @behaviour Mockingjay.Strategy

  @impl true
  def init(ensemble, opts \\ []) do
    opts = Keyword.validate!(opts, reorder_trees: true)

    trees = DecisionTree.trees(ensemble)

    condition = DecisionTree.condition(ensemble)
    n_classes = DecisionTree.num_classes(ensemble)
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
      case trees |> hd |> Tree.get_decision_values() |> hd do
        value when is_list(value) ->
          length(value)

        _value ->
          1
      end

    num_nodes =
      Enum.reduce(trees, 0, fn tree, acc ->
        max(acc, length(Tree.bfs(tree)))
      end)

    max_tree_depth =
      Enum.reduce(trees, 0, fn tree, acc ->
        max(acc, Tree.depth(tree))
      end)

    {lefts, rights, features, thresholds, values} =
      trees
      |> Enum.reduce({[], [], [], [], []}, fn tree,
                                              {all_lefts, all_rights, all_features,
                                               all_thresholds, all_values} ->
        dfs_tree = Tree.dfs(tree)

        id_to_index =
          Enum.reduce(Enum.with_index(dfs_tree), %{}, fn {node, index}, acc ->
            Map.put_new(acc, node.id, index)
          end)

        {tl, tr, tf, tt, tv} =
          Enum.reduce(dfs_tree, {[], [], [], [], []}, fn node,
                                                         {tree_lefts, tree_rights, tree_features,
                                                          tree_thresholds, tree_values} ->
            case node do
              %Tree{left: nil, right: nil} ->
                tree_lefts = tree_lefts ++ [id_to_index[node.id]]
                tree_rights = tree_rights ++ [id_to_index[node.id]]
                tree_features = tree_features ++ [0]
                tree_thresholds = tree_thresholds ++ [0]
                current_value = if is_list(node.value), do: node.value, else: [node.value]
                tree_values = tree_values ++ [current_value]
                {tree_lefts, tree_rights, tree_features, tree_thresholds, tree_values}

              %Tree{left: left, right: right} ->
                tree_lefts = tree_lefts ++ [id_to_index[left.id]]
                tree_rights = tree_rights ++ [id_to_index[right.id]]
                tree_features = tree_features ++ [node.value.feature]
                tree_thresholds = tree_thresholds ++ [node.value.threshold]
                tree_values = tree_values ++ [[-1]]
                {tree_lefts, tree_rights, tree_features, tree_thresholds, tree_values}
            end
          end)

        tl = Nx.tensor(tl) |> Nx.pad(0, [{0, num_nodes - length(tl), 0}])
        tr = Nx.tensor(tr) |> Nx.pad(0, [{0, num_nodes - length(tr), 0}])
        tf = Nx.tensor(tf) |> Nx.pad(0, [{0, num_nodes - length(tf), 0}])
        tt = Nx.tensor(tt) |> Nx.pad(0, [{0, num_nodes - length(tt), 0}])
        tv = Nx.tensor(tv) |> Nx.pad(0, [{0, num_nodes - length(tv), 0}, {0, 0, 0}])

        {[tl | all_lefts], [tr | all_rights], [tf | all_features], [tt | all_thresholds],
         [tv | all_values]}
      end)

    lefts =
      Nx.stack(Enum.reverse(lefts))
      |> Nx.reshape({:auto})

    rights =
      Nx.stack(Enum.reverse(rights))
      |> Nx.reshape({:auto})

    features =
      Nx.stack(Enum.reverse(features))
      |> Nx.reshape({:auto})

    thresholds =
      Nx.stack(Enum.reverse(thresholds))
      |> Nx.reshape({:auto})

    values =
      Nx.stack(Enum.reverse(values))
      |> Nx.reshape({:auto, n_weak_learner_classes})

    nodes_offset =
      Nx.iota({1, num_trees}, type: :s64)
      |> Nx.multiply(num_nodes)

    [
      nodes_offset: nodes_offset,
      num_trees: num_trees,
      max_tree_depth: max_tree_depth,
      lefts: lefts,
      rights: rights,
      features: features,
      thresholds: thresholds,
      values: values,
      condition: Mockingjay.Strategy.cond_to_fun(condition),
      n_classes: n_classes
    ]
  end

  @impl true
  deftransform forward(x, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :custom_forward,
        :max_tree_depth,
        :num_trees,
        :n_classes,
        :nodes_offset,
        :lefts,
        :rights,
        :features,
        :thresholds,
        :values,
        :condition,
        unroll: false
      ])

    _forward(
      x,
      opts[:features],
      opts[:lefts],
      opts[:rights],
      opts[:thresholds],
      opts[:nodes_offset],
      opts[:values],
      opts
    )
  end

  defn _forward(x, features, lefts, rights, thresholds, nodes_offset, values, opts \\ []) do
    max_tree_depth = opts[:max_tree_depth]
    num_trees = opts[:num_trees]
    n_classes = opts[:n_classes]
    condition = opts[:condition]
    unroll = opts[:unroll]

    batch_size = Nx.axis_size(x, 0)

    indices =
      nodes_offset
      |> Nx.broadcast({batch_size, num_trees})
      |> Nx.reshape({:auto})

    {indices, _} =
      while {tree_nodes = indices, {features, lefts, rights, thresholds, nodes_offset, x}},
            _ <- 1..max_tree_depth,
            unroll: unroll do
        feature_nodes = Nx.take(features, tree_nodes) |> Nx.reshape({:auto, num_trees})
        feature_values = Nx.take_along_axis(x, feature_nodes, axis: 1)
        local_thresholds = Nx.take(thresholds, tree_nodes) |> Nx.reshape({:auto, num_trees})
        local_lefts = Nx.take(lefts, tree_nodes) |> Nx.reshape({:auto, num_trees})
        local_rights = Nx.take(rights, tree_nodes) |> Nx.reshape({:auto, num_trees})

        result =
          Nx.select(
            condition.(feature_values, local_thresholds),
            local_lefts,
            local_rights
          )
          |> Nx.add(nodes_offset)
          |> Nx.reshape({:auto})

        {result, {features, lefts, rights, thresholds, nodes_offset, x}}
      end

    values
    |> Nx.take(indices)
    |> Nx.reshape({:auto, num_trees, n_classes})
  end
end
