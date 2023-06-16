defmodule Mockingjay.Strategies.TreeTraversal do
  import Nx.Defn
  alias Mockingjay.Tree
  alias Mockingjay.DecisionTree
  @behaviour Mockingjay.Strategy

  @impl true
  def init(ensemble, opts \\ []) do
    opts = Keyword.validate!(opts, [:forward, :aggregate, :post_transform])
    trees = DecisionTree.trees(ensemble)
    condition = DecisionTree.condition(ensemble)
    n_classes = DecisionTree.num_classes(ensemble)
    num_trees = length(trees)
    # Number of classes each weak learner can predict
    # TODO: This is currently always 1, but could be more
    n_weak_learner_classes = 1

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
        # The trees are traversed BFS but node indices are assigned DFS
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
                tree_values = tree_values ++ [node.value]
                {tree_lefts, tree_rights, tree_features, tree_thresholds, tree_values}

              %Tree{left: left, right: right} ->
                tree_lefts = tree_lefts ++ [id_to_index[left.id]]
                tree_rights = tree_rights ++ [id_to_index[right.id]]
                tree_features = tree_features ++ [node.value.feature]
                tree_thresholds = tree_thresholds ++ [node.value.threshold]
                tree_values = tree_values ++ [-1]
                {tree_lefts, tree_rights, tree_features, tree_thresholds, tree_values}
            end
          end)

        tl = Nx.tensor(tl) |> Nx.pad(0, [{0, num_nodes - length(tl), 0}])
        tr = Nx.tensor(tr) |> Nx.pad(0, [{0, num_nodes - length(tr), 0}])
        tf = Nx.tensor(tf) |> Nx.pad(0, [{0, num_nodes - length(tf), 0}])
        tt = Nx.tensor(tt) |> Nx.pad(0, [{0, num_nodes - length(tt), 0}])
        # TODO - For categorical leaf values, we need to pad with n_classes
        tv = Nx.tensor(tv) |> Nx.pad(0, [{0, num_nodes - length(tv), 0}])

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

    nodes_offset = Nx.iota({num_trees}) |> Nx.multiply(num_nodes)

    forward_args =
      if opts[:forward] do
        [custom_forward: opts[:forward]]
      else
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
        :condition
      ])

    if opts[:custom_forward] do
      opts[:custom_forward].(x, opts)
    else
      max_tree_depth = opts[:max_tree_depth]
      num_trees = opts[:num_trees]
      n_classes = opts[:n_classes]
      nodes_offset = opts[:nodes_offset]
      lefts = opts[:lefts]
      rights = opts[:rights]
      features = opts[:features]
      thresholds = opts[:thresholds]
      values = opts[:values]
      condition = opts[:condition]

      batch_size = Nx.axis_size(x, 0)

      indexes =
        nodes_offset
        |> Nx.broadcast({batch_size, num_trees})
        |> Nx.reshape({:auto})

      indexes =
        Enum.reduce(1..max_tree_depth, indexes, fn _, tree_nodes ->
          feature_nodes = Nx.take(features, tree_nodes) |> Nx.reshape({:auto, num_trees})
          feature_values = Nx.take_along_axis(x, feature_nodes, axis: 1)
          local_thresholds = Nx.take(thresholds, tree_nodes) |> Nx.reshape({:auto, num_trees})
          local_lefts = Nx.take(lefts, tree_nodes) |> Nx.reshape({:auto, num_trees})
          local_rights = Nx.take(rights, tree_nodes) |> Nx.reshape({:auto, num_trees})

          Nx.select(
            condition.(feature_values, local_thresholds),
            local_lefts,
            local_rights
          )
          |> Nx.add(nodes_offset)
          |> Nx.reshape({:auto})
        end)

      values
      |> Nx.take(indexes)
      |> Nx.reshape({:auto, num_trees, n_classes})
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
