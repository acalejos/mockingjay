defmodule Mockingjay.Strategies.GEMM do
  import Nx.Defn

  alias Mockingjay.Tree
  alias Mockingjay.DecisionTree

  @behaviour Mockingjay.Strategy

  @impl true
  def init(ensemble, opts \\ []) do
    opts = Keyword.validate!(opts, [:forward, :aggregate, :post_transform])
    trees = DecisionTree.trees(ensemble)
    num_features = DecisionTree.num_features(ensemble)
    condition = DecisionTree.condition(ensemble)

    # Overall number of classes for classification, 1 for regression
    n_classes = DecisionTree.num_classes(ensemble)

    # Number of classes each weak learner can predict
    # TODO: This is currently always 1, but could be more
    n_weak_learner_classes = 1

    {max_decision_nodes, max_leaf_nodes} =
      Enum.reduce(trees, {0, 0}, fn tree, {h1, h2} ->
        {max(h1, length(Tree.get_decision_nodes(tree))),
         max(h2, length(Tree.get_leaf_nodes(tree)))}
      end)

    hidden_three_size = n_weak_learner_classes

    n_trees = length(trees)

    {mat_A, mat_B} = generate_matrices_AB(trees, num_features, max_decision_nodes)
    mat_C = generate_matrix_C(trees, max_decision_nodes, max_leaf_nodes)
    {mat_D, mat_E} = generate_matrices_DE(trees, max_leaf_nodes, hidden_three_size)

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
          n_trees: n_trees
        ]
      end

    forward_args =
      if opts[:forward] do
        [custom_forward: opts[:forward]]
      else
        [
          mat_A: mat_A,
          mat_B: mat_B,
          mat_C: mat_C,
          mat_D: mat_D,
          mat_E: mat_E,
          condition: Mockingjay.Strategy.cond_to_fun(condition),
          n_trees: n_trees,
          max_decision_nodes: max_decision_nodes,
          max_leaf_nodes: max_leaf_nodes,
          hidden_three_size: hidden_three_size
        ]
      end

    {forward_args, aggregate_args, post_transform_args}
  end

  @impl true
  def forward(x, opts \\ []) do
    opts =
      Keyword.validate!(opts, [
        :mat_A,
        :mat_B,
        :mat_C,
        :mat_D,
        :mat_E,
        :condition,
        :n_trees,
        :max_decision_nodes,
        :max_leaf_nodes,
        :hidden_three_size,
        :custom_forward
      ])

    if opts[:custom_forward] do
      opts[:custom_forward].(x)
    else
      _forward(x, opts)
    end
  end

  defn _forward(x, opts \\ []) do
    opts =
      keyword!(opts, [
        :mat_A,
        :mat_B,
        :mat_C,
        :mat_D,
        :mat_E,
        :condition,
        :n_trees,
        :max_decision_nodes,
        :max_leaf_nodes,
        :hidden_three_size
      ])

    mat_A = opts[:mat_A]
    mat_B = opts[:mat_B] |> Nx.as_type(:f64)
    mat_C = opts[:mat_C]
    mat_D = opts[:mat_D]
    mat_E = opts[:mat_E] |> Nx.as_type(:f64)
    condition = opts[:condition]
    n_trees = opts[:n_trees]
    max_decision_nodes = opts[:max_decision_nodes]
    max_leaf_nodes = opts[:max_leaf_nodes]
    hidden_three_size = opts[:hidden_three_size]

    mat_A
    |> Nx.dot([1], x, [1])
    |> condition.(mat_B)
    |> Nx.reshape({n_trees, max_decision_nodes, :auto})
    |> then(&Nx.dot(mat_C, [2], [0], &1, [1], [0]))
    |> Nx.reshape({n_trees * max_leaf_nodes, :auto})
    |> Nx.equal(mat_D)
    |> Nx.reshape({n_trees, max_leaf_nodes, :auto})
    |> then(&Nx.dot(mat_E, [2], [0], &1, [1], [0]))
    |> Nx.reshape({n_trees, hidden_three_size, :auto})
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

  # Leaves are ordered as DFS rather than BFS that internal nodes are
  # TO-DO: make TCOptimizable
  defp get_leaf_left_depths(root) do
    _get_leaf_left_depths(root, 0)
  end

  defp _get_leaf_left_depths(root, depth) do
    case root do
      %{left: nil, right: nil} ->
        [depth]

      %{left: left, right: right} ->
        _get_leaf_left_depths(left, depth + 1) ++ _get_leaf_left_depths(right, depth)
    end
  end

  deftransformp ensemble_aggregate(x, n_gbdt_classes, n_trees_per_class) do
    x
    |> Nx.squeeze()
    |> Nx.transpose()
    |> Nx.reshape({:auto, n_gbdt_classes, n_trees_per_class})
    |> Nx.sum(axes: [2])
  end

  deftransformp _aggregate(x) do
    x
    |> Nx.sum(axes: [0])
    |> Nx.transpose()
  end

  defp generate_matrices_AB(trees, num_features, max_decision_nodes) do
    n_trees = length(trees)

    {indices_list, updates_list} =
      trees
      |> Enum.with_index()
      |> Enum.flat_map(fn {tree, tree_index} ->
        Enum.with_index(Tree.get_decision_values(tree), fn value, node_index ->
          {[tree_index, node_index, value.feature], value.threshold}
        end)
      end)
      |> Enum.unzip()

    a_indices = Nx.tensor(indices_list)
    b_indices = a_indices[[.., 0..1]]

    a_updates = Nx.broadcast(1, {Nx.axis_size(a_indices, 0)})
    b_updates = Nx.tensor(updates_list)

    a_zeros = Nx.broadcast(0, {n_trees, max_decision_nodes, num_features})
    b_zeros = Nx.slice_along_axis(a_zeros, 0, 1, axis: -1) |> Nx.squeeze(axes: [-1])

    a = Nx.indexed_put(a_zeros, a_indices, a_updates)

    b = Nx.indexed_put(b_zeros, b_indices, b_updates)

    num_rows = n_trees * max_decision_nodes
    {Nx.reshape(a, {num_rows, num_features}), Nx.reshape(b, {num_rows, 1})}
  end

  defp generate_matrix_C(trees, max_decision_nodes, max_leaf_nodes) do
    n_trees = length(trees)

    child_matrix =
      Enum.flat_map(Enum.with_index(trees), fn {tree, tree_index} ->
        Enum.flat_map(Enum.with_index(Tree.get_decision_nodes(tree)), fn {internal_node,
                                                                          internal_index} ->
          Enum.with_index(Tree.get_leaf_nodes(tree), fn leaf_node, leaf_index ->
            truth_value =
              cond do
                Tree.is_child(internal_node.left, leaf_node.id) -> 1
                Tree.is_child(internal_node.right, leaf_node.id) -> -1
                true -> 0
              end

            [tree_index, leaf_index, internal_index, truth_value]
          end)
        end)
      end)
      |> Nx.tensor()

    # Gets the tensor of 'truth values'
    axis_size = Nx.axis_size(child_matrix, -1)
    updates = Nx.transpose(child_matrix)[-1]
    indices = Nx.slice_along_axis(child_matrix, 0, axis_size - 1, axis: -1)

    Nx.indexed_put(
      Nx.broadcast(0, {n_trees, max_leaf_nodes, max_decision_nodes}),
      indices,
      updates
    )
  end

  defp generate_matrices_DE(trees, max_leaf_nodes, hidden_three_size) do
    n_trees = length(trees)

    {indices_list, updates_list} =
      trees
      |> Enum.with_index()
      |> Enum.flat_map(fn {tree, index} ->
        Enum.with_index(Tree.get_leaf_nodes(tree), fn node, node_index ->
          if hidden_three_size == 1 do
            {[index, 0, node_index], node.value}
          else
            {[index, trunc(node.value), node_index], 1}
          end
        end)
      end)
      |> Enum.unzip()

    e_indices = Nx.tensor(indices_list)

    d_indices = Nx.take(e_indices, Nx.tensor([0, 2]), axis: 1)

    d_updates = trees |> Enum.flat_map(&get_leaf_left_depths/1) |> Nx.tensor()
    d_zero = Nx.broadcast(0, {n_trees, max_leaf_nodes})

    d = Nx.indexed_put(d_zero, d_indices, d_updates)

    e_updates = Nx.tensor(updates_list)
    e_zero = Nx.broadcast(0, {n_trees, hidden_three_size, max_leaf_nodes})

    e = Nx.indexed_put(e_zero, e_indices, e_updates)

    {Nx.reshape(d, {:auto, 1}), e}
  end
end
