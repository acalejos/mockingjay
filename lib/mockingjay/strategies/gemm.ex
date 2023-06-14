defmodule Mockingjay.Strategies.GEMM do
  import Nx.Defn

  alias Mockingjay.Tree
  alias Mockingjay.DecisionTree
  alias Mockingjay.Model

  @behaviour Mockingjay.Strategy

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

  deftransformp aggregate(x) do
    x
    |> Nx.sum(axes: [0])
    |> Nx.transpose()
  end

  defnp forward(x, mat_A, mat_B, mat_C, mat_D, mat_E, condition, opts \\ []) do
    opts = keyword!(opts, [:n_trees, :hidden_one_size, :hidden_two_size, :hidden_three_size])

    n_trees = opts[:n_trees]
    hidden_one_size = opts[:hidden_one_size]
    hidden_two_size = opts[:hidden_two_size]
    hidden_three_size = opts[:hidden_three_size]

    mat_A
    |> Nx.dot([1], x, [1])
    |> condition.(mat_B)
    |> Nx.reshape({n_trees, hidden_one_size, :auto})
    |> then(&Nx.dot(mat_C, [2], [0], &1, [1], [0]))
    |> Nx.reshape({n_trees * hidden_two_size, :auto})
    |> Nx.equal(mat_D)
    |> Nx.reshape({n_trees, hidden_two_size, :auto})
    |> then(&Nx.dot(mat_E, [2], [0], &1, [1], [0]))
    |> Nx.reshape({n_trees, hidden_three_size, :auto})
  end

  # TODO The generation of matrices can likely be done in 1 pass rather than a different pass for each

  defp generate_matrices_AB(trees, num_features, hidden_one_size) do
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

    a_zeros = Nx.broadcast(0, {n_trees, hidden_one_size, num_features})
    b_zeros = Nx.slice_along_axis(a_zeros, 0, 1, axis: -1) |> Nx.squeeze(axes: [-1])

    a = Nx.indexed_put(a_zeros, a_indices, a_updates)

    b = Nx.indexed_put(b_zeros, b_indices, b_updates)

    num_rows = n_trees * hidden_one_size
    {Nx.reshape(a, {num_rows, num_features}), Nx.reshape(b, {num_rows, 1})}
  end

  defp generate_matrix_C(trees, hidden_one_size, hidden_two_size) do
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
      Nx.broadcast(0, {n_trees, hidden_two_size, hidden_one_size}),
      indices,
      updates
    )
  end

  defp generate_matrices_DE(trees, hidden_two_size, hidden_three_size) do
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
    d_zero = Nx.broadcast(0, {n_trees, hidden_two_size})

    d = Nx.indexed_put(d_zero, d_indices, d_updates)

    e_updates = Nx.tensor(updates_list)
    e_zero = Nx.broadcast(0, {n_trees, hidden_three_size, hidden_two_size})

    e = Nx.indexed_put(e_zero, e_indices, e_updates)

    {Nx.reshape(d, {:auto, 1}), e}
  end

  @impl true
  def compile(ensemble, _opts \\ []) do
    trees = DecisionTree.trees(ensemble)
    num_features = DecisionTree.num_features(ensemble)
    condition = DecisionTree.condition(ensemble)

    # Overall number of classes for classification, 1 for regression
    n_classes = DecisionTree.num_classes(ensemble)

    # Number of classes each weak learner can predict
    # TODO: This is currently always 1, but could be more
    n_weak_learner_classes = 1

    {hidden_one_size, hidden_two_size} =
      Enum.reduce(trees, {0, 0}, fn tree, {h1, h2} ->
        {max(h1, length(Tree.get_decision_nodes(tree))),
         max(h2, length(Tree.get_leaf_nodes(tree)))}
      end)

    hidden_three_size = n_weak_learner_classes

    n_trees = length(trees)

    {mat_A, mat_B} = generate_matrices_AB(trees, num_features, hidden_one_size)
    mat_C = generate_matrix_C(trees, hidden_one_size, hidden_two_size)
    {mat_D, mat_E} = generate_matrices_DE(trees, hidden_two_size, hidden_three_size)

    aggregation =
      cond do
        n_classes > 1 and n_trees > 1 ->
          n_gbdt_classes = if n_classes > 2, do: n_classes, else: 1
          n_trees_per_class = trunc(n_trees / n_gbdt_classes)

          &ensemble_aggregate(
            &1,
            n_gbdt_classes,
            n_trees_per_class
          )

        n_classes > 1 and n_trees == 1 ->
          &aggregate(&1)

        true ->
          raise "Unknown output type"
      end

    post_transform =
      Mockingjay.Strategy.infer_post_transform(n_classes)
      |> Mockingjay.Strategy.post_transform_to_func()

    model = %Model{
      forward:
        &forward(
          &1,
          mat_A,
          mat_B,
          mat_C,
          mat_D,
          mat_E,
          Mockingjay.Strategy.cond_to_fun(condition),
          n_trees: n_trees,
          hidden_one_size: hidden_one_size,
          hidden_two_size: hidden_two_size,
          hidden_three_size: hidden_three_size
        ),
      aggregate: aggregation,
      post_transform: post_transform
    }

    model
  end
end
