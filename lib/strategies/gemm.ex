defmodule Mockingjay.Strategies.GEMM do
  import Nx.Defn
  alias Mockingjay.Tree

  # Leaves are ordered as DFS rather than BFS that internal nodes are
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

  defn forward(
         x,
         hidden_one_size,
         hidden_two_size,
         hidden_three_size,
         mat_A,
         mat_B,
         mat_C,
         mat_D,
         mat_E,
         n_trees,
         condition
       ) do
    x
    |> Nx.multiply(mat_A)
    |> Mockingjay.Strategies.cond_to_fun(condition).(mat_B)
    |> Nx.reshape({n_trees, hidden_one_size, :auto})
    |> Nx.multiply(mat_C)
    |> Nx.reshape({n_trees * hidden_two_size, :auto})
    |> Nx.equal(mat_D)
    |> Nx.reshape({n_trees, hidden_two_size, :auto})
    |> Nx.multiply(mat_E)
    |> Nx.reshape({n_trees, hidden_three_size, :auto})
  end

  def generate_matrix_A(trees, num_features, hidden_one_size) do
    n_trees = length(trees)
    mat_A = Nx.broadcast(0, {n_trees, num_features, hidden_one_size})

    mat_A_indices =
      Enum.flat_map(
        Enum.with_index(trees),
        fn {tree, tree_index} ->
          Enum.with_index(Tree.get_decision_values(tree), fn node, node_index ->
            [tree_index, node.feature, node_index]
          end)
        end
      )
      |> Nx.tensor()

    mat_A =
      Nx.indexed_put(
        mat_A,
        mat_A_indices,
        Nx.broadcast(1, {mat_A_indices.shape |> elem(0)})
      )
  end

  def generate_matrix_B(trees, hidden_one_size) do
    n_trees = length(trees)

    Nx.indexed_put(
      Nx.broadcast(0, {n_trees, hidden_one_size}),
      Nx.tensor(
        Enum.flat_map(
          Enum.with_index(trees),
          fn {tree, index} ->
            Enum.with_index(Tree.get_decision_nodes(tree), fn _node, node_index ->
              [index, node_index]
            end)
          end
        )
      ),
      Nx.tensor(
        Enum.flat_map(trees, fn tree ->
          Enum.map(Tree.get_decision_values(tree), fn node -> node.threshold end)
        end)
      )
    )
  end

  def generate_matrix_C(trees, hidden_one_size, hidden_two_size) do
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

            [tree_index, internal_index, leaf_index, truth_value]
          end)
        end)
      end)
      |> Nx.tensor()

    # Gets the tensor of 'truth values'
    axis_size = Nx.axis_size(child_matrix, -1)
    updates = Nx.transpose(child_matrix)[-1]
    indices = Nx.slice_along_axis(child_matrix, 0, axis_size - 1, axis: -1)

    Nx.indexed_put(
      Nx.broadcast(0, {n_trees, hidden_one_size, hidden_two_size}),
      indices,
      updates
    )
  end

  def generate_matrix_D(trees, hidden_two_size) do
    n_trees = length(trees)

    indices =
      Enum.flat_map(Enum.with_index(trees), fn {tree, index} ->
        Enum.with_index(Tree.get_leaf_nodes(tree), fn _node, node_index ->
          [index, node_index]
        end)
      end)
      |> Nx.tensor()

    updates =
      Enum.flat_map(
        trees,
        &get_leaf_left_depths(&1)
      )
      |> Nx.tensor()

    Nx.indexed_put(Nx.broadcast(0, {n_trees, hidden_two_size}), indices, updates)
  end

  def generate_matrix_E(trees, hidden_two_size, hidden_three_size) do
    n_trees = length(trees)

    indices =
      if hidden_three_size == 1 do
        Enum.flat_map(
          Enum.with_index(trees),
          fn {tree, index} ->
            Enum.with_index(Tree.get_decision_nodes(tree), fn _node, node_index ->
              [index, node_index]
            end)
          end
        )
        |> Nx.tensor()
      else
        Enum.flat_map(
          Enum.with_index(trees),
          fn {tree, index} ->
            Enum.with_index(Tree.get_leaf_values(tree), fn value, node_index ->
              [index, node_index, value]
            end)
          end
        )
        |> Nx.tensor()
      end

    updates =
      if hidden_three_size == 1 do
        Nx.tensor(
          Enum.flat_map(trees, fn tree ->
            Tree.get_leaf_values(tree)
          end)
        )
      else
        Nx.broadcast(1, {indices.shape |> elem(0)})
      end

    Nx.indexed_put(
      Nx.broadcast(0, {n_trees, hidden_two_size, hidden_three_size}),
      indices,
      updates
    )
  end

  def compile(ensemble, opts \\ []) do
    trees = DecisionTree.trees(ensemble)
    num_features = DecisionTree.num_features(ensemble)
    condition = DecisionTree.condition(ensemble)

    n_classes =
      if DecisionTree.output_type(ensemble) == :classification do
        DecisionTree.num_classes(ensemble)
      else
        1
      end

    {hidden_one_size, hidden_two_size} =
      Enum.reduce(trees, {0, 0}, fn tree, {h1, h2} ->
        {max(h1, length(Tree.get_decision_nodes(tree))),
         max(h2, length(Tree.get_leaf_nodes(tree)))}
      end)

    hidden_three_size = n_classes

    n_trees = length(ensemble)
    mat_A = generate_matrix_A(trees, num_features, hidden_one_size)
    mat_B = generate_matrix_B(trees, hidden_one_size)
    mat_C = generate_matrix_C(trees, hidden_one_size, hidden_two_size)
    mat_D = generate_matrix_D(trees, hidden_two_size)
    mat_E = generate_matrix_E(trees, hidden_two_size, hidden_three_size)

    &forward(
      &1,
      hidden_one_size,
      hidden_two_size,
      hidden_three_size,
      mat_A,
      mat_B,
      mat_C,
      mat_D,
      mat_E,
      n_trees,
      condition
    )
  end
end
