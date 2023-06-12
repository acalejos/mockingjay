defmodule Mockingjay.Strategies.GEMM do
  import Nx.Defn

  alias Mockingjay.Tree
  alias Mockingjay.DecisionTree

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

  defn forward(x, mat_A, mat_B, mat_C, mat_D, mat_E, condition, opts \\ []) do
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
    |> Nx.squeeze()
  end

  # TODO The generation of matrices can likely be done in 1 pass rather than a different pass for each

  def generate_matrices_AB(trees, num_features, hidden_one_size) do
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

    dbg({a_zeros, a_indices, a_updates, b_zeros, b_indices, b_updates})
    b = Nx.indexed_put(b_zeros, b_indices, b_updates)

    num_rows = n_trees * hidden_one_size
    {Nx.reshape(a, {num_rows, num_features}), Nx.reshape(b, {num_rows, 1})}
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
    |> Nx.reshape({:auto, 1})
  end

  def generate_matrix_E(trees, hidden_two_size, hidden_three_size) do
    n_trees = length(trees)

    indices =
      if hidden_three_size == 1 do
        Enum.flat_map(
          Enum.with_index(trees),
          fn {tree, index} ->
            Enum.with_index(Tree.get_leaf_values(tree), fn _node, node_index ->
              [index, 0, node_index]
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
      Nx.broadcast(0, {n_trees, hidden_three_size, hidden_two_size}),
      indices,
      updates
    )
  end

  def compile(ensemble, _opts \\ []) do
    trees = DecisionTree.trees(ensemble)
    num_features = DecisionTree.num_features(ensemble)
    condition = DecisionTree.condition(ensemble)

    n_classes = 1
    # TODO : Infer this from shape of leaf node values
    # if DecisionTree.output_type(ensemble) == :classification do
    #   DecisionTree.num_classes(ensemble)
    # else
    #   1
    # end

    {hidden_one_size, hidden_two_size} =
      Enum.reduce(trees, {0, 0}, fn tree, {h1, h2} ->
        {max(h1, length(Tree.get_decision_nodes(tree))),
         max(h2, length(Tree.get_leaf_nodes(tree)))}
      end)

    hidden_three_size = n_classes

    # TODO
    # Setup as many matrices as possible in one pass
    # {a_indices, {b_indices, c_updates}, d_matrix, {d_indices, d_updates}, {e_indices, e_updates}} =
    #   Enum.reduce(
    #     Enum.with_index(trees),
    #     {[], {[], []}, [], {[], []}, {[], []}},
    #     fn {tree, tree_index}, {ai, {bi, bu}, c, {di, du}, {ei, eu}} ->
    #       du = du ++ get_leaf_left_depths(tree)

    #       Enum.reduce(Enum.with_index(Tree.get_decision_nodes(tree)), fn {internal_node,
    #                                                                       internal_index} ->
    #         ai = ai ++ [tree_index, internal_index, internal_node.value.feature]
    #         bi = bi ++ [tree_index, internal_index]
    #         bu = bu ++ [internal_node.value.threshold]

    #         Enum.reduce(Tree.get_leaf_nodes(tree), fn leaf_node, leaf_index ->
    #           truth_value =
    #             cond do
    #               Tree.is_child(internal_node.left, leaf_node.id) -> 1
    #               Tree.is_child(internal_node.right, leaf_node.id) -> -1
    #               true -> 0
    #             end

    #           c = c ++ [tree_index, leaf_index, internal_index, truth_value]
    #           di = di ++ [tree_index, leaf_index]

    #           ei =
    #             ei ++
    #               if hidden_three_size == 1 do
    #                 [tree_index, 0, leaf_index]
    #               else
    #                 [tree_index, leaf_index, leaf_node.value]
    #                 eu = eu ++ []
    #               end

    #           eu =
    #             eu ++
    #               if hidden_three_size == 1 do
    #                 [leaf_node.value]
    #               else
    #                 [1]
    #               end

    #           {ai, {bi, bu}, c, {di, du}, {ei, eu}}
    #         end)
    #       end)
    #     end
    #   )

    n_trees = length(trees)

    {mat_A, mat_B} = generate_matrices_AB(trees, num_features, hidden_one_size)
    # IO.puts("mat_A: #{inspect(mat_A)}")
    # IO.puts("mat_B: #{inspect(mat_B)}")
    mat_C = generate_matrix_C(trees, hidden_one_size, hidden_two_size)
    # IO.puts("mat_C: #{inspect(mat_C)}")
    mat_D = generate_matrix_D(trees, hidden_two_size)
    # IO.puts("mat_D: #{inspect(mat_D)}")
    mat_E = generate_matrix_E(trees, hidden_two_size, hidden_three_size)
    # IO.puts("mat_E: #{inspect(mat_E)}")
    # IO.puts("n_trees: #{inspect(n_trees)}")
    # IO.puts("hidden_one_size: #{inspect(hidden_one_size)}")
    # IO.puts("hidden_two_size: #{inspect(hidden_two_size)}")
    # IO.puts("hidden_three_size: #{inspect(hidden_three_size)}")

    &forward(
      &1,
      mat_A,
      mat_B,
      mat_C,
      mat_D,
      mat_E,
      Mockingjay.Strategies.cond_to_fun(condition),
      n_trees: n_trees,
      hidden_one_size: hidden_one_size,
      hidden_two_size: hidden_two_size,
      hidden_three_size: hidden_three_size
    )
  end
end
