defmodule Mockingjay.Strategies.GEMM do
  import Nx.Defn
  alias Mockingjay.Tree

  # Leaves are ordered as DFS rather than BFS that internal nodes are
  def get_leaf_left_depths(root) do
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

  defp get_child_indices_and_updates(trees) do
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
    {indices, updates}
  end

  defn forward(
         x,
         hidden_one_size,
         hidden_two_size,
         hidden_three_size,
         weight1,
         bias1,
         weight2,
         bias2,
         weight3,
         n_trees,
         condition
       ) do
    x
    |> Nx.multiply(weight1)
    |> Mockingjay.Strategies.cond_to_fun(condition).(bias1)
    |> Nx.reshape({n_trees, hidden_one_size, :auto})
    |> Nx.multiply(weight2)
    |> Nx.reshape({n_trees * hidden_two_size, :auto})
    |> Nx.equal(bias2)
    |> Nx.reshape({n_trees, hidden_two_size, :auto})
    |> Nx.multiply(weight3)
    |> Nx.reshape({n_trees, hidden_three_size, :auto})
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

    IO.puts("hidden_one_size: #{inspect(hidden_one_size)}")
    IO.puts("hidden_two_size: #{inspect(hidden_two_size)}")

    hidden_three_size = n_classes

    n_trees = length(ensemble)
    weight_1 = Nx.broadcast(0, {n_trees, num_features, hidden_one_size})
    bias_1 = Nx.broadcast(0, {n_trees, hidden_one_size})
    weight_2 = Nx.broadcast(0, {n_trees, hidden_one_size, hidden_two_size})
    bias_2 = Nx.broadcast(0, {n_trees, hidden_two_size})
    weight_3 = Nx.broadcast(0, {n_trees, hidden_two_size, hidden_three_size})

    # These correspond to Matrix A in the Hummingbird paper
    w1_indices =
      Enum.flat_map(
        Enum.with_index(trees),
        fn {tree, tree_index} ->
          Enum.with_index(Tree.get_decision_values(tree), fn node, node_index ->
            [tree_index, node.feature, node_index]
          end)
        end
      )
      |> Nx.tensor()

    weight_1 =
      Nx.indexed_put(
        weight_1,
        w1_indices,
        Nx.broadcast(1, {w1_indices.shape |> elem(0)})
      )

    IO.puts("weight_1: #{inspect(weight_1)}")

    # These correspond to Matrix B in the Hummingbird paper
    bias_1 =
      Nx.indexed_put(
        bias_1,
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

    IO.puts("bias_1: #{inspect(bias_1)}")

    # These correspond to Matrix C in the Hummingbird paper
    {w2_indices, w2_updates} = get_child_indices_and_updates(trees)
    weight_2 = Nx.indexed_put(weight_2, w2_indices, w2_updates)
    IO.puts("weight_2: #{inspect(weight_2)}")

    # These correspond to Matrix D in the Hummingbird paper
    b2_indices =
      Enum.flat_map(Enum.with_index(trees), fn {tree, index} ->
        Enum.with_index(Tree.get_leaf_nodes(tree), fn _node, node_index ->
          [index, node_index]
        end)
      end)
      |> Nx.tensor()

    b2_updates =
      Enum.flat_map(
        trees,
        &get_leaf_left_depths(&1)
      )
      |> Nx.tensor()

    bias_2 = Nx.indexed_put(bias_2, b2_indices, b2_updates)
    IO.puts("bias_2: #{inspect(bias_2)}")

    # These correspond to Matrix E in the Hummingbird paper
    w3_indices =
      if n_classes == 1 do
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
      |> IO.inspect()

    w3_updates =
      if n_classes == 1 do
        Nx.tensor(
          Enum.flat_map(trees, fn tree ->
            Tree.get_leaf_values(tree)
          end)
        )
      else
        Nx.broadcast(1, {w3_indices.shape |> elem(0)})
      end

    weight_3 = Nx.indexed_put(weight_3, w3_indices, w3_updates)

    IO.puts("weight_3: #{inspect(weight_3)}")

    &forward(
      &1,
      hidden_one_size,
      hidden_two_size,
      hidden_three_size,
      weight_1,
      bias_1,
      weight_2,
      bias_2,
      weight_3,
      n_trees,
      condition
    )
  end
end
