defmodule Mockingjay.Strategies.GEMM do
  @moduledoc false
  import Nx.Defn

  alias Mockingjay.Tree
  alias Mockingjay.DecisionTree

  @behaviour Mockingjay.Strategy

  @impl true
  def init(ensemble, opts \\ []) do
    opts = Keyword.validate!(opts, reorder_trees: true)
    trees = DecisionTree.trees(ensemble)

    num_features = DecisionTree.num_features(ensemble)
    condition = DecisionTree.condition(ensemble)

    # Overall number of classes for classification, 1 for regression
    n_classes = DecisionTree.num_classes(ensemble)

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

    {max_decision_nodes, max_leaf_nodes} =
      Enum.reduce(trees, {0, 0}, fn tree, {h1, h2} ->
        {max(h1, length(Tree.get_decision_nodes(tree))),
         max(h2, length(Tree.get_leaf_nodes(tree)))}
      end)

    n_trees = length(trees)

    {mat_A, mat_B} = generate_matrices_AB(trees, num_features, max_decision_nodes)
    mat_C = generate_matrix_C(trees, max_decision_nodes, max_leaf_nodes)
    {mat_D, mat_E} = generate_matrices_DE(trees, max_leaf_nodes, n_weak_learner_classes)

    arg = %{
      mat_A: mat_A,
      mat_B: mat_B,
      mat_C: mat_C,
      mat_D: mat_D,
      mat_E: mat_E
    }

    opts = [
      condition: Mockingjay.Strategy.cond_to_fun(condition),
      n_trees: n_trees,
      max_decision_nodes: max_decision_nodes,
      max_leaf_nodes: max_leaf_nodes,
      n_weak_learner_classes: n_weak_learner_classes,
      n_classes: n_classes
    ]

    {arg, opts}
  end

  @impl true
  deftransform forward(x, {arg, opts}) do
    opts =
      Keyword.validate!(opts, [
        :condition,
        :n_trees,
        :n_classes,
        :max_decision_nodes,
        :max_leaf_nodes,
        :n_weak_learner_classes,
        :custom_forward
      ])

    _forward(x, arg, opts)
  end

  defnp _forward(x, arg, opts \\ []) do
    %{mat_A: mat_A, mat_B: mat_B, mat_C: mat_C, mat_D: mat_D, mat_E: mat_E} = arg

    condition = opts[:condition]
    n_trees = opts[:n_trees]
    n_classes = opts[:n_classes]
    max_decision_nodes = opts[:max_decision_nodes]
    max_leaf_nodes = opts[:max_leaf_nodes]
    n_weak_learner_classes = opts[:n_weak_learner_classes]

    mat_A
    |> Nx.dot([1], x, [1])
    |> condition.(mat_B)
    |> Nx.reshape({n_trees, max_decision_nodes, :auto})
    |> then(&Nx.dot(mat_C, [2], [0], &1, [1], [0]))
    |> Nx.reshape({n_trees * max_leaf_nodes, :auto})
    |> Nx.equal(mat_D)
    |> Nx.reshape({n_trees, max_leaf_nodes, :auto})
    |> then(&Nx.dot(mat_E, [2], [0], &1, [1], [0]))
    |> Nx.reshape({n_trees, n_weak_learner_classes, :auto})
    |> Nx.transpose()
    |> Nx.reshape({:auto, n_trees, n_classes})
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
                Tree.child?(internal_node.left, leaf_node.id) -> 1
                Tree.child?(internal_node.right, leaf_node.id) -> -1
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

  defp generate_matrices_DE(trees, max_leaf_nodes, n_weak_learner_classes) do
    n_trees = length(trees)

    {indices_list, updates_list} =
      trees
      |> Enum.with_index()
      |> Enum.flat_map(fn {tree, index} ->
        Enum.with_index(Tree.get_leaf_nodes(tree), fn node, node_index ->
          if n_weak_learner_classes == 1 do
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

    e_zero = Nx.broadcast(0, {n_trees, n_weak_learner_classes, max_leaf_nodes})

    e = Nx.indexed_put(e_zero, e_indices, e_updates)

    {Nx.reshape(d, {:auto, 1}), e}
  end
end
