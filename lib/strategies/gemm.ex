defmodule Mockingjay.Strategies.GEMM do
  import Nx.Defn
  alias Mockingjay.Tree

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

  def compile(ensemble) do
    trees = Mockingjay.DecisionTre.trees(ensemble)
    features = Mockingjay.DecisionTree.features(ensemble)
    condition = Mockingjay.DecisionTree.condition(ensemble)

    num_inner_nodes =
      {hidden_one_size, hidden_two_size} =
      Enum.reduce(trees, {0, 0}, fn tree, {h1, h2} ->
        {max(h1, length(Tree.get_decision_nodes(tree))),
         max(h2, length(Tree.get_decision_nodes(tree)))}
      end)

    hidden_three_size = length(Mockingjay.DecisionTree.classes(ensemble))

    n_trees = length(ensemble)
    weight_1 = Nx.broadcast(0, {n_trees, hidden_one_size, length(features)})
    bias_1 = Nx.broadcast(0, {n_trees, hidden_one_size})
    weight_2 = Nx.broadcast(0, {n_trees, hidden_two_size, hidden_one_size})
    bias_2 = Nx.broadcast(0, {n_trees, hidden_two_size})
    weight_3 = Nx.broadcast(0, {n_trees, hidden_three_size, hidden_two_size})

    # These correspond to Matrix A in the Hummingbird paper
    w1_indices =
      Nx.flatten(
        Nx.tensor(
          Enum.with_index(
            trees,
            fn tree, index ->
              Enum.with_index(Tree.get_decision_nodes(tree), fn node, node_index ->
                [index, node_index, node.feature]
              end)
            end
          )
        ),
        axes: [0, 1]
      )

    weight_1 =
      Nx.indexed_put(
        weight_1,
        w1_indices,
        Nx.broadcast(1, {w1_indices.shape |> elem(0)})
      )

    # These correspond to Matrix B in the Hummingbird paper
    bias_1 =
      Nx.indexed_put(
        bias_1,
        Nx.flatten(
          Nx.tensor(
            Enum.with_index(
              trees,
              fn tree, index ->
                Enum.with_index(Tree.get_decision_nodes(tree), fn node, node_index ->
                  [index, node_index]
                end)
              end
            )
          ),
          axes: [0, 1]
        ),
        Nx.tensor(
          Enum.flat_map(trees, fn tree ->
            Enum.map(Tree.get_decision_nodes(tree), fn node -> node.threshold end)
          end)
        )
      )

    fn x ->
      forward(
        x,
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
end
