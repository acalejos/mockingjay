defmodule(Mockingjay.TreeTraversal) do
  # def forward(X, ) do

  # end

  def compile(ensemble) do
    # DecisionTree.trees(ensemble)
    trees = ensemble
    # DecisionTree.num_features(ensemble)
    num_features = 5
    # DecisionTree.condition(ensemble)
    condition = :classification

    n_classes = 2
    # if DecisionTree.output_type(ensemble) == :classification do
    #   DecisionTree.num_classes(ensemble)
    # else
    #   1
    # end
    num_trees = length(trees)

    num_nodes =
      Enum.reduce(trees, 0, fn tree, acc ->
        max(acc, length(Tree.bfs(tree)))
      end)

    lefts = Nx.broadcast(0, {num_trees, num_nodes})
    rights = Nx.broadcast(0, {num_trees, num_nodes})
    features = Nx.broadcast(0, {num_trees, num_nodes})
    thresholds = Nx.broadcast(0, {num_trees, num_nodes})
    values = Nx.broadcast(0, {num_trees, num_nodes, n_classes})

    nodes_offset = Enum.with_index(trees, fn _tree, index -> index * num_nodes end) |> Nx.tensor()
  end
end
