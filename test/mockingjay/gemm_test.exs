defmodule Mockingjay.GEMMTest do
  alias Mockingjay.Tree
  alias Mockingjay.Strategies.GEMM
  use ExUnit.Case, async: true

  # Test tree and matrix outputs from the Hummingbird paper
  setup do
    tree = %Tree{
      id: 1,
      value: %{feature: 2, condition: :lt, threshold: 0.5},
      left: %Tree{
        id: 2,
        value: %{feature: 1, condition: :lt, threshold: 2.0},
        left: %Tree{id: 3, value: 0},
        right: %Tree{id: 4, value: 1}
      },
      right: %Tree{
        id: 5,
        value: %{feature: 4, condition: :lt, threshold: 5.5},
        left: %Tree{
          id: 6,
          value: %{feature: 2, condition: :lt, threshold: 2.4},
          left: %Tree{id: 7, value: 1},
          right: %Tree{id: 8, value: 0}
        },
        right: %Tree{id: 9, value: 0}
      }
    }

    trees = [tree]

    {hidden_one_size, hidden_two_size} =
      Enum.reduce(trees, {0, 0}, fn tree, {h1, h2} ->
        {max(h1, length(Tree.get_decision_nodes(tree))),
         max(h2, length(Tree.get_leaf_nodes(tree)))}
      end)

    %{
      tree: tree,
      trees: trees,
      hidden_one_size: hidden_one_size,
      hidden_two_size: hidden_two_size,
      num_features: 5,
      num_classes: 2
    }
  end

  # test "A matrix", context do
  #   assert GEMM.generate_matrix_A(
  #            context.trees,
  #            context.num_features,
  #            context.hidden_one_size
  #          ) ==
  #            Nx.tensor([
  #              [
  #                [0, 0, 0, 0],
  #                [0, 1, 0, 0],
  #                [1, 0, 0, 1],
  #                [0, 0, 0, 0],
  #                [0, 0, 1, 0]
  #              ]
  #            ])
  # end

  test "B matrix", context do
    assert GEMM.generate_matrix_B(context.trees, context.hidden_one_size) ==
             Nx.tensor([
               [0.5, 2.0, 5.5, 2.4000000953674316]
             ])
  end

  test "C matrix", context do
    assert GEMM.generate_matrix_C(context.trees, context.hidden_one_size, context.hidden_two_size) ==
             Nx.tensor([
               [
                 [1, 1, -1, -1, -1],
                 [1, -1, 0, 0, 0],
                 [0, 0, 1, 1, -1],
                 [0, 0, 1, -1, 0]
               ]
             ])
  end

  test "D matrix", context do
    assert GEMM.generate_matrix_D(context.trees, context.hidden_two_size) ==
             Nx.tensor([
               [2, 1, 2, 1, 0]
             ])
  end

  test "E matrix", context do
    hidden_three_size = context.num_classes

    assert GEMM.generate_matrix_E(
             context.trees,
             context.hidden_two_size,
             hidden_three_size
           ) ==
             Nx.tensor([
               [
                 [1, 0],
                 [0, 1],
                 [0, 1],
                 [1, 0],
                 [1, 0]
               ]
             ])
  end

  test "compile", context do
    assert is_function(GEMM.compile(context.trees), 1)
  end
end
