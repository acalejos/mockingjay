defmodule Mockingjay.GEMMTest do
  use ExUnit.Case, async: true

  alias Mockingjay.Tree
  alias Mockingjay.Strategies.GEMM

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

  test "convert", context do
    model = %Mockingjay.Model{
      trees: context.trees,
      num_classes: 1,
      num_features: 5,
      output_type: :classification,
      condition: :less
    }

    f = Mockingjay.convert(model)
    assert is_function(f, 1)
  end
end
