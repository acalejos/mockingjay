defmodule GEMMTests do
  alias Mockingjay.Tree
  alias Mockingjay.Strategies.GEMM
  use ExUnit.Case, async: true

  setup do
    %{
      tree: %Tree{
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
    }
  end

  test "get tree left depth", context do
    assert GEMM.get_leaf_left_depths(context.tree) == [2, 1, 2, 1, 0]
  end

  # test "get child matrix", context do
  #   assert GEMM.get_child_matrices([context.tree]) ==
  #            Nx.tensor([
  #              [0, 1, 1, -1, -1, -1],
  #              [0, 1, -1, 0, 0, 0],
  #              [0, 0, 0, 1, 1, -1],
  #              [0, 0, 0, 1, -1, 0]
  #            ])
  # end

  test "compile", context do
    assert GEMM.compile([context.tree]) == :ok
  end
end
