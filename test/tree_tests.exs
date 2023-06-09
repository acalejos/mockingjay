defmodule TreeTests do
  alias Mockingjay.Tree
  use ExUnit.Case, async: true

  setup do
    %{
      tree: %Tree{
        id: 1,
        value: %{feature: 2, condition: :gt, threshold: 3},
        left: %Tree{
          id: 2,
          value: %{feature: 1, condition: :gt, threshold: 5},
          left: %Tree{id: 3, value: 10},
          right: %Tree{
            id: 4,
            value: %{feature: 3, condition: :gt, threshold: 3},
            left: %Tree{id: 5, value: 40},
            right: %Tree{id: 6, value: 50}
          }
        },
        right: %Tree{
          id: 7,
          value: %{feature: 0, condition: :lt, threshold: 1.2},
          left: %Tree{id: 8, value: 30},
          right: %Tree{id: 9, value: 20}
        }
      }
    }
  end

  test "bfs", context do
    assert Tree.bfs(context.tree) |> Enum.map(& &1.value) == [
             %{condition: :gt, feature: 2, threshold: 3},
             %{condition: :gt, feature: 2, threshold: 5},
             %{condition: :lt, feature: 2, threshold: 1.2},
             10,
             %{condition: :gt, feature: 4, threshold: 3},
             30,
             20,
             40,
             50
           ]
  end

  test "depth", context do
    assert Tree.depth(context.tree) == 4
  end

  test "get decision values", context do
    assert Tree.get_decision_values(context.tree) == [
             %{condition: :gt, feature: 3, threshold: 3},
             %{condition: :gt, feature: 2, threshold: 5},
             %{condition: :lt, feature: 1, threshold: 1.2},
             %{condition: :gt, feature: 4, threshold: 3}
           ]
  end

  test "get leaf values", context do
    assert Tree.get_leaf_values(context.tree) == [10, 30, 20, 40, 50]
  end
end
