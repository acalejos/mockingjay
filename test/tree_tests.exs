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
    assert Tree.bfs(context.tree) == [
             %{condition: :gt, feature: "x3", threshold: 3},
             %{condition: :gt, feature: "x2", threshold: 5},
             %{condition: :lt, feature: "x1", threshold: 1.2},
             10,
             %{condition: :gt, feature: "x4", threshold: 3},
             30,
             20,
             40,
             50
           ]
  end

  test "depth", context do
    assert Tree.depth(context.tree) == 4
  end

  test "get decision nodes", context do
    assert Tree.get_decision_nodes(context.tree) == [
             %{condition: :gt, feature: "x3", threshold: 3},
             %{condition: :gt, feature: "x2", threshold: 5},
             %{condition: :lt, feature: "x1", threshold: 1.2},
             %{condition: :gt, feature: "x4", threshold: 3}
           ]
  end

  test "get leaf nodes", context do
    assert Tree.get_leaf_nodes(context.tree) == [10, 30, 20, 40, 50]
  end
end
