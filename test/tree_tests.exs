defmodule TreeTests do
  alias Mockingjay.Tree
  use ExUnit.Case

  test "bfs" do
    tree = %{
      id: 1,
      value: %{feature: "x3", condition: :gt, threshold: 3},
      left: %{
        id: 2,
        value: %{feature: "x2", condition: :gt, threshold: 5},
        left: %{id: 3, value: 10},
        right: %{
          id: 4,
          value: %{feature: "x4", condition: :gt, threshold: 3},
          left: %{id: 5, value: 40},
          right: %{id: 6, value: 50}
        }
      },
      right: %{
        id: 7,
        value: %{feature: "x1", condition: :lt, threshold: 1.2},
        left: %{id: 8, value: 30},
        right: %{id: 9, value: 20}
      }
    }

    assert Tree.bfs(tree) == [
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

  test "depth" do
    tree = %{
      value: %{feature: "x3", condition: :gt, threshold: 3},
      left: %{
        value: %{feature: "x2", condition: :gt, threshold: 5},
        left: %{value: 10},
        right: %{
          value: %{feature: "x4", condition: :gt, threshold: 3},
          left: %{value: 40},
          right: %{value: 50}
        }
      },
      right: %{
        value: %{feature: "x1", condition: :lt, threshold: 1.2},
        left: %{value: 30},
        right: %{value: 20}
      }
    }

    assert Tree.depth(tree) == 4
  end
end
