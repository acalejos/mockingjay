defmodule TreeTests do
  alias Mockingjay.Tree
  use ExUnit.Case, async: true

  setup do
    %{
      tree: %Tree{
        id: 1,
        value: %{feature: 2, threshold: 3},
        left: %Tree{
          id: 2,
          value: %{feature: 1, threshold: 5},
          left: %Tree{id: 3, value: 10},
          right: %Tree{
            id: 4,
            value: %{feature: 3, threshold: 3},
            left: %Tree{id: 5, value: 40},
            right: %Tree{id: 6, value: 50}
          }
        },
        right: %Tree{
          id: 7,
          value: %{feature: 0, threshold: 1.2},
          left: %Tree{id: 8, value: 30},
          right: %Tree{id: 9, value: 20}
        }
      }
    }
  end

  test "from_map", context do
    new_tree =
      Tree.from_map(%{
        value: %{feature: 2, threshold: 3},
        left: %{
          id: 2,
          value: %{feature: 1, threshold: 5},
          left: %{id: 3, value: 10},
          right: %{
            id: 4,
            value: %{feature: 3, threshold: 3},
            left: %{id: 5, value: 40},
            right: %{id: 6, value: 50}
          }
        },
        right: %{
          id: 7,
          value: %{feature: 0, threshold: 1.2},
          left: %{id: 8, value: 30},
          right: %{id: 9, value: 20}
        }
      })

    Enum.zip(Tree.bfs(context.tree), Tree.bfs(new_tree))
    |> Enum.each(fn {a, b} ->
      assert a.value == b.value
    end)
  end

  test "bfs", context do
    assert Tree.bfs(context.tree) |> Enum.map(& &1.value) == [
             %{feature: 2, threshold: 3},
             %{feature: 1, threshold: 5},
             %{feature: 0, threshold: 1.2},
             10,
             %{feature: 3, threshold: 3},
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
             %{feature: 2, threshold: 3},
             %{feature: 1, threshold: 5},
             %{feature: 0, threshold: 1.2},
             %{feature: 3, threshold: 3}
           ]
  end

  test "get leaf values", context do
    assert Tree.get_leaf_values(context.tree) == [10, 40, 50, 30, 20]
  end
end
