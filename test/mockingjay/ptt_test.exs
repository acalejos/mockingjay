defmodule Mockingjay.GEMMTest do
  alias Mockingjay.Tree
  alias Mockingjay.Strategies.PerfectTreeTraversal
  use ExUnit.Case, async: true
  # Skipping to keep the function it's testing private
  @tag :skip
  test "make tree perfect" do
    bad_tree =
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

    # Ensure tree is not already perfect binary tree
    assert 2 ** Tree.depth(bad_tree) - 1 != length(Tree.bfs(bad_tree))

    # Make tree perfect
    perfect_tree = PerfectTreeTraversal._make_tree_perfect(bad_tree, 0, Tree.depth(bad_tree))
    assert 2 ** Tree.depth(perfect_tree) - 1 == length(Tree.bfs(perfect_tree))

    assert Tree.get_leaf_nodes(perfect_tree) |> Enum.uniq() ==
             Tree.get_leaf_nodes(bad_tree) |> Enum.uniq()
  end
end
