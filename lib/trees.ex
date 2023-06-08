defprotocol DecisionTree do
  @doc """
  Returns a Tree struct representing the decision tree.
  """
  @spec trees(data :: any) :: [Tree.t()]
  def trees(data)

  @spec classes(data :: any) :: list
  def classes(data)

  @spec features(data :: any) :: list
  def features(data)

  @spec output_type(data :: any) :: :classification | :regression
  def output_type(data)

  @spec condition(data :: any) :: :gt | :lt | :ge | :le
  def condition(data)
end

defmodule Mockingjay.Tree do
  @moduledoc """
  A struct containing a convenient in-memory representation of a decision tree. Each "node" in the tree is a `Tree` struct.
  Each child of a node is either a `Tree` struct or `nil` if it is a leaf.

  id: The id of the node. This designates the position in a BFS traversal of the tree. nil for leaf nodes.
  left: The left child of the node.
  right: The right child of the node.
  value: The value of the node:
    - For leaf nodes, this is the value as a number.
    - For non-leaf nodes, this is a map containing the following keys:
      - feature: The feature used to split the data (if it is not a leaf).
      - threshold: The threshold used to split the data (if it is not a leaf).
      - operator: The condition used to split the data (if it is not a leaf) Can be `:le`, `:lt`, `:ge`, `:gt`.
  """
  @enforce_keys [:id, :value]
  defstruct [:id, :left, :right, :value]

  @typedoc "A simple binary tree implementation."
  @type t() :: %__MODULE__{
          id: pos_integer(),
          left: __MODULE__.t() | nil,
          right: __MODULE__.t() | nil,
          value: any()
        }

  # Credit to this SO answer: https://stackoverflow.com/questions/55327307/flatten-a-binary-tree-to-list-ordered

  def bfs(root) do
    root
    |> reduce_tree([], fn val, acc ->
      [val | acc]
    end)
    |> Enum.reverse()
  end

  def reduce_tree(root, acc, reducer) do
    :queue.new()
    |> :queue.snoc(root)
    |> process_queue(acc, reducer)
  end

  def process_queue(queue, acc, reducer) do
    case :queue.out(queue) do
      {{:value, %{value: value, left: nil, right: nil}}, popped} ->
        new_acc = reducer.(value, acc)
        process_queue(popped, new_acc, reducer)

      {{:value, %{left: left, right: right, value: value}}, popped} ->
        new_acc = reducer.(value, acc)

        popped
        |> :queue.snoc(left)
        |> :queue.snoc(right)
        |> process_queue(new_acc, reducer)

      _other ->
        acc
    end
  end

  def depth(tree) do
    depth_from_level(tree, 1)
  end

  def depth_from_level(%{} = tree, current_depth) do
    case tree.value do
      %{} ->
        max(
          depth_from_level(tree.left, current_depth + 1),
          depth_from_level(tree.right, current_depth + 1)
        )

      _ ->
        current_depth
    end
  end

  def get_decision_nodes(tree) do
    bfs(tree)
    |> Enum.filter(fn node ->
      is_map(node)
    end)
  end

  def get_leaf_nodes(tree) do
    bfs(tree)
    |> Enum.filter(fn node ->
      not is_map(node)
    end)
  end
end
