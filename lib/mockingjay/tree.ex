defprotocol Mockingjay.DecisionTree do
  @moduledoc """
  A protocol for extracting decision trees from a model and getting information about the model.

  This protocol can be used for any model that uses decision trees as its base model.
  As such, this can be used for both ensemble and single decision tree models.  This protocol
  requires that the model implement the `Mockingjay.Tree` struct for representing decision trees.
  This protocol also requires that all decsision split conditions in the model are the same condition.
  The model does not need to be a perfect binary tree, but it must be a binary tree.
  """

  @doc """
  Returns a list of `Mockingjay.Tree` struct representing the decision tree.
  """
  @spec trees(data :: any) :: [Tree.t()]
  def trees(data)

  @doc """
  Returns the number of classes in the model.
  """
  @spec num_classes(data :: any) :: pos_integer()
  def num_classes(data)

  @doc """
  Returns the number of features in the model.
  """
  @spec num_features(data :: any) :: pos_integer()
  def num_features(data)

  @doc """
  Returns the condition used to split the data.
  """
  @spec condition(data :: any) :: :greater | :less | :greater_equal | :less_equal
  def condition(data)
end

defmodule Mockingjay.Tree do
  @moduledoc """
  A struct containing a convenient in-memory representation of a decision tree.

  Each "node" in the tree is a `Tree` struct. A "decision" or "inner" node is any node whose
  `:left` and `:right` values are a valid `Tree`. A "leaf" or "outer" node is any node whose `:left`
  and `:right` values are `nil`. `:left` and `:right` must either both be `nil` or both be a `Tree`.

  * `:id` - The id of the node. This is a unique reference. This is generated automatically when using the `from_map` function.
  * `:left` - The left child of the node. This is `nil` for leaf nodes. This is a `Tree` for decision nodes.
  * `:right` - The right child of the node. This is `nil` for leaf nodes. This is a `Tree` for decision nodes.
  * `:value` - The value of the node:
    * For leaf nodes, this is the value as a number.
    * For non-leaf nodes, this is a map containing the following keys:
      * `:feature` - The feature used to split the data (if it is not a leaf).
      * `:threshold` - The threshold used to split the data (if it is not a leaf).
  """

  @enforce_keys [:id, :value]
  defstruct [:id, :left, :right, :value]

  @doc """
  Returns a `Tree` struct from a map.
  The map must have the appropriate required keys for the `Tree` struct. Any extra keys are ignored.
  """
  def from_map(%__MODULE__{} = t), do: t

  def from_map(%{} = map) do
    case map do
      %{left: nil, right: nil, value: value} when is_number(value) ->
        %__MODULE__{
          id: make_ref(),
          left: nil,
          right: nil,
          value: value
        }

      %{value: value} when is_number(value) ->
        %__MODULE__{
          id: make_ref(),
          left: nil,
          right: nil,
          value: value
        }

      %{left: nil, right: nil, value: value} ->
        raise ArgumentError, "Leaf nodes must have a numeric value. Got: #{inspect(value)}"

      %{left: left, right: right, value: %{threshold: threshold, feature: feature}}
      when is_number(threshold) and is_number(feature) ->
        %__MODULE__{
          id: make_ref(),
          left: from_map(left),
          right: from_map(right),
          value: %{threshold: threshold, feature: feature}
        }

      %{left: _left, right: _right, value: %{threshold: _threshold, feature: _feature}} ->
        raise ArgumentError,
              "Non-leaf nodes must have a numeric threshold and feature. Got: #{inspect(map)}"

      %{value: value} ->
        raise ArgumentError, "Leaf nodes must have a numeric value. Got: #{inspect(value)}"

      _ ->
        raise ArgumentError, "Invalid tree map: #{inspect(map)}"
    end
  end

  @typedoc "A simple binary tree implementation."
  @type t() :: %__MODULE__{
          id: reference(),
          left: t() | nil,
          right: t() | nil,
          value: number() | %{feature: pos_integer(), threshold: number()}
        }

  # TO-DO: make TCOptimizable
  @doc """
  Returns tree nodes as a list in DFS order.
  """
  def dfs(root) do
    _dfs(root, [])
  end

  defp _dfs(root, acc) do
    case root do
      %{left: nil, right: nil} ->
        acc ++ [root]

      %{left: left, right: right} ->
        [root] ++ _dfs(left, acc) ++ _dfs(right, acc)
    end
  end

  # Credit to this SO answer: https://stackoverflow.com/questions/55327307/flatten-a-binary-tree-to-list-ordered
  @doc """
  Returns a list of nodes in BFS order.
  For the uses in Mockingjay, BFS is tree-level order from right to left on each level.
  The nodes include their children nodes.
  """
  def bfs(root) do
    root
    |> reduce_tree([], fn val, acc ->
      [val | acc]
    end)
    |> Enum.reverse()
  end

  @doc """
  Traverse the tree in BFS order, applying a reducer function to each node.
  """
  def reduce_tree(root, acc, reducer) do
    :queue.new()
    |> :queue.snoc(root)
    |> process_queue(acc, reducer)
  end

  defp process_queue(queue, acc, reducer) do
    case :queue.out(queue) do
      {{:value, %{left: nil, right: nil} = node}, popped} ->
        new_acc = reducer.(node, acc)
        process_queue(popped, new_acc, reducer)

      {{:value, %{left: left, right: right} = node}, popped} ->
        new_acc = reducer.(node, acc)

        popped
        |> :queue.snoc(right)
        |> :queue.snoc(left)
        |> process_queue(new_acc, reducer)

      _other ->
        acc
    end
  end

  @doc """
  Returns the depth of the tree. The root node is at depth 0.
  """
  def depth(tree) do
    depth_from_level(tree, 0)
  end

  @doc """
  Returns the depth of the tree starting from the given level.
  """
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

  @doc """
  Returns a list of the decision nodes in the tree in BFS order.
  """
  def get_decision_nodes(tree) do
    tree
    |> bfs()
    |> Enum.filter(fn node ->
      is_map(node.value)
    end)
  end

  @doc """
  Returns a list of the leaf nodes in the tree in BFS order.
  """
  def get_leaf_nodes(tree) do
    tree
    |> do_get_leaf_nodes([])
    |> Enum.reverse()
  end

  defp do_get_leaf_nodes(tree, nodes) do
    # Unlilke get_decision_nodes, this function returns the leaf nodes in DFS order.
    case tree do
      %__MODULE__{left: nil, right: nil} ->
        [tree | nodes]

      %__MODULE__{left: left, right: right} ->
        left
        |> do_get_leaf_nodes(nodes)
        |> then(&do_get_leaf_nodes(right, &1))
    end
  end

  @doc """
  Returns a list of the values of the leaf nodes in the tree in BFS order.
  """
  def get_leaf_values(tree) do
    tree
    |> get_leaf_nodes()
    |> Enum.map(& &1.value)
  end

  @doc """
  Returns a list of the values of the decision nodes in the tree in BFS order.
  """
  def get_decision_values(tree) do
    tree
    |> bfs()
    |> Enum.filter(fn node ->
      is_map(node.value)
    end)
    |> Enum.map(& &1.value)
  end

  @doc """
  Checks is the given child_id exists in the tree.
  """
  def child?(tree, child_id) do
    case tree do
      nil ->
        false

      %__MODULE__{id: id, left: left, right: right} ->
        id == child_id or child?(left, child_id) or child?(right, child_id)
    end
  end
end
