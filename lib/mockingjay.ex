defmodule Mockingjay do
  @moduledoc """
  Mockingjay is a library for compiling trained decision trees to `Nx` `defn` functions.

  It is based on the paper [Taming Model Serving Complexity, Performance and Cost:
  A Compilation to Tensor Computations Approach](https://scnakandala.github.io/papers/TR_2020_Hummingbird.pdf)
  and the accompanying [Hummingbird library](https://github.com/microsoft/hummingbird) from Microsoft.

  ## Protocol

  Mockingjay can be used with any model that implements the `Mockingjay.DecisionTree` protocol. For an example implementation,
  this protocol is implemented by `EXGBoost` in its `EXGBoost.Compile` module. This protocol is used to extract the trees from the model
  and to get the number of classes and features in the model.

  ## Strategies

  Mockingjay supports three strategies for compiling decision trees: `:gemm`, `:tree_traversal`, and `:perfect_tree_traversal`,
  or `:auto` to select using heuristics. The `:auto` strategy will select the best strategy based on the depth of the tree
  according to the following rules:

    * GEMM: Shallow Trees (<=3)

    * PerfectTreeTraversal: Tall trees where depth <= 10

    * TreeTraversal: Tall trees unfit for PerfectTreeTraversal (depth > 10)

  """

  @doc """
  Compiles a model that implements the `Mockingjay.DecisionTree` protocol to a `defn` function.

  ## Options

    * `:reorder_trees` - whether to reorder the trees in the model to optimize inference accuracy. Defaults to `true`. This assumes
      that trees are ordered such that they classify classes in order 0..n then repeat (e.g. a cyclic class prediction). If this is not
      the case, set this to `false` and implement custom ordering in the DecisionTree protocol implementation.

    * `:post_transform` - the post transform to use. Must be one of :none, :softmax, :sigmoid, :log_softmax, :log_sigmoid or :linear,
      or a custom function that receives the aggregation results. Defaults to sigmoid if n_classes <= 2, otherwise softmax.
  """
  def convert(data, opts \\ []) do
    {strategy, opts} = Keyword.pop(opts, :strategy, :auto)

    strategy =
      case strategy do
        :gemm ->
          Mockingjay.Strategies.GEMM

        :tree_traversal ->
          Mockingjay.Strategies.TreeTraversal

        :perfect_tree_traversal ->
          Mockingjay.Strategies.PerfectTreeTraversal

        :auto ->
          Mockingjay.Strategy.get_strategy(data, opts)

        _ ->
          raise ArgumentError,
                "strategy must be one of :gemm, :tree_traversal, :perfect_tree_traversal, or :auto"
      end

    {post_transform, opts} = Keyword.pop(opts, :post_transform, nil)
    state = strategy.init(data, opts)

    fn data ->
      result = strategy.forward(data, state)
      {_, n_trees, n_classes} = Nx.shape(result)

      result
      |> aggregate(n_trees, n_classes)
      |> post_transform(post_transform, n_classes)
    end
  end

  defp aggregate(x, n_trees, n_classes) do
    cond do
      n_classes > 1 and n_trees > 1 ->
        n_gbdt_classes = if n_classes > 2, do: n_classes, else: 1
        n_trees_per_class = trunc(n_trees / n_gbdt_classes)

        x
        |> Nx.reshape({:auto, n_gbdt_classes, n_trees_per_class})
        |> Nx.sum(axes: [2])

      n_classes > 1 and n_trees == 1 ->
        Nx.squeeze(x, axes: [1])

      true ->
        raise "unknown output type from strategy"
    end
  end

  defp post_transform(x, post_transform, n_classes) do
    fun = post_transform_to_fun(post_transform || infer_post_transform(n_classes))
    fun.(x)
  end

  defp infer_post_transform(n_classes) when n_classes <= 2, do: :sigmoid
  defp infer_post_transform(_), do: :softmax

  defp post_transform_to_fun(:none) do
    &Function.identity/1
  end

  defp post_transform_to_fun(post_transform)
       when post_transform in [:softmax, :linear, :sigmoid, :log_softmax, :log_sigmoid] do
    &apply(Axon.Activations, post_transform, [&1])
  end

  defp post_transform_to_fun(post_transform) when is_function(post_transform, 1) do
    post_transform
  end

  defp post_transform_to_fun(post_transform) do
    raise ArgumentError,
          "invalid post_transform: #{inspect(post_transform)} -- must be one of :none, :softmax, :sigmoid, :log_softmax, :log_sigmoid or :linear -- or a custom function of arity 1"
  end
end
