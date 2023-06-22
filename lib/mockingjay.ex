defmodule Mockingjay do
  @moduledoc """
  Mockingjay is a library for compiling trained decision trees to `Nx` `defn` functions.
  It is based on the paper [Taming Model Serving Complexity, Performance and Cost:
  A Compilation to Tensor Computations Approach](https://scnakandala.github.io/papers/TR_2020_Hummingbird.pdf)
  and the accompanying [Hummingbird](https://github.com/microsoft/hummingbird) library from Microsoft.

  ## Protocol

  Mockingjay can be used with any model that implements the `Mockingjay.DecisionTree` protocol. For an example implementation,
  this protocol is implemented by `EXGBoost` in its `EXGBoost.Compile` module. This protocol is used to extract the trees from the model
  and to get the number of classes and features in the model.

  ## Strategies

  Mockingjay supports three strategies for compiling decision trees: `:gemm`, `:tree_traversal`, and `:ptt`, or `:auto` to select
  using heuristics. The `:auto` strategy will select the best strategy based on the depth of the tree according to the
  following rules:
  * GEMM: Shallow Trees (<=3)
  * PerfectTreeTraversal: Tall trees where depth <= 10
  * TreeTraversal: Tall trees unfit for PTT (depth > 10)

  ## Conversion Pipeline

  `Mockingjay` compiles a model using a pipeline composed of three functions, all of which take a Nx.Container.t() and returns a Nx.Container.t().
  These functions will be determined by the strategy chosen, but can also be specified manually. Practically speaking,
  you should not need to specify these manually (especially the forward function). The functions are:
  * `forward` - The forward function -- determined by strategy
  * `aggregate` - Aggregates the output of the forward function -- determined by strategy and output type (ensemble or single tree)
  * `post_transform` - Applies a post transform to the output of the aggregate function -- determined by strategy and output type (classification or regression)

  The `convert` function returns a `defn` function that takes a Nx.Container.t() and returns a Nx.Container.t(), running
  the input through the pipeline (forward -> aggregate -> post_transform).
  """

  @doc """
  Compiles a model that implements the `Mockingjay.DecisionTree` protocol to a `defn` function.

  ## Options

  * `:reorder_trees` - whether to reorder the trees in the model to optimize inference accuracy. Defaults to `true`. This assumes
  that trees are ordere such that they classify classes in order 0..n then repeat (e.g. a cyclic class prediction). If this is not
  the case, set this to `false` and implement custom ordering in the DecisionTree protocol implementation.
  * `:forward` - the forward function to use. A function that takes a Nx.Container.t() and returns a Nx.Container.t().
  If none is specified, the best option will be chosen based on the output type of the model.
  * `:aggregate` - The aggregation function to use. A function that takes a Nx.Container.t() and returns a Nx.Container.t(). If none is specified,
  the best option will be chosen based on the output type of the model.
  * `:post_transform` - the post transform to use. A function that takes a Nx.Container.t() and returns a Nx.Container.t().
  If none is specified, the best option will be chosen based on the output type of the model.
  """
  def convert(data, opts \\ []) do
    {strategy, opts} = Keyword.pop(opts, :strategy, :auto)

    strategy =
      case strategy do
        :gemm ->
          Mockingjay.Strategies.GEMM

        :tree_traversal ->
          Mockingjay.Strategies.TreeTraversal

        :ptt ->
          Mockingjay.Strategies.PerfectTreeTraversal

        :auto ->
          Mockingjay.Strategy.get_strategy(data, opts)

        _ ->
          raise ArgumentError, "strategy must be one of :gemm, :tree_traversal, :ptt, or :auto"
      end

    {forward_opts, aggregate_opts, post_transform_opts} = strategy.init(data, opts)

    &(&1
      |> strategy.forward(forward_opts)
      |> strategy.aggregate(aggregate_opts)
      |> strategy.post_transform(post_transform_opts))
  end
end
