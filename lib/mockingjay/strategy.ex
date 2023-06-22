defmodule Mockingjay.Strategy do
  @moduledoc false
  @type t :: Nx.Container.t()

  @callback init(data :: any(), opts :: Keyword.t()) :: {any(), any(), any()}
  @callback forward(x :: Nx.Container.t(), opts :: Keyword.t()) :: Nx.Container.t()
  @callback aggregate(x :: Nx.Container.t(), opts :: Keyword.t()) :: Nx.Container.t()
  @callback post_transform(x :: Nx.Container.t(), opts :: Keyword.t()) :: Nx.Container.t()

  def cond_to_fun(condition)
      when condition in [:greater, :less, :greater_equal, :less_equal, :equal, :not_equal] do
    &apply(Nx, condition, [&1, &2])
  end

  def cond_to_fun(condition) when is_function(condition, 2) do
    condition
  end

  def cond_to_fun(condition),
    do:
      raise(
        ArgumentError,
        "Invalid condition: #{inspect(condition)} -- must be one of :greater, :less, :greater_equal, :less_equal, :equal, :not_equal -- or a custom function of arity 2"
      )

  def infer_post_transform(n_classes) when is_integer(n_classes) do
    cond do
      n_classes <= 2 ->
        :sigmoid

      true ->
        :softmax
    end
  end

  def post_transform_to_func(post_transform)
      when post_transform in [:softmax, :linear, :sigmoid, :log_softmax, :log_sigmoid] do
    &apply(Axon.Activations, post_transform, [&1])
  end

  def post_transform_to_func(post_transform) when is_function(post_transform, 1) do
    post_transform
  end

  def post_transform_to_func(post_transform),
    do:
      "Invalid post_transform: #{inspect(post_transform)} -- must be one of :none, :softmax, :sigmoid, :log_softmax, :log_sigmoig or :linear -- or a custom function of arity 1"

  def get_strategy(ensemble, opts \\ []) do
    opts = Keyword.validate!(opts, high: 10, low: 3)
    # The current heuristic is such that GEMM <= low < PerfTreeTrav <= high < TreeTrav
    max_tree_depth =
      Enum.reduce(Mockingjay.DecisionTree.trees(ensemble), 0, fn tree, acc ->
        max(acc, Mockingjay.Tree.depth(tree))
      end)

    cond do
      max_tree_depth <= opts[:low] ->
        Mockingjay.Strategies.GEMM

      max_tree_depth <= opts[:high] ->
        Mockingjay.Strategies.PerfectTreeTraversal

      true ->
        Mockingjay.Strategies.TreeTraversal
    end
  end
end
