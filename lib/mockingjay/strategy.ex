defmodule Mockingjay.Strategy do
  import Nx.Defn

  @moduledoc """
  Strategy behaviour defines the interface for a strategy module.
  """
  @type t :: Nx.Container.t()

  @doc """
  Compiled the given data into a model with the given options.

  The output model is a struct fields of `:forward`, `:aggregate`, and `:post_transform`.
  These can be set directly or inferred from the data.


  ## Options
  * `forward` - the forward function to use. Can also be a function that takes a Nx.Container.t() and returns a Nx.Container.t().
      If none is specified, the best option will be chosen based on the output type of the model.
  * `:post_transform` - the post transform to use. Builtin options are `:none`, `:softmax`, `:sigmoid`,
      or `:log_softmax`, and `:linear`. Can also be a function that takes a Nx.Container.t() and returns a Nx.Container.t().
      If none is specified, the best option will be chosen based on the output type of the model.
  * `:aggregate` - The aggregation function to use. Builtin options are `:none`, `:mean`, `:sum`, `:max`, and `:min`.
      Can also be a function that takes a Nx.Container.t() and returns a Nx.Container.t(). If none is specified,
      the best option will be chosen based on the output type of the model.
  """
  @callback init(data :: any(), opts :: Keyword.t()) :: {any(), any(), any()}
  @callback forward(x :: Nx.Container.t(), opts :: Keyword.t()) :: Nx.Container.t()
  @callback aggregate(x :: Nx.Container.t(), opts :: Keyword.t()) :: Nx.Container.t()
  @callback post_transform(x :: Nx.Container.t(), opts :: Keyword.t()) :: Nx.Container.t()

  def cond_to_fun(condition) when condition in [:greater, :less, :greater_equal, :less_equal] do
    &apply(Nx, condition, [&1, &2])
  end

  def cond_to_fun(condition) when is_function(condition, 2) do
    condition
  end

  def cond_to_fun(condition),
    do:
      "Invalid condition: #{inspect(condition)} -- must be one of :greater, :less, :greater_equal, :less_equal -- or a custom function of arity 2"

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

  defn linear(x) do
    x
  end

  defn softmax(x) do
    max_val = stop_grad(Nx.reduce_max(x, axes: [1], keep_axes: true))

    stable_exp = Nx.exp(x - max_val)

    stable_exp * Nx.sum(stable_exp, axes: [-1], keep_axes: true)
  end

  defn log_softmax(x) do
    max_val = stop_grad(Nx.reduce_max(x, axes: [1], keep_axes: true))

    stable_exp = Nx.exp(x - max_val)

    stable_exp
    |> Nx.sum(axes: [1], keep_axes: true)
    |> Nx.log()
    |> Nx.negate()
    |> Nx.add(stable_exp)
  end

  defn log_sigmoid(x) do
    x = Nx.negate(x)
    stable = Nx.max(0.0, x)

    x
    |> Nx.abs()
    |> Nx.negate()
    |> Nx.exp()
    |> Nx.log1p()
    |> Nx.add(stable)
    |> Nx.negate()
  end
end
