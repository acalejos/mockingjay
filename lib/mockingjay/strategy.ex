defmodule Mockingjay.Strategy do
  @moduledoc """
  Strategy behaviour defines the interface for a strategy module.
  """
  @type t :: Nx.Container.t()

  @doc """
  Compiled the given data into a model with the given options.

  The output model is a struct fields of `:forward`, `:aggregate`, and `:post_transform`.
  These can be set directly or inferred from the data.


  ## Options
  * `:post_transform` - the post transform to use. Builtin options are `:none`, `:softmax`, `:sigmoid`,
      or `:log_softmax`, and `:linear`. Can also be a function that takes a Nx.Container.t() and returns a Nx.Container.t().
      If none is specified, the best option will be chosen based on the output type of the model.
  * `:aggregate` - The aggregation function to use. Builtin options are `:none`, `:mean`, `:sum`, `:max`, and `:min`.
      Can also be a function that takes a Nx.Container.t() and returns a Nx.Container.t(). If none is specified,
      the best option will be chosen based on the output type of the model.
  """
  @callback compile(data :: any(), opts :: Keyword.t()) :: t()

  def cond_to_fun(condition) do
    case condition do
      :gt ->
        &Nx.greater/2

      :lt ->
        &Nx.less/2

      :ge ->
        &Nx.greater_equal/2

      :le ->
        &Nx.less_equal/2

      _ ->
        raise ArgumentError,
              "Invalid condition: #{inspect(condition)} -- must be one of :gt, :lt, :ge, :le"
    end
  end

  def infer_post_transform(n_classes) when is_integer(n_classes) do
    cond do
      n_classes <= 2 ->
        :sigmoid

      true ->
        :softmax
    end
  end

  def post_transform_to_func(post_transform) do
    case post_transform do
      :none ->
        & &1

      :softmax ->
        fn x -> Axon.Activations.softmax(x, axis: 1) end

      :sigmoid ->
        &Axon.Activations.sigmoid/1

      :log_softmax ->
        fn x -> Axon.Activations.log_softmax(x, axis: 1) end

      :log_sigmoid ->
        &Axon.Activations.log_sigmoid/1

      :linear ->
        &Axon.Activations.linear/1

      custom when is_function(custom) ->
        custom

      _ ->
        raise ArgumentError,
              "Invalid post_transform: #{inspect(post_transform)} -- must be one of :none, :softmax, :sigmoid, :log_softmax, :log_sigmoig or :linear -- or a custom function"
    end
  end
end
