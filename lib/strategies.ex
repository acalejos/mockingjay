defmodule Mockingjay.Strategies do
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
end
