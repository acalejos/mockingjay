defmodule Mockingjay.Strategies do
  def cond_to_fun(condition) when condition in [:gt, :lt, :ge, :le] do
    case condition do
      :gt -> &Nx.greater/2
      :lt -> &Nx.less/2
      :ge -> &Nx.greater_equal/2
      :le -> &Nx.less_equal/2
    end
  end
end
