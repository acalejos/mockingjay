defmodule Mockingjay do
  def convert(model, opts \\ []) do
    opts = Keyword.validate!(opts, strategy: :auto)

    if opts[:strategy] not in [:gemm, :tree_traversal, :ptt, :auto] do
      raise ArgumentError, "strategy must be one of :gemm, :tree_traversal, :ptt, or :auto"
    end

    case opts[:strategy] do
      :gemm -> Mockingjay.Strategies.Gemm.compile(model)
      :tree_traversal -> raise NotImplementedError
      :ptt -> raise NotImplementedError
      :auto -> Mockingjay.Strategies.Gemm.compile(model)
    end
  end
end
