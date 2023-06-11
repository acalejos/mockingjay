defmodule Mockingjay do
  def convert(model, opts \\ []) do
    opts = Keyword.validate!(opts, strategy: :auto)

    if opts[:strategy] not in [:gemm, :tree_traversal, :ptt, :auto] do
      raise ArgumentError, "strategy must be one of :gemm, :tree_traversal, :ptt, or :auto"
    end

    case opts[:strategy] do
      :gemm ->
        Mockingjay.Strategies.GEMM.compile(model)

      :tree_traversal ->
        raise NotImplementedError,
              "TreeTraversal strategy not implemented yet -- use :gemm instead"

      :ptt ->
        raise NotImplementedError, "PTT strategy not implemented yet -- use :gemm instead"

      :auto ->
        Mockingjay.Strategies.GEMM.compile(model)
    end
  end
end

defmodule NotImplementedError do
  defexception [:message]

  @impl true
  def exception(msg) do
    %NotImplementedError{message: msg}
  end
end
