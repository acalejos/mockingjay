defmodule Mockingjay do
  def convert(data, opts \\ []) do
    opts = Keyword.validate!(opts, strategy: :auto)

    if opts[:strategy] not in [:gemm, :tree_traversal, :ptt, :auto] do
      raise ArgumentError, "strategy must be one of :gemm, :tree_traversal, :ptt, or :auto"
    end

    strategy =
      case opts[:strategy] do
        :gemm ->
          Mockingjay.Strategies.GEMM

        :tree_traversal ->
          raise NotImplementedError,
                "TreeTraversal strategy not implemented yet -- use :gemm instead"

        :ptt ->
          raise NotImplementedError, "PTT strategy not implemented yet -- use :gemm instead"

        :auto ->
          Mockingjay.Strategies.GEMM
      end

    strategy.compile(data)
  end

  def predict(model, x) do
    x
    |> model.forward.()
    |> model.aggregate.()
    |> model.post_transform.()
  end
end

defmodule NotImplementedError do
  defexception [:message]

  @impl true
  def exception(msg) do
    %NotImplementedError{message: msg}
  end
end
