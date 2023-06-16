defmodule Mockingjay do
  def convert(data, opts \\ []) do
    {strategy, opts} = Keyword.pop(opts, :strategy, :auto)

    strategy =
      case strategy do
        :gemm ->
          Mockingjay.Strategies.GEMM

        :tree_traversal ->
          Mockingjay.Strategies.TreeTraversal

        :ptt ->
          raise NotImplementedError, "PTT strategy not implemented yet -- use :gemm instead"

        :auto ->
          Mockingjay.Strategies.GEMM

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

defmodule NotImplementedError do
  defexception [:message]

  @impl true
  def exception(msg) do
    %NotImplementedError{message: msg}
  end
end
