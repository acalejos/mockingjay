defmodule Mockingjay.Model do
  defstruct [:trees, :num_classes, :num_features, :output_type, :condition]

  defimpl Mockingjay.DecisionTree do
    def trees(data) do
      Enum.map(data.trees, &Mockingjay.Tree.from_map/1)
    end

    def num_classes(data), do: data.num_classes
    def num_features(data), do: data.num_features
    def output_type(data), do: data.output_type
    def condition(data), do: data.condition
  end
end
