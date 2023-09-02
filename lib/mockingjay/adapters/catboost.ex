defmodule Mockingjay.Adapters.Catboost do
  @enforce_keys [:booster]
  defstruct [:booster]
  # This is a "mocked" function for Catboost while there is no Elixir Catboost library
  # Here we simply make a mock module that can load a catboost json model file and implement the DecisionTree protocol
  def load_model(model_path) do
    unless File.exists?(model_path) and File.regular?(model_path) do
      raise "Could not find model file at #{model_path}"
    end

    json = File.read!(model_path) |> Jason.decode!()
    %__MODULE__{booster: json}
  end

  defp _to_tree(splits, leafs) do
    case splits do
      [] ->
        unless length(leafs) == 1 do
          raise "Bad model: leafs must have length 1"
        end

        %{value: hd(leafs)}

      [split | rest] ->
        # This should always be even since we checked before that
        # its length is a power of 2
        half = (length(leafs) / 2) |> round()
        left_leaves = Enum.take(leafs, half)
        right_leaves = Enum.drop(leafs, half)

        %{
          left: _to_tree(rest, left_leaves),
          right: _to_tree(rest, right_leaves),
          value: %{threshold: split["border"], feature: split["float_feature_index"]}
        }
    end
  end

  def to_tree(%{} = booster_json, n_classes) do
    leaf_values = Map.get(booster_json, "leaf_values")
    splits = Map.get(booster_json, "splits") |> Enum.reverse()

    cond do
      length(leaf_values) == 2 ** length(splits) * n_classes ->
        # Classifier model
        # Will need to argmax to get the class label from the leaf values

        leaf_values = Enum.chunk_every(leaf_values, n_classes)
        _to_tree(splits, leaf_values)

      length(leaf_values) == 2 ** length(splits) ->
        # Regression model
        _to_tree(splits, leaf_values)

      true ->
        raise "Bad model: leaf_values must have length 2 ** length(splits) * n_classes (#{2 ** length(splits) * n_classes}): got #{length(leaf_values)}"
    end
  end

  defimpl Mockingjay.DecisionTree do
    def trees(booster) do
      trees = Map.get(booster.booster, "oblivious_trees")
      n_classes = num_classes(booster)

      if is_nil(trees) do
        raise "Could not find trees in model, found keys #{inspect(Map.keys(booster.booster))}"
      end

      trees
      |> Enum.map(fn tree ->
        Mockingjay.Adapters.Catboost.to_tree(tree, n_classes) |> Mockingjay.Tree.from_map()
      end)
    end

    def num_classes(booster) do
      model_info = Map.get(booster.booster, "model_info")
      keys = Map.keys(model_info)

      cond do
        # Regression models don't have 'class_params' but classifier models still have 'params' key
        "class_params" in keys ->
          class_names = get_in(model_info, ["class_params", "class_names"])

          class_to_label = get_in(model_info, ["class_params", "class_to_label"])

          unless length(class_names) == length(class_to_label) do
            raise "Bad model: class_names and class_to_label must have the same length, got #{length(class_names)} and #{length(class_to_label)}"
          end

          length(class_names)

        "params" in keys ->
          1

        true ->
          raise "Bad model: model must have either 'class_params' or 'params' key -- could not determine number of classes, got #{keys}"
      end
    end

    def num_features(booster) do
      float_features = get_in(booster.booster, ["features_info", "float_features"]) || []

      categorical_features =
        get_in(booster.booster, ["features_info", "categorical_features"]) || []

      length(float_features ++ categorical_features)
    end

    def condition(booster) do
      :greater
    end
  end
end
