defmodule Mockingjay.Adapters.Lightgbm do
  @enforce_keys [:model]
  defstruct [:model]
  # This is a "mocked" function for Catboost while there is no Elixir Catboost library
  # Here we simply make a mock module that can load a catboost json model file and implement the DecisionTree protocol
  def load_model(model_path) do
    unless File.exists?(model_path) and File.regular?(model_path) do
      raise "Could not find model file at #{model_path}"
    end

    json = File.read!(model_path) |> Jason.decode!()
    %__MODULE__{model: json}
  end

  defimpl Mockingjay.DecisionTree do
    def trees(booster) do
    end

    def n_classes(booster) do
    end

    def num_features(booster) do
    end

    def condition(booster) do
    end
  end
end
