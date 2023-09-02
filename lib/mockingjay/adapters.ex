defmodule Mockingjay.Adapters do
  @moduledoc """
  The Adapter module provides adapters for `EXGBoost`,`Catboost` and 'LightGBM' models.
  These adapters are used to implement the `Mockingjay.DecisionTree` protocol for these models.

  The 'EXGBoost' adapter works with 'EXGBoost' 'Booster' structs, and thus can be used directly with 'EXGBoost' models.

  The 'Catboost' and 'LightGBM' adapter work by creating mock modules for these libraries that implement the 'Mockingjay.DecisionTree' protocol.
  These adapters can be used with models from these libraries by passing the model to the 'Mockingjay.convert' function.

  Refer to each adapter module for more information on how to load models from each library. Please note that as these are mock modules,
  they only serve to load the JSON model files and implement the 'Mockingjay.DecisionTree' protocol. They do not provide any other functionality
  from the original libraries.
  """
end
