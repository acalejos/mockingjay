defmodule CatboostTest do
  use ExUnit.Case, async: true
  alias Mockingjay.DecisionTree
  alias Mockingjay.Adapters.Catboost

  test "load json" do
    clf_booster = Catboost.load_model("test/support/catboost_classifier.json")
    reg_booster = Catboost.load_model("test/support/catboost_regressor.json")
  end

  test "protocol implementation" do
    clf_booster = Catboost.load_model("test/support/catboost_classifier.json")
    reg_booster = Catboost.load_model("test/support/catboost_regressor.json")

    for {booster, expected_num_class} <- [{clf_booster, 5}, {reg_booster, 1}] do
      trees = DecisionTree.trees(booster)

      assert is_list(trees)
      assert is_struct(hd(trees) |> IO.inspect(label: "Tree"), Mockingjay.Tree)
      assert DecisionTree.num_classes(booster) == expected_num_class
      assert DecisionTree.num_features(booster) == 137
    end
  end
end
