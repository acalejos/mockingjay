defmodule CatboostTest do
  use ExUnit.Case, async: true
  alias Mockingjay.DecisionTree
  alias Mockingjay.Adapters.Catboost

  test "load json" do
    clf_booster = Catboost.load_model("test/support/catboost_iris.json")
    reg_booster = Catboost.load_model("test/support/catboost_regressor.json")
  end

  test "protocol implementation" do
    clf_booster = Catboost.load_model("test/support/catboost_iris.json")
    reg_booster = Catboost.load_model("test/support/catboost_regressor.json")

    for {booster, expected_num_class, expected_num_feature} <- [
          {clf_booster, 3, 4},
          {reg_booster, 1, 137}
        ] do
      trees = DecisionTree.trees(booster)

      assert is_list(trees)
      assert is_struct(hd(trees) |> IO.inspect(label: "Tree"), Mockingjay.Tree)
      assert DecisionTree.num_classes(booster) == expected_num_class
      assert DecisionTree.num_features(booster) == expected_num_feature
    end
  end

  test "iris performance" do
    {x, y} = Scidata.Iris.download()
    # data = Enum.zip(x, y) |> Enum.shuffle()
    # {train, test} = Enum.split(data, ceil(length(data) * 0.8))
    # {x_train, y_train} = Enum.unzip(train)
    # {x_test, y_test} = Enum.unzip(test)
    x = Nx.tensor(x)
    y = Nx.tensor(y)

    # x_train = Nx.tensor(x_train)
    # y_train = Nx.tensor(y_train)

    # x_test = Nx.tensor(x_test)
    # y_test = Nx.tensor(y_test)

    booster = Catboost.load_model("test/support/catboost_iris.json")

    gemm_predict = Mockingjay.convert(booster, strategy: :gemm, post_transform: :linear)
    # tt_predict = Mockingjay.convert(booster, strategy: :tree_traversal)
    # ptt_predict = Mockingjay.convert(booster, strategy: :perfect_tree_traversal)
    # auto_predict = Mockingjay.convert(booster, strategy: :auto)

    gemm_preds = gemm_predict.(x) |> IO.inspect() |> Nx.argmax(axis: -1)

    gemm_accuracy =
      Scholar.Metrics.accuracy(y, gemm_preds)
      |> Nx.to_number()
      |> IO.inspect(label: "gemm_accuracy")

    assert false
  end
end
