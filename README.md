# Mockingjay

Implementation of Microsoft's [Hummingbird](https://github.com/microsoft/hummingbird) library for converting trained Decision Tree
models into tensor computations. 

## How to Use

Implement the `DecisionTree` protocol for any data source you would like to compile. Then you can use `Mockingjay.convert/1`
to generate an `Nx.Defn` prediction function that makes inferences. The output of `convert` will be a function with the signature
`fn x -> predict(x)`. The three strategies are GEMM, TreeTraversal, and PerfectTree traversal. You can specify the strategy using the
`:strategy` option in `convert` or use a heuristic strategy by default. The heuristic used is generally:
* GEMM: Shallow Trees (<= 3 on CPU, <= 10 on GPU)
* PerfectTreeTraversal: Tall trees where depth <= 10 (TODO)
* TreeTraversal: Tall trees unfit for PTT (depth > 10) (TODO)

## Installation

```elixir
def deps do
  [
    {:mockingjay, github: "acalejos/mockingjay", branch: "main"}
  ]
end
```
