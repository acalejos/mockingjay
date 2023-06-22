defmodule Mockingjay.MixProject do
  use Mix.Project

  def project do
    [
      app: :mockingjay,
      version: "0.1.0",
      elixir: "~> 1.14",
      start_permanent: Mix.env() == :prod,
      elixirc_paths: elixirc_paths(Mix.env()),
      deps: deps(),
      package: package(),
      docs: docs(),
      preferred_cli_env: [
        docs: :docs,
        "hex.publish": :docs
      ],
      name: "Mockingjay",
      description:
        "A library to convert trained decision tree models into [Nx](https://hexdocs.pm/nx/Nx.html) tensor operations."
    ]
  end

  defp elixirc_paths(:test), do: ["lib", "test/support"]
  defp elixirc_paths(_), do: ["lib"]

  def application do
    [
      extra_applications: [:logger]
    ]
  end

  defp deps do
    [
      {:nx, "~> 0.5"},
      {:axon, "~> 0.5"},
      {:ex_doc, "~> 0.29.0", only: :docs}
    ]
  end

  defp package do
    [
      maintainers: ["Andres Alejos"],
      licenses: ["Apache-2.0"],
      links: %{"GitHub" => "https://github.com/acalejos/mockingjay"}
    ]
  end

  defp docs do
    [
      main: "Mockingjay"
    ]
  end
end
