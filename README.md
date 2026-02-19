<p align="center">
    <img src="/docs/images/logo.png" width="100%" alt="torchbayesian-logo">
</p>

# torchbayesian — Bayesian Neural Networks made easy

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange?logo=pytorch)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://static.pepy.tech/badge/torchbayesian)](https://pepy.tech/project/torchbayesian)

**torchbayesian** is a lightweight [PyTorch](https://pytorch.org/) extension that lets you turn any PyTorch model into a **Bayesian Neural Network** (BNN) with just one line of code.
It makes [Bayes-by-Backprop](https://arxiv.org/abs/1505.05424) and **variational inference** effortless and **compatible with any** `nn.Module`, without you having to rewrite your model using custom layers or change your usual PyTorch workflow.
Its goal is to make **uncertainty-aware** and **Bayesian deep learning** as easy as working with any traditional neural network.

### One line to transform any torch model into a BNN

Simply wrap any `nn.Module` model `bnn.BayesianModule` to make it a BNN :

```python
from torchbayesian.bnn import BayesianModule

net = BayesianModule(net)  # 'net' is now a BNN
```

The resulting model behaves exactly like any standard `nn.Module`, but instead of learning fixed weight values, your model now learns distributions from which weight values are sampled during training and inference, allowing it to capture uncertainty in its parameters and predictions.

<p align="center">
    <img src="https://raw.githubusercontent.com/raphbrodeur/torchbayesian/main/docs/images/bnn_1d_regression.png" width="50%">
</p>

## Key Features

- **One line to "BNN-ize" any model** — Turn any already existing PyTorch model into a BNN with a single line of code. No need to rewrite your model, redefine layers, or modify your existing architecture.
- **Truly compatible with all layers** — Unlike other "BNN-izers" that swap specific supported layers for variational versions (most often, only `nn.Linear` and `nn.ConvNd` layers are made variational), torchbayesian actually converts every trainable parameter in your model into a variational posterior `nn.Module`, actually making the entire model Bayesian, not just parts of it.
- **PyTorch-native design** — Works entirely within PyTorch's framework; training, inference, evaluation remain unchanged. Fully compatible with other PyTorch-based tools such as [Lightning](https://lightning.ai/docs/pytorch/stable/), [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/), and [MONAI](https://monai.io/).
- **Custom priors and variational posteriors** — Specify priors and variational posteriors directly as arguments. You can also define your own custom priors and variational posteriors and register them with the API using a simple decorator logic. This allows both plug-and-play use and deep customization without having to touch the core library.
- **KL divergence easily accessible** — Retrieve the model's KL divergence at any point using the `.kl_divergence()` method of `bnn.BayesianModule`.
- **Flexible KL computation** — When analytic computation is not available for some pair of variational posterior and prior, falls back to an estimation using Monte-Carlo sampling. This ensures generality and support for arbitrary user-defined distributions.

## Requirements

torchbayesian works with Python 3.10+ and has a direct dependency on [PyTorch](https://pytorch.org/get-started/locally/).

## Installation

To install the current release, run :

```bash
pip install torchbayesian
```

## Getting started

> This 	`README.md` is still a work in progress. Further details will be added.

A working example is available at [torchbayesian/examples](https://github.com/raphbrodeur/torchbayesian/tree/main/examples).

The [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Evidence_lower_bound) of the model can be retrieved at any point using the `.kl_divergence()` method of `bnn.BayesianModule`.

Different priors and posteriors can be used.

## Motivation

Modern deep learning models are remarkably powerful, but they often make predictions with high confidence even when they’re wrong.
In safety-critical domains such as health, finance, or autonomous systems, this overconfidence makes it difficult to trust model outputs and impedes automatization.
`torchbayesian` was created to make Bayesian Neural Networks (BNNs) and uncertainty quantification in PyTorch as simple as possible.
The goal is to lower the barrier to practical Bayesian deep learning, enabling researchers and practitioners to integrate principled uncertainty estimation directly into their existing framework.

## Citation

...

## Contact

...
