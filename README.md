<p align="center">
    <img src="/docs/images/logo.png" width="100%" alt="torchbayesian-logo">
</p>

<p align="center">
    <i>
        The simplest way to build Bayesian Neural Networks in PyTorch.
    </i>
</p>

# torchbayesian

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange?logo=pytorch)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)
[![Downloads](https://static.pepy.tech/badge/torchbayesian)](https://pepy.tech/project/torchbayesian)

torchbayesian is a [PyTorch](https://pytorch.org/)-based, [open-source](https://github.com/raphbrodeur/torchbayesian/blob/main/LICENSE) library for building uncertainty-aware neural networks.
It serves as a lightweight extension to PyTorch, providing support for Bayesian deep learning and principled uncertainty quantification of model predictions while preserving the standard PyTorch workflow.

The package integrates seamlessly into the [PyTorch Ecosystem](https://pytorch.org/ecosystem/), enabling the conversion of existing models into Bayesian neural networks via [Bayes-by-Backprop](https://arxiv.org/abs/1505.05424) variational inference.
Its modular design supports configurable priors and variational posteriors, and also includes support for other uncertainty estimation methods such as Monte Carlo dropout.

### Quickly turn any PyTorch model into a Bayesian neural network

Any `nn.Module` can be wrapped with the module container `bnn.BayesianModule` to make it a Bayesian neural network (BNN):

```python
import torchbayesian.bnn as bnn

net = bnn.BayesianModule(net)
```

The resulting module remains a standard `nn.Module`, fully compatible with the PyTorch API.
Internally, its parameters are reparameterized as variational distributions over the weights, enabling approximate Bayesian inference and uncertainty-aware predictions.

<p align="center">
    <img src="https://raw.githubusercontent.com/raphbrodeur/torchbayesian/main/docs/images/bnn_1d_regression.png" width="50%">
</p>

## Key Features

- **Deterministic-to-Bayesian conversion** — Convert any existing model into a Bayesian neural network with a single line of code.
- **Compatible with any `nn.Module`** — Rather than replacing a few specific layers (e.g., `nn.Linear`, `nn.ConvNd`) with Bayesian counterparts, torchbayesian operates at the `nn.Parameter` level.
  Any trainable parameter, including those defined in custom or third-party modules, can be reparameterized as a variational distribution, ensuring the entire model is treated as Bayesian.
- **Modular priors and posteriors** — Configure different priors and variational posteriors directly, or use custom implementations.
- **Direct KL divergence access** — Bayesian models expose their KL divergence via `.kl_divergence()`, using analytic computation when available and Monte Carlo estimation otherwise.
- **PyTorch-native design** — Preserves existing workflows and is fully compatible with other PyTorch-based tools such as MONAI.
- **Monte Carlo dropout support** — Includes dedicated layers and utilities for Monte Carlo dropout uncertainty estimation.


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
