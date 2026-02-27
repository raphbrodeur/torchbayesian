<p align="center">
    <img src="https://raw.githubusercontent.com/raphbrodeur/torchbayesian/main/docs/images/logo.png" width="100%" alt="torchbayesian-logo">
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
Its modular design supports configurable priors and variational posteriors, and also includes support for other uncertainty estimation methods such as [Monte Carlo dropout](https://arxiv.org/abs/1506.02142).

### Quickly turn any PyTorch model into a variational Bayesian neural network

Any `nn.Module` can be wrapped with the module container `bnn.BayesianModule` to make it a Bayesian neural network (BNN):

```python
import torchbayesian.bnn as bnn

net = bnn.BayesianModule(net)
```

The resulting module remains a standard `nn.Module`, fully compatible with the PyTorch API.
Internally, its parameters are reparameterized as variational distributions over the weights, enabling approximate Bayesian inference and uncertainty-aware predictions.

<p align="center">
    <img src="https://raw.githubusercontent.com/raphbrodeur/torchbayesian/main/docs/images/bnn_1d_regression.png" width="32%">
    <img src="https://raw.githubusercontent.com/raphbrodeur/torchbayesian/main/docs/images/prostate_segmentation_3d.png" width="32%">
</p>

<p align="center">
    <img src="https://raw.githubusercontent.com/raphbrodeur/torchbayesian/main/docs/images/prostate_segmentation_2d.png" width="50%">
</p>

## Key features

- **Deterministic-to-Bayesian conversion** — Convert any existing model into a Bayesian neural network with a single line of code.
- **Compatible with any `nn.Module`** — Rather than replacing a few specific layers (e.g., `nn.Linear`, `nn.ConvNd`) with Bayesian counterparts, torchbayesian operates at the `nn.Parameter` level.
  Any trainable parameter, including those defined in custom or third-party modules, can be reparameterized as a variational distribution, ensuring the entire model is treated as Bayesian.
- **Modular priors and posteriors** — Configure different priors and variational posteriors directly, or use custom implementations.
- **Direct KL divergence access** — Bayesian models expose their KL divergence via `.kl_divergence()`, using analytic computation when available and Monte Carlo estimation otherwise.
- **PyTorch-native design** — Preserves existing workflows and is fully compatible with other PyTorch-based tools such as MONAI.
- **Monte Carlo dropout support** — Includes dedicated layers and utilities for Monte Carlo dropout uncertainty estimation.

## Installation

torchbayesian works with Python 3.10+ and has a direct dependency on [PyTorch](https://pytorch.org/get-started/locally/).

#### Current release

To install the [current release](https://pypi.org/project/torchbayesian/) with `pip`, run the following:

```bash
pip install torchbayesian
```

## Getting started

A complete [working example](https://github.com/raphbrodeur/torchbayesian/tree/main/examples/bnn_example.py) is available on GitHub.

The model's [Kullback-Leibler divergence](https://en.wikipedia.org/wiki/Evidence_lower_bound) can be retrieved at any point via `.kl_divergence()` on a `bnn.BayesianModule` instance.

Different or custom [priors](https://github.com/raphbrodeur/torchbayesian/tree/main/torchbayesian/bnn/priors) and [variational posteriors](https://github.com/raphbrodeur/torchbayesian/tree/main/torchbayesian/bnn/variational_posteriors) can be used.

For Monte Carlo dropout, dropout layers can be activated during evaluation using the [`enable_mc_dropout()`](https://github.com/raphbrodeur/torchbayesian/tree/main/torchbayesian/bnn/utils/enable_mc_dropout.py) utility.
Dedicated [Bayesian dropout layers](https://github.com/raphbrodeur/torchbayesian/tree/main/torchbayesian/bnn/modules/dropout.py) are also provided.

## Motivation

Modern deep learning models are remarkably powerful but often make predictions with high confidence even when they are wrong.
In safety-critical domains such as health, finance, or autonomous systems, this overconfidence makes it difficult to trust model outputs and impedes automatization.
The torchbayesian package was created to make Bayesian Neural Networks (BNN) and uncertainty quantification in PyTorch as simple as possible.
The goal is to lower the barrier to practical Bayesian deep learning, enabling researchers and practitioners to integrate principled uncertainty estimation directly into their existing framework.

## Contributing

This library is still a work in progress. All contributions via pull requests are welcome.
