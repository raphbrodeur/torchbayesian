# torchbayesian — Bayesian Neural Networks made easy

![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue?logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-1.10+-orange?logo=pytorch)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://opensource.org/licenses/Apache-2.0)

**torchbayesian** is a lightweight [PyTorch](https://pytorch.org/) extension that lets you turn any PyTorch model into a **Bayesian Neural Network** (BNN) with just one line of code.
It makes [Bayes-by-Backprop](https://arxiv.org/abs/1505.05424) and **variational inference** effortless and **compatible with any** `nn.Module`, without you having to rewrite your model using custom layers or change your usual PyTorch workflow.
Its goal is to make **uncertainty-aware** and **Bayesian deep learning** as easy as working with any traditional neural network.

### One line to make any torch model into a BNN

Simply wrap any `nn.Module` model `bnn.BayesianModule` to make it a BNN.

```python
from torchbayesian.bnn import BayesianModule

net = BayesianModule(net)  # 'net' is now a BNN
```

The resulting model behaves exactly like any standard `nn.Module`, but instead of learning fixed weight values, your model now learns distributions from which weight values are sampled during training and inference, allowing it to capture uncertainty in its parameters and predictions.

Figure


## Key Features

- **One line** to "BNN-ize" any model — Turn any already existing PyTorch model into a BNN with a single line of code. No need to rewrite your model, redefine layers, or modify your existing architecture.
- **Truly compatible with all layers** — ...
- **PyTorch-native design** — Works entirely within the PyTorch framework; training, inference, evaluation remain unchanged. Fully compatible with other PyTorch-based tools such as [Lightning](https://lightning.ai/docs/pytorch/stable/), [TorchMetrics](https://lightning.ai/docs/torchmetrics/stable/), and [MONAI](https://monai.io/).
- **Custom priors and variational posteriors** — ...
- **KL divergence made simple** — ...
- **Monte-Carlo fallback for KL divergence estimation** — ...

## Requirements

...

## Installation

...

## Getting started

How to use it

How it works

KL divergence

Example .py

Factories

## Motivation

...

## License

...

## Citation

...

## Contact

...
