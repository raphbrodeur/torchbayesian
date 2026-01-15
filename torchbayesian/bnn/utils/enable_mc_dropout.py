"""
    @file:              enable_mc_dropout.py
    @Author:            Raphael Brodeur

    @Creation Date:     01/2026
    @Last modification: 01/2026

    @Description:       This file contains the 'enable_mc_dropout()' function which is used to enable Monte Carlo
                        Dropout (see paper "Dropout as a Bayesian Approximation: Representing Model Uncertainty in Deep
                        Learning" by Y. Gal and Z. Ghahramani).
"""

from torch.nn import Module
from torch.nn.modules.dropout import _DropoutNd


__all__ = ["enable_mc_dropout"]


def enable_mc_dropout(module: Module) -> None:
    """
    Enables dropout at inference time for Monte Carlo Dropout (see paper "Dropout as a Bayesian Approximation:
    Representing Model Uncertainty in Deep Learning" by Y. Gal and Z. Ghahramani).

    Sets all dropout layers of a torch 'nn.Module' to training mode. Only affects subclasses of 'nn._DropoutNd'; other
    modules remain in their current training state.

    Parameters
    ----------
    module : Module
        The torch 'nn.Module' for which to enable Monte Carlo dropout.

    Examples
    --------
        net.eval()                  # 'net' is a typical 'nn.Module' trained with dropout
        enable_mc_dropout(net)      # MC dropout is now possible
    """
    for m in module.modules():
        if isinstance(m, _DropoutNd):
            m.train()
