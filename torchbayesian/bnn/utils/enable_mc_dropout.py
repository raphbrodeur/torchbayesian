"""
    @file:              enable_mc_dropout.py
    @Author:            Raphael Brodeur

    @Creation Date:     01/2026
    @Last modification: 01/2026

    @Description:       This file contains the 'enable_mc_dropout()' function which is used to enable the dropout layers
                        in a model in eval mode. This is used for Monte Carlo dropout (MCD), as described in "Dropout as
                        a Bayesian Approximation: Representing Model Uncertainty in Deep Learning" by Y. Gal and Z.
                        Ghahramani.
"""

from torch.nn import Module
from torch.nn.modules.dropout import _DropoutNd


__all__ = ["enable_mc_dropout"]


def enable_mc_dropout(module: Module) -> None:
    """
    Enables only the dropout layers of a module, even in eval mode.

    Puts the dropout layers of a module in train mode and leaves other layers in their current train/eval modes.

    This is used for Monte Carlo dropout (MCD), as described in "Dropout as a Bayesian Approximation: Representing Model
    Uncertainty in Deep Learning" by Y. Gal and Z. Ghahramani.

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
