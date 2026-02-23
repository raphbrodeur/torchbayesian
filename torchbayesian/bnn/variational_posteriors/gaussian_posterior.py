"""
    @file:              gaussian_posterior.py
    @Author:            Raphael Brodeur

    @Creation Date:     07/2025
    @Last modification: 02/2026

    @Description:       This file contains the 'GaussianPosterior' class, a diagonal Gaussian variational posterior
                        using the reparametrization trick for Bayes by Backprop (BBB) variational inference (VI). It is
                        a commonly used variational posterior.
"""

from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import Parameter
import torch.nn.functional as F
from torch.types import Device

from torchbayesian.bnn.variational_posteriors.base import VariationalPosterior
from torchbayesian.types import _dtype, _size


__all__ = ["GaussianPosterior", "NormalPosterior"]


class GaussianPosterior(VariationalPosterior):
    """
    This class is a diagonal Gaussian variational posterior.

    Samples tensors via the reparametrization trick.

    This is a commonly used variational posterior for Bayes by Backprop (BBB) variational inference (VI), as described
    in "Weight Uncertainty in Neural Networks" by Blundell et al.

    Parameters
    ----------
    shape : _size
        The shape of the parameter being replaced by the variational posterior.
    dtype : Optional[_dtype]
        The dtype of the parameter being replaced by the variational posterior. Optional. Defaults to torch default
        dtype.
    device : Device
        The device of the parameter being replaced by the variational posterior. Optional. Defaults to torch default
        device.

    Attributes
    ----------
    mu : Parameter
        The variational parameter of the mean of the distribution.
    rho : Parameter
        The variational parameter that parametrizes the standard deviation of the distribution via softplus.
    """

    mu: Parameter
    rho: Parameter

    def __init__(
            self,
            shape: _size,
            *,
            dtype: Optional[_dtype] = None,
            device: Device = None
    ) -> None:
        """
        Initializes a diagonal Gaussian variational posterior.

        Parameters
        ----------
        shape : _size
            The shape of the parameter being replaced by the variational posterior.
        dtype : Optional[_dtype]
            The dtype of the parameter being replaced by the variational posterior. Optional. Defaults to torch default
            dtype.
        device : Device
            The device of the parameter being replaced by the variational posterior. Optional. Defaults to torch default
            device.
        """
        super().__init__(shape=shape, dtype=dtype, device=device)

        # Create empty variational parameters 'mu' and 'rho'
        self.mu = Parameter(torch.empty(size=shape, dtype=dtype, device=device))
        self.rho = Parameter(torch.empty(size=shape, dtype=dtype, device=device))

        self.reset_parameters()     # Initializes variational parameters

    def reset_parameters(self) -> None:
        """
        Initializes the variational parameters 'mu' and 'rho' of the Gaussian posterior N(mu, sigma), where
        sigma = softplus(rho).
        """
        torch.nn.init.normal_(self.mu, mean=0.0, std=0.1)
        torch.nn.init.constant_(self.rho, -3.0)

    @property
    def sigma(self) -> Tensor:
        """
        Returns the standard deviation parameter of the Gaussian distribution.

        Returns
        -------
        sigma : Tensor
            The standard deviation parameter of the Gaussian distribution
        """
        return F.softplus(self.rho)

    @property
    def distribution(self) -> Normal:
        """
        Returns a 'torch.distributions.Normal' for KL divergence computation.

        Returns
        -------
        distribution : Normal
            A diagonal Gaussian distribution.
        """
        return Normal(self.mu, self.sigma)

    def sample_parameters(self) -> Tensor:
        """
        Samples parameter values from the variational posterior distribution.

        Returns
        -------
        param : Tensor
            A value of the parameter, sampled from the variational posterior.
        """
        eps = torch.randn_like(self.mu)    # eps ~ N(0, 1)
        param = self.mu + self.sigma * eps  # Reparametrization trick

        return param


NormalPosterior = GaussianPosterior
