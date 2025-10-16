"""
    @file:              gaussian_posterior.py
    @Author:            Raphael Brodeur

    @Creation Date:     07/2025
    @Last modification: 08/2025

    @Description:       This file contains the 'GaussianPosterior' class, a diagonal gaussian variational posterior for
                        Bayes-by-backprop. It is the standard variational posterior for BBB variational inference.
"""

from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal
from torch.nn import Parameter
import torch.nn.functional as F
from torch.types import (
    Device,
    _dtype,
    _size
)

from torchbayesian.bnn.variational_posteriors.base import VariationalPosterior


__all__ = ["GaussianPosterior", "NormalPosterior"]


class GaussianPosterior(VariationalPosterior):
    """
    This class is a diagonal gaussian variational posterior that produces samples for the parameters via the
    reparametrization trick.

    This is the standard variational posterior with reparametrization trick for Bayes-by-backprop variational inference
    (for more details, see paper "Weight Uncertainty in Neural Networks" by Blundell).
    """

    def __init__(
            self,
            shape: _size,
            *,
            dtype: Optional[_dtype] = None,
            device: Optional[Device] = None
    ) -> None:
        """
        Initializes a gaussian variational posterior distribution.

        Parameters
        ----------
        shape : _size
            The shape of the parameter replaced by the variational posterior.
        dtype : Optional[_dtype]
            The dtype of the parameter replaced by the variational posterior. Optional. Defaults to torch's default
            dtype.
        device : Optional[Device]
            The device of the parameter replaced by the variational posterior. Optional. Defaults to torch's default
            device.
        """
        super().__init__(shape=shape, dtype=dtype, device=device)

        # Create empty variational parameters mu and rho
        self.mu = Parameter(torch.empty(size=shape, dtype=dtype, device=device))
        self.rho = Parameter(torch.empty(size=shape, dtype=dtype, device=device))

        self.reset_parameters()     # Initializes variational parameters

    def reset_parameters(self) -> None:
        """
        Initializes the variational parameters mu and rho of the posterior N(mu, sigma), where sigma = ln(1 + exp(rho)).
        """
        torch.nn.init.normal_(self.mu, mean=0.0, std=0.1)
        torch.nn.init.constant_(self.rho, -3.0)

    @property
    def sigma(self) -> Tensor:
        """
        Returns the sigma parameter of the gaussian distribution (which is reparametrized with rho).

        Returns
        -------
        sigma : Tensor
            The sigma parameter of the gaussian distribution
        """
        return F.softplus(self.rho)

    @property
    def distribution(self) -> Distribution:
        """
        Returns torch.distributions.Normal for KL divergence computation.

        Returns
        -------
        distribution : Distribution
            A torch.distributions.Normal distribution.
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
        eps = torch.randn_like(self.rho)    # eps ~ N(0, 1)
        param = self.mu + self.sigma * eps  # Reparametrization trick

        return param


NormalPosterior = GaussianPosterior
