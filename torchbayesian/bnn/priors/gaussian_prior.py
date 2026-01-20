"""
    @file:              gaussian_prior.py
    @Author:            Raphael Brodeur

    @Creation Date:     08/2025
    @Last modification: 08/2025

    @Description:       This file contains the 'GaussianPrior' class, a parameter gaussian prior distribution for
                        Bayes-by-backprop. It is a common prior distribution for BBB variational inference in practice.
"""

from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal
from torch.types import (
    Device,
    _dtype,
    _size
)

from torchbayesian.bnn.priors.base import Prior


__all__ = ["GaussianPrior", "NormalPrior"]


class GaussianPrior(Prior):
    """
    This class is a diagonal gaussian prior distribution. It is a standard prior distribution for the parameters in
    practice for BBB.
    """

    def __init__(
            self,
            shape: Optional[_size] = None,
            mu: Optional[float | Tensor] = None,
            sigma: Optional[float | Tensor] = None,
            *,
            dtype: Optional[_dtype] = None,
            device: Optional[Device] = None
    ) -> None:
        """
        Initializes a diagonal gaussian prior distribution. Common prior in practice for BBB.

        Parameters
        ----------
        shape : _size
            The supposed shape of the parameter for which to initialize a Prior.
        mu : Optional[float | Tensor]
            The mean of the gaussian prior distribution. Either a float that will be assigned to each element of the
            mean matrix or a Tensor whose shape, dtype and device match the mean matrix. Optional. Defaults to 0.
        sigma : Optional[float | Tensor]
            The standard deviation of the gaussian prior distribution. Either a float that will be assigned to each
            element of the std matrix or a Tensor whose shape, dtype and device match the std matrix. Optional. Defaults
            to 1.
        dtype: Optional[_dtype]
            The supposed dtype of the parameter for which to initialize a Prior. Optional. Defaults to torch's default
            dtype.
        device: Optional[Device]
            The supposed device of the parameter for which to initialize a Prior. Optional. Defaults to torch's default
            device.

        Raises
        ------
        ValueError
            If 'mu' and/or 'sigma' are None or float, and 'shape' is not provided.
        TypeError
            If either 'mu' or 'sigma' is not type Optional[float | Tensor].
        """
        super().__init__()

        if not isinstance(mu, Tensor) and shape is None:
            raise ValueError(f"Must provide 'shape' argument or a tensor 'mu' with appropriate shape.")
        if not isinstance(sigma, Tensor) and shape is None:
            raise ValueError(f"Must provide 'shape' argument or a tensor 'sigma' with appropriate shape.")

        # Defaults to N(0, 1) distribution
        if mu is None:
            mu = 0.
        if sigma is None:
            sigma = 1.

        # Assign attribute 'mu'
        if isinstance(mu, float):
            self.mu = torch.full(shape, mu, dtype=dtype, device=device)
        elif isinstance(mu, Tensor):
            self.mu = mu
        else:
            raise TypeError(f"Argument 'mu' must be Optional[float | Tensor], {type(mu)} was provided.")

        # Assign attribute 'sigma'
        if isinstance(sigma, float):
            self.sigma = torch.full(shape, sigma, dtype=dtype, device=device)
        elif isinstance(sigma, Tensor):
            self.sigma = sigma
        else:
            raise TypeError(f"Argument 'sigma' must be Optional[float | Tensor], {type(sigma)} was provided.")

    @property
    def distribution(self) -> Distribution:
        """
        Returns a 'torch.distributions.Normal' distribution.

        Returns
        -------
        distribution : Distribution
            A torch.Distribution.
        """
        return Normal(self.mu, self.sigma)

    def extra_repr(self) -> str:
        """
        Returns the extra representation of the prior.

        Returns
        -------
        extra_repr : str
            The str extra representation of the prior.
        """
        return f"mu: {self.mu.size()}, sigma: {self.sigma.size()}"


NormalPrior = GaussianPrior
