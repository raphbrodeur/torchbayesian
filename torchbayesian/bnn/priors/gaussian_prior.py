"""
    @file:              gaussian_prior.py
    @Author:            Raphael Brodeur

    @Creation Date:     08/2025
    @Last modification: 08/2025

    @Description:       This file contains the GaussianPrior class, a parameter gaussian prior distribution for
                        Bayes-by-backprop. It is a common prior distribution for BBB variational inference in practice.
"""

from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal
from torch.nn import Parameter
from torch.types import _size

from torchbayesian.bnn.priors.base import Prior


__all__ = ["GaussianPrior", "NormalPrior"]


class GaussianPrior(Prior):
    """
    This class is a diagonal gaussian prior distribution. It is a standard prior distribution for the parameters in
    practice for BBB rather than a mixture of two gaussians as in the paper "Weight Uncertainty in Neural Networks" by
    Blundell et al..
    """

    def __init__(
            self,
            mu: Optional[float | Tensor] = None,
            sigma: Optional[float | Tensor] = None,
            param: Optional[Parameter | Tensor] = None,
            shape: Optional[_size] = None
    ) -> None:
        """
        Initializes a diagonal gaussian prior distribution. Standard in practice for BBB.

        Parameters
        ----------
        mu : Optional[float | Tensor]
            The mean of the gaussian prior distribution. Either a float that will be assigned to each element of the
            mean matrix or a Tensor whose shape match the mean matrix. Optional. Defaults to 0.
        sigma : Optional[float | Tensor]
            The standard deviation of the gaussian prior distribution. Either a float that will be assigned to each
            element of the std matrix or a Tensor whose shape match the std matrix. Optional. Defaults to 1.
        param : Optional[Parameter | Tensor]
            A parameter or tensor whose shape to use for the mu and sigma tensors. If None, then specified shape is
            used.
        shape : Optional[_size]
            The shape of the mu and sigma tensors. If None, defaults to the shape of the input param.

        Raises
        ------
        ValueError
            If 'mu' and/or 'sigma' are None or float, and both 'param' or 'shape' are provided.
        ValueError
            If either or both 'mu' and 'sigma' are None or float, and both 'param' or 'shape' are not provided.
        ValueError
            If either or both 'mu' and 'sigma' are Tensor, and 'param' and/or 'shape' are specified.
        TypeError
            If either 'mu' or 'sigma' is not type Optional[float | Tensor].
        """
        super().__init__()

        # If needed, get 'shape' argument
        if not isinstance(mu, Tensor) or not isinstance(sigma, Tensor):
            if param is not None:
                if shape is not None:
                    raise ValueError("Provide either 'param' or 'shape' argument, not both.")
                shape = param.size()
            else:
                if shape is None:
                    raise ValueError("Must provide either 'param' or 'shape' argument.")

        # Defaults to N(0, 1) distribution
        if mu is None:
            mu = 0.
        if sigma is None:
            sigma = 1.

        # Assign attribute 'mu'
        if isinstance(mu, float):
            self.mu = torch.full(shape, mu)
        elif isinstance(mu, Tensor):
            if param is not None or shape is not None:
                raise ValueError("If 'mu' is a Tensor, then 'param' and 'shape' arguments must not be provided.")
            self.mu = mu
        else:
            raise TypeError(f"Argument 'mu' must be Optional[float | Tensor], {type(mu)} was provided.")

        # Assign attribute 'sigma'
        if isinstance(sigma, float):
            self.sigma = torch.full(shape, sigma)
        elif isinstance(sigma, Tensor):
            if param is not None or shape is not None:
                raise ValueError("If 'sigma' is a Tensor, then 'param' and 'shape' arguments must not be provided.")
            self.sigma = sigma
        else:
            raise TypeError(f"Argument 'sigma' must be Optional[float | Tensor], {type(sigma)} was provided.")

    @property
    def distribution(self) -> Distribution:
        """
        Returns a torch.distributions.Normal distribution.

        Returns
        -------
        distribution : Distribution
            A torch.Distribution.
        """
        return Normal(self.mu, self.sigma)


NormalPrior = GaussianPrior
