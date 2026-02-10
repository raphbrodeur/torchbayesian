"""
    @file:              gaussian_prior.py
    @Author:            Raphael Brodeur

    @Creation Date:     08/2025
    @Last modification: 02/2026

    @Description:       This file contains the 'GaussianPrior' class, a diagonal Gaussian prior distribution used for
                        Bayes by Backprop (BBB) variational inference (VI).
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
    This class is a diagonal Gaussian prior distribution used for Bayes by Backprop (BBB).

    Parameters
    ----------
    shape : _size
        The supposed shape of the parameter for which to initialize a Prior.
    mu : Optional[float | Tensor]
        The mean of the Gaussian prior distribution. Either a float that will be assigned to each element of the mean
        matrix or a Tensor whose shape, dtype and device match the mean matrix. Optional. Defaults to 0.
    sigma : Optional[float | Tensor]
        The standard deviation of the Gaussian prior distribution. Either a float that will be assigned to each element
        of the std matrix or a Tensor whose shape, dtype and device match the std matrix. Optional. Defaults to 1.
    dtype: Optional[_dtype]
        The supposed dtype of the parameter for which to initialize a Prior. Optional. Defaults to torch default dtype.
    device: Device
        The supposed device of the parameter for which to initialize a Prior. Optional. Defaults to torch's default
        device.

    Attributes
    ----------
    mu : Tensor
        The mean of the diagonal Gaussian prior distribution.
    sigma : Tensor
        The standard deviation of the Gaussian prior distribution
    """

    mu: Tensor
    sigma: Tensor

    def __init__(
            self,
            shape: Optional[_size] = None,
            mu: Optional[float | Tensor] = None,
            sigma: Optional[float | Tensor] = None,
            *,
            dtype: Optional[_dtype] = None,
            device: Device = None
    ) -> None:
        """
        Initializes a diagonal Gaussian prior.

        Parameters
        ----------
        shape : Optional[_size]
            The shape of the tensor for which to initialize the prior. Optional. Required if 'mu' and/or 'sigma' are not
            provided as tensors (e.g. if they are floats or None).
        mu : Optional[float | Tensor]
            The mean of the Gaussian prior distribution. If a float is provided, it is assigned to each element of the
            mean matrix. If a Tensor is provided, 'shape' is ignored and its dtype and device are used as-is unless
            'dtype' and/or 'device' are specified, in which case the tensor 'mu' will be moved to 'dtype'/'device'.
            Optional. Defaults to a 0-filled tensor with shape 'shape'.
        sigma : Optional[float | Tensor]
            The standard deviation of the Gaussian prior distribution. If a float is provided, it is assigned to each
            element of the element-wise standard deviation matrix. If a Tensor is provided, 'shape' is ignored and its
            dtype and device are used as-is unless 'dtype' and/or 'device' are specified, in which case the tensor
            'sigma' will be moved to 'dtype'/'device'. Optional. Defaults to a 1-filled tensor with shape 'shape'.
        dtype: Optional[_dtype]
            The dtype of the tensor for which to initialize the prior. Optional. Defaults to torch's default dtype.
        device: Device
            The device of the tensor for which to initialize the prior. Optional. Defaults to torch's default device.

        Raises
        ------
        ValueError
            If 'mu' and/or 'sigma' are None or float, and 'shape' is not provided.
        TypeError
            If either 'mu' or 'sigma' is not type Optional[float | Tensor].
        """
        super().__init__()

        if not isinstance(mu, Tensor) and shape is None:
            raise ValueError(f"Must provide 'shape' argument or tensors 'mu' and 'sigma' with appropriate shape.")
        if not isinstance(sigma, Tensor) and shape is None:
            raise ValueError(f"Must provide 'shape' argument or tensors 'mu' and 'sigma' with appropriate shape.")

        # Defaults to N(0, 1) distribution
        if mu is None:
            mu = 0.
        if sigma is None:
            sigma = 1.

        # Assign attribute 'mu'
        if isinstance(mu, float):
            self.mu = torch.full(shape, mu, dtype=dtype, device=device)
        elif isinstance(mu, Tensor):
            # Align dtype/device if specified; otherwise ('dtype'/'device' is None) keep 'mu' 's dtype/device
            self.mu = mu.to(dtype=dtype, device=device)
        else:
            raise TypeError(f"Argument 'mu' must be Optional[float | Tensor], {type(mu)} was provided.")

        # Assign attribute 'sigma'
        if isinstance(sigma, float):
            self.sigma = torch.full(shape, sigma, dtype=dtype, device=device)
        elif isinstance(sigma, Tensor):
            # Align dtype/device if specified; otherwise ('dtype'/'device' is None) keep 'sigma' 's dtype/device
            self.sigma = sigma.to(dtype=dtype, device=device)
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
