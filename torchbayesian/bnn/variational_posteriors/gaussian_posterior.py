"""
    @file:              gaussian_posterior.py
    @Author:            Raphael Brodeur

    @Creation Date:     07/2025
    @Last modification: 08/2025

    @Description:       This file contains the GaussianPosterior class, a diagonal gaussian variational posterior for
                        Bayes-by-backprop. It is the standard variational posterior for BBB variational inference.
"""

from typing import Optional

import torch
from torch import Tensor
from torch.distributions import Distribution, Normal
from torch.nn import Parameter
import torch.nn.functional as F
from torch.types import _dtype, _size

from torchbayesian.bnn.variational_posteriors.base import VariationalPosterior


class GaussianPosterior(VariationalPosterior):
    """
    This class is a diagonal gaussian variational posterior that produces samples for the parameters via the
    reparametrization trick.

    This is the standard variational posterior with reparametrization trick for Bayes-by-backprop variational inference
    (for more details, see paper "Weight Uncertainty in Neural Networks" by Blundell).
    """

    def __init__(
            self,
            param: Optional[Parameter | Tensor] = None,
            shape: Optional[_size] = None,
            dtype: Optional[_dtype] = None
    ) -> None:
        """
        Initializes a gaussian variational posterior distribution for a parameter.

        Parameters
        ----------
        param : Optional[Parameter | Tensor]
            The parameter or tensor to reparametrize with the variational posterior. It is used to specify shape and
            dtype. If None, then specified shape and dtype are used.
        shape : Optional[_size]
            The shape of the parameter or tensor to reparametrize with the variational posterior. If None, defaults to
            the shape of the input param.
        dtype : Optional[_dtype]
            The data type of the parameter or tensor to reparametrize with the variational posterior. If None, defaults
            to torch.float32 or the dtype of the input param.

        Raises
        ------
        ValueError
            If both 'param' and 'shape' arguments are given.
        ValueError
            If both 'param' and 'shape' arguments are None.
        """
        if param is not None and shape is not None:
            raise ValueError("Provide either 'param' or 'shape' argument, not both.")
        if param is None and shape is None:
            raise ValueError("Must provide either 'param' or 'shape' argument.")

        super().__init__()

        # Set shape and dtype
        if param is not None:
            shape = param.size()
            if dtype is None:
                dtype = param.dtype
        else:
            if dtype is None:
                dtype = torch.float32

        # Create empty variational parameters mu and rho
        self.mu = Parameter(torch.empty(size=shape, dtype=dtype))
        self.rho = Parameter(torch.empty(size=shape, dtype=dtype))

        self.reset_parameters()     # Initializes variational parameters

    def reset_parameters(self) -> None:
        """
        Initializes the variational parameters.

        TODO Normal or Uniform initialization ?
        """
        # TODO dummy initialization for the moment
        torch.nn.init.ones_(self.mu)
        torch.nn.init.ones_(self.rho)

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
