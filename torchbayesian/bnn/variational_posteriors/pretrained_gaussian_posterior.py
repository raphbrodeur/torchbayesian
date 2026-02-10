"""
    @file:              pretrained_gaussian_posterior.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2026
    @Last modification: 02/2026

    @Description:       This file contains the 'PretrainedGaussianPosterior' class, a diagonal Gaussian variational
                        posterior whose mean parameter is initialized to the original value of the tensor. Used for
                        Bayes by Backprop (BBB) variational inference (VI).
"""

import torch
from torch import Tensor
from torch.distributions import Normal
from torch.nn import Parameter
import torch.nn.functional as F

from torchbayesian.bnn.variational_posteriors.base import VariationalPosterior


__all__ = ["PretrainedGaussianPosterior", "PretrainedNormalPosterior"]


class PretrainedGaussianPosterior(VariationalPosterior):
    """
    This class is a diagonal Gaussian variational posterior whose mean parameter is initialized from an existing tensor.

    Samples tensors via the reparametrization trick.

    This posterior is useful when converting a pretrained model into a Bayesian neural network (BNN) via Bayes by
    Backprop (BBB) variational inference (VI), as it initializes the Gaussian variational distribution centered on the
    pretrained weights.

    Parameters
    ----------
    param : Tensor
        The tensor (e.g. pretrained parameter) to initially center the posterior distribution on.

    Attributes
    ----------
    mu : Parameter
        The variational parameter of the mean of the distribution.
    rho : Parameter
        The variational parameter that parametrizes the standard deviation of the distribution via softplus.
    """

    mu: Parameter
    rho: Parameter

    def __init__(self, param: Tensor) -> None:
        """
        Initializes a diagonal Gaussian variational posterior centered on a parameter.

        Parameters
        ----------
        param : Tensor
            The tensor (e.g. pretrained parameter) to initially center the posterior distribution on.
        """
        shape = param.shape
        dtype = param.dtype
        device = param.device

        super().__init__(shape=shape, dtype=dtype, device=device)

        # Create variational parameters 'mu' and 'rho'
        self.mu = Parameter(param.detach().clone())     # Initialized to same values as 'param'
        self.rho = Parameter(torch.empty(size=shape, dtype=dtype, device=device))

        self.reset_parameters()     # Initialize variational parameters

    @classmethod
    def from_param(cls, param: Tensor, **kwargs) -> "PretrainedGaussianPosterior":
        """
        Alternate constructor used by 'get_posterior' inside 'bnn.BayesianModule'.

        Overrides the default 'from_param' constructor of 'VariationalPosterior' to pass along 'param'.

        Parameters
        ----------
        param : Tensor
            The tensor (parameter or buffer) being replaced by a variational posterior.
        **kwargs
            Additional keyword arguments passed along to the variational posterior constructor.

        Returns
        -------
        posterior_instance : PretrainedGaussianPosterior
            An instance of the class.
        """
        return cls(param=param, **kwargs)     # type: ignore[arg-type]

    def reset_parameters(self) -> None:
        """
        Initializes the variational parameter 'rho' of the Gaussian posterior N(mu, sigma), where sigma = softplus(rho).
        """
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


PretrainedNormalPosterior = PretrainedGaussianPosterior
