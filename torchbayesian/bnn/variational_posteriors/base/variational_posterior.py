"""
    @file:              variational_posterior.py
    @Author:            Raphael Brodeur

    @Creation Date:     07/2025
    @Last modification: 08/2025

    @Description:       This file contains the VariationalPosterior base class for all variational posteriors used for
                        Bayes-by-backprop variational inference.
"""

from abc import ABC, abstractmethod

from torch import Tensor
from torch.distributions import Distribution
from torch.nn import Module


__all__ = ["VariationalPosterior"]


class VariationalPosterior(Module, ABC):
    """
    This class serves as a base class for all variational posteriors used for Bayes-by-backprop variational inference.

    Notes
    -----
    Subclasses of VariationalPosterior used in BayesianModule must accept param: Union[nn.Parameter, torch.Tensor] as
    argument in their constructor __init__ method.

    In the constructor __init__ methods of subclasses of VariationalPosterior:
    (1) Call  super().__init__() then;
    (2) Create empty variational parameters with appropriate size. e.g. self.mu = nn.Parameter(torch.empty(...)) then;
    (3) Call self.reset_parameters() at the end of __init__ to initialize the variational parameters.
    """

    @abstractmethod
    def reset_parameters(self) -> None:
        """
        Initializes the variational parameters.
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def distribution(self) -> Distribution:
        """
        A torch.Distribution corresponding to the variational posterior. This is used for KL computation aligned
        with torch's framework.

        Returns
        -------
        distribution : Distribution
            A torch.Distribution.
        """
        raise NotImplementedError

    @abstractmethod
    def sample_parameters(self) -> Tensor:
        """
        Samples a value for the parameters from the variational posterior distribution (itself parametrized by the
        variational parameters).

        Returns
        -------
        param : Tensor
            A value of the parameter, sampled from the variational posterior.
        """
        raise NotImplementedError

    def forward(self) -> Tensor:
        """
        Forward call for the reparametrization module.

        Samples the parameters from the variational posterior distribution (itself parametrized by the variational
        parameters).

        Returns
        -------
        param : Tensor
            A value of the parameter, sampled from the variational posterior.
        """
        return self.sample_parameters()
