"""
    @file:              posterior_factory.py
    @Author:            Raphael Brodeur

    @Creation Date:     08/2025
    @Last modification: 02/2026

    @Description:       This file contains the variational factory 'PosteriorFactory' which is used to get
                        'VariationalPosterior' classes from the variational posterior factory 'PosteriorFactory', and
                        the function 'get_posterior()' which is used by 'bnn.BayesianModule' to get and instantiate
                        'VariationalPosterior' classes registered to 'PosteriorFactory' given the parameter to replace
                        and the variational posterior's name and optional keyword arguments.
"""

from typing import (
    Any,
    Dict,
    Tuple,
    Type
)

from torch import Tensor

from torchbayesian.bnn.utils.factories.factory import Factory
from torchbayesian.bnn.variational_posteriors import GaussianPosterior, VariationalPosterior


__all__ = [
    "get_posterior",
    "get_variational_posterior",
    "PosteriorFactory",
    "VariationalPosteriorFactory"
]


# Create a 'VariationalPosterior' Factory() and register some factory functions to it
PosteriorFactory = Factory()


@PosteriorFactory.register_factory_function("gaussian")
def gaussian_posterior_factory() -> Type[GaussianPosterior]:
    return GaussianPosterior


@PosteriorFactory.register_factory_function("normal")
def normal_posterior_factory() -> Type[GaussianPosterior]:
    return GaussianPosterior


VariationalPosteriorFactory = PosteriorFactory


# Create a 'VariationalPosterior' instantiation function
def get_posterior(param: Tensor, posterior: str | Tuple[str, Dict[str, Any]]) -> VariationalPosterior:
    """
    Creates an instance of a 'VariationalPosterior' subclass to replace a given tensor. For use in 'bnn.BayesianModule'.

    Parameters
    ----------
    param : Tensor
        The tensor for which to initialize a variational posterior.
    posterior : str | Tuple[str, Dict[str, Any]]
        The variational posterior to instantiate. Either the posterior's name (str) or a tuple of the name and a
        dictionary of keyword arguments for instantiation.

    Returns
    -------
    posterior_instance : VariationalPosterior
        The instantiated variational posterior.
    """
    if isinstance(posterior, str):
        posterior_name = posterior
        posterior_kwargs = {}
    else:
        posterior_name, posterior_kwargs = posterior

    posterior_type = PosteriorFactory[posterior_name]   # Get appropriate posterior class using factory
    posterior_instance = posterior_type.from_param(param=param, **posterior_kwargs)   # Instantiate said posterior class

    return posterior_instance


get_variational_posterior = get_posterior
