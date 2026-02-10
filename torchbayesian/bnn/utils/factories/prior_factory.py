"""
    @file:              prior_factory.py
    @Author:            Raphael Brodeur

    @Creation Date:     08/2025
    @Last modification: 02/2026

    @Description:       This file contains the prior factory 'PriorFactory' which is used to get 'Prior' classes from
                        the prior factory 'PriorFactory', and the function 'get_prior()' which is used by
                        'bnn.BayesianModule' to get and instantiate 'Prior' classes registered to 'PriorFactory' given
                        the parameter shape and the prior's name and optional keyword arguments.
"""

from typing import (
    Any,
    Dict,
    Optional,
    Tuple,
    Type
)

from torch.types import (
    Device,
    _dtype,
    _size
)

from torchbayesian.bnn.priors import GaussianPrior, Prior
from torchbayesian.bnn.utils.factories.factory import Factory


__all__ = ["get_prior", "PriorFactory"]


# Create a 'Prior' Factory() and register some factory functions to it
PriorFactory = Factory()


@PriorFactory.register_factory_function("gaussian")
def gaussian_prior_factory() -> Type[GaussianPrior]:
    return GaussianPrior


@PriorFactory.register_factory_function("normal")
def normal_prior_factory() -> Type[GaussianPrior]:
    return GaussianPrior


# Create a 'Prior' instantiation function
def get_prior(
        shape: _size,
        prior: str | Tuple[str, Dict[str, Any]],
        *,
        dtype: Optional[_dtype] = None,
        device: Device = None
) -> Prior:
    """
    Creates an instance of a 'Prior' subclass. For use in 'bnn.BayesianModule'.

    Parameters
    ----------
    shape : _size
        The shape of the tensor for which to initialize a prior.
    prior : str | Tuple[str, Dict[str, Any]]
        The prior to instantiate. Either the prior's name (str) or a tuple of the name and a dictionary of keyword
        arguments for instantiation.
    dtype : Optional[_dtype]
        The dtype of the tensor for which to initialize a prior. Optional. Defaults to torch's default dtype.
    device : Device
        The device of the parameter or tensor for which to initialize a prior. Optional. Defaults to torch's default
        device.

    Returns
    -------
    prior_instance : Prior
        The instantiated prior.
    """
    if isinstance(prior, str):
        prior_name = prior
        prior_kwargs = {}
    else:
        prior_name, prior_kwargs = prior

    prior_type = PriorFactory[prior_name]   # Get appropriate prior class using factory
    prior_instance = prior_type(            # Instantiate said prior class
        shape=shape,
        dtype=dtype,
        device=device,
        **prior_kwargs
    )

    return prior_instance
