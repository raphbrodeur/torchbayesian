"""
    @file:              variational_posterior.py
    @Author:            Raphael Brodeur

    @Creation Date:     07/2025
    @Last modification: 02/2026

    @Description:       This file contains the 'VariationalPosterior' base class for all variational posteriors used for
                        Bayes by Backprop (BBB) variational inference (VI).
"""

from abc import ABC, abstractmethod
from typing import (
    Optional,
    Type,
    TypeVar
)

import torch
from torch import Tensor
from torch.distributions import Distribution
from torch.nn import Module
from torch.types import (
    Device,
    _dtype,
    _size
)


__all__ = ["VariationalPosterior"]


T = TypeVar("T", bound="VariationalPosterior")  # PEP 673 style 'Self' type for class methods constructors


class VariationalPosterior(Module, ABC):
    """
    This class serves as a base class for all variational posteriors used for Bayes by Backprop (BBB) variational
    inference (VI).

    Parameters
    ----------
    shape : _size
        The shape of the parameter replaced by the variational posterior.
    dtype : Optional[_dtype]
        The dtype of the parameter replaced by the variational posterior. Optional. Defaults to torch's default dtype.
    device : Optional[Device]
        The device of the parameter replaced by the variational posterior. Optional. Defaults to torch's default device.

    Attributes
    ----------
    shape : _size
        The shape of the tensors sampled from the variational posterior.

    Notes
    -----
    Subclasses used in 'bnn.BayesianModule' must work with 'get_posterior()'; see 'from_param' constructor class method.

    Recommended PyTorch-esque pattern for the constructor ('__init__' method) of custom subclasses of
    'VariationalPosterior':
    (1) Call  'super().__init__(...)' then;
    (2) Create empty variational parameters with appropriate size. e.g. 'self.mu = nn.Parameter(torch.empty(...))' then;
    (3) Call a method 'self.reset_parameters()' at the end of '__init__' to initialize the variational parameters.
    """

    shape: _size
    _posterior_meta: Tensor

    def __init__(
            self,
            shape: _size,
            *,
            dtype: Optional[_dtype] = None,
            device: Optional[Device] = None,
    ) -> None:
        """
        Initializes the variational posterior.

        Tracks the supposed attributes of the replaced parameter through calls to '_apply' (e.g. '.to(dtype)',
        '.cuda()', etc.) with a dummy buffer '_posterior_meta'. This is used to instantiate priors fitting the
        variational posterior.

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
        super().__init__()

        # Track replaced parameter's supposed shape, dtype and device.
        # Does so by registering an empty buffer so that calls to '.to(device, dtype)' update the replaced parameter's
        # supposed shape, dtype and device.
        # This is used to instantiate priors fitting the posterior for KL divergence evaluation.
        # Another (perhaps cleaner/more efficient ?) approach would be to overwrite/wrap the '_apply(fn)' method, which
        # is called by '.to()', '.cuda()', etc., so that it updates attributes 'self.dtype' and 'self.device',
        # e.g. with a probe:
        # def _apply(fn, ...)
        #    super()._apply(fn, ...)
        #    probe = fn(torch.empty(0, dtype=self.dtype, device=self.device))
        #    self.dtype, self.device = probe.dtype, probe.device
        # TODO -- Would this be more efficient ? Perhaps if the number of VariationalPosterior's is quite large...
        self.shape = shape
        self.register_buffer("_posterior_meta", torch.empty(0, dtype=dtype, device=device), persistent=False)

    @classmethod
    def from_param(cls: Type[T], param: Tensor, **kwargs) -> T:
        """
        Alternate constructor used by the 'bnn.BayesianModule' factory logic.

        Instantiates a variational posterior given the parameter to be replaced. This construction path is used by
        'get_posterior' inside 'bnn.BayesianModule'.

        The default implementation constructs the posterior using only the parameter's shape, dtype and device.
        Subclasses of 'VariationalPosterior' that require further access to 'param' should override this method.

        Parameters
        ----------
        param : Tensor
            The tensor (parameter or buffer) being replaced by a variational posterior.
        **kwargs
            Additional keyword arguments passed along to the variational posterior constructor.

        Returns
        -------
        posterior_instance : Self
            An instance of the variational posterior class.
        """
        return cls(shape=param.shape, dtype=param.dtype, device=param.device, **kwargs)     # type: ignore[arg-type]

    @property
    def dtype(self) -> _dtype:
        """
        The replaced parameter's supposed dtype.
        """
        return self._posterior_meta.dtype

    @property
    def device(self) -> Device:
        """
        The replaced parameter's supposed device.
        """
        return self._posterior_meta.device

    @property
    @abstractmethod
    def distribution(self) -> Distribution:
        """
        An element-wise torch.Distribution corresponding to the variational posterior. This is used for KL computation
        aligned with torch's framework.

        Shape should be the same as the shape of the parameters sampled from the variational posterior.

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
