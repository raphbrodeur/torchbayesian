"""
    @file:              reparametrize.py
    @Author:            Raphael Brodeur, PyTorch

    @Creation Date:     07/2025
    @Last modification: 07/2025

    @Description:       This file contains the register_reparametrization() function which is used to replace a Torch
                        parameter or buffer by a variational posterior from which the parameter or buffer is sampled.
                        Wraps and modifies code from PyTorch's nn.utils.parametrize module.
"""

from copy import deepcopy
from typing import (
    Dict,
    Optional,
    Tuple,
    Union
)

import torch
import torch.nn as nn


# Are associated code segments dead code ? Kept in case if Torch internals interact with this.
_cache_enabled = 0
_cache: Dict[Tuple[int, str], Optional[torch.Tensor]] = {}


class Reparametrization(nn.Module):
    """
    This class wraps the parametrization and handles safe checks for the replacement by the variational posterior and
    its forward call.

    It is the type of module.reparametrizations[tensor_name] when module[tensor_name] has been reparametrized with
    register_parametrization().

    Notes
    -----
    This class is used internally by register_parametrization(). It shall not be instantiated by the user.
    """

    def __init__(
            self,
            parametrization: nn.Module,
            original: Union[nn.Parameter, torch.Tensor]
    ) -> None:
        """
        Initializes class and check if parametrization function fits with original tensor.

        Parameters
        ----------
        parametrization : nn.Module
            The reparametrization function; the variational posterior.
        original : Union[nn.Parameter, torch.Tensor]
            The tensor that is being reparametrized.
        """
        super().__init__()

        # TODO Check if safe; do original and parametrization() fit dtype, shape etc.
        ...

        self.variational_posterior: nn.Module = parametrization

    def forward(self) -> torch.Tensor:
        """
        Wraps the parametrization's forward call. Checks for scripting.
        """
        if torch.jit.is_scripting():
            raise RuntimeError("Reparametrization is not working with scripting.")

        x = self.variational_posterior()

        return x


def register_reparametrization(
    module: nn.Module,
    tensor_name: str,
    parametrization: nn.Module
) -> nn.Module:
    """
    Registers a reparametrization to a tensor in a module.

    Assume that tensor_name="weight" for simplicity. When accessing module.weight, the module will return the
    parametrized version parametrization(module.weight). If the original tensor requires a gradient, the backward pass
    will differentiate through attribute parametrization and the optimizer will update the tensor accordingly. The
    parameters or buffers of the parametrization are registered to the model and state_dict.

    Examples
    --------
    TODO
    """
    parametrization.train(module.training)

    # Set the parametrization mechanism
    if tensor_name in module._buffers or tensor_name in module._parameters:
        # Fetch the original buffer or parameter
        original = getattr(module, tensor_name)

        # Wrap the parametrization with Reparametrization to check for errors
        reparametrization = Reparametrization(parametrization, original)

        # Delete the previous parameter or buffer that is now reparametrized
        delattr(module, tensor_name)

        # If this is the first reparametrization registered on the module,
        # we prepare the module to inject the property
        if not is_reparametrized(module):
            # Sets up the module to be reparametrized. Adds prefix Reparametrized to module's name
            _inject_new_class(module)

            # Inject a ModuleDict into the instance under module.reparametrizations
            module.reparametrizations = nn.ModuleDict()

        # Add a property into the class
        _inject_property(module, tensor_name)

        # Add a ParametrizationList
        assert isinstance(module.reparametrizations, nn.ModuleDict)  # Make mypy happy
        module.reparametrizations[tensor_name] = reparametrization

    else:
        raise ValueError(
            f"Module '{module}' does not have a parameter, a buffer, or a "
            f"parametrized element with name '{tensor_name}'"
        )

    return module


def is_reparametrized(module: nn.Module, tensor_name: Optional[str] = None) -> bool:
    """
    Whether the module has a reparametrization.

    If tensor_name is specified, checks for the specified parameter or
    buffer, otherwise checks for all parameters and buffers in the module.

    Parameters
    ----------
    module : nn.Module
        The module to check for.
    tensor_name : Optional[str]
        The name of the parameter or buffer to check for within the module. Optional. Defaults to None.

    Returns
    -------
    is_reparametrized : bool
        Whether the module or the tensor is reparametrized.
    """
    reparametrizations = getattr(module, "reparametrizations", None)

    if reparametrizations is None or not isinstance(reparametrizations, nn.ModuleDict):
        return False

    if tensor_name is None:
        # Check that there is at least one parametrized buffer or Parameter
        return len(reparametrizations) > 0

    else:
        return tensor_name in reparametrizations


def _inject_new_class(module: nn.Module) -> None:
    """
    Sets up a module to be reparametrized.

    This works by substituting the class of the module by a class that extends it to be able to inject a property.
    Adds the prefix Reparametrized to the module's name.

    Parameters
    ----------
    module : nn.Module
        The module to prepare for reparametrization.
    """
    cls = module.__class__

    def default_deepcopy(self, memo):
        """
        Emulates a standard deepcopy procedure when __deepcopy__ doesn't exist in the current class.
        """
        obj = memo.get(id(self), None)

        if obj is not None:
            return obj

        replica = self.__new__(self.__class__)
        memo[id(self)] = replica
        replica.__dict__ = deepcopy(self.__dict__, memo)

        # Also save all slots if they exist.
        slots_to_save = copyreg._slotnames(self.__class__)  # type: ignore[attr-defined]
        for slot in slots_to_save:
            if hasattr(self, slot):
                setattr(replica, slot, deepcopy(getattr(self, slot), memo))

        return replica

    def getstate(self):
        raise RuntimeError(
            "Serialization of parametrized modules is only "
            "supported through state_dict(). See:\n"
            "https://pytorch.org/tutorials/beginner/saving_loading_models.html"
            "#saving-loading-a-general-checkpoint-for-inference-and-or-resuming-training"
        )

    dct = {"__getstate__": getstate}

    # We don't allow serialization of parametrized modules but should still allow deepcopying.
    # Default 'deepcopy' function invokes __deepcopy__ method instead of __getstate__ when it exists.
    if not hasattr(cls, "__deepcopy__"):
        dct["__deepcopy__"] = default_deepcopy  # type: ignore[assignment]

    param_cls = type(
        f"Reparametrized{cls.__name__}",
        (cls,),
        dct,
    )

    module.__class__ = param_cls


def _inject_property(module: nn.Module, tensor_name: str) -> None:
    """
    Injects a property into module[tensor_name].

    It assumes that the class in the module has already been modified from its original one using _inject_new_class and
    that the tensor under :attr:`tensor_name` has already been moved out.

    Same as in PyTorch's implementation but the getter function is modified to get reparametrization instead of
    parametrization.

    Parameters
    ----------
    module : nn.Module
        The module into which a property is injected.
    tensor_name : str
        The name of the property to create.
    """
    # We check if an attribute already exists under that name.
    # This should never fire if register_parametrization is correctly implemented
    assert not hasattr(module, tensor_name)

    @torch.jit.unused
    def get_cached_reparametrization(reparametrization) -> torch.Tensor:
        global _cache
        key = (id(module), tensor_name)
        tensor = _cache.get(key)
        if tensor is None:
            tensor = reparametrization()
            _cache[key] = tensor
        return tensor

    def get_parametrized(self) -> torch.Tensor:
        if torch.jit.is_scripting():
            raise RuntimeError("Parametrization is not working with scripting.")

        reparametrization = self.reparametrizations[tensor_name]

        if _cache_enabled:
            if torch.jit.is_scripting():
                # Scripting
                raise RuntimeError(
                    "Caching is not implemented for scripting. "
                    "Either disable caching or avoid scripting."
                )
            elif torch._C._get_tracing_state() is not None:
                # Tracing
                raise RuntimeError(
                    "Cannot trace a model while caching parametrizations."
                )
            else:
                return get_cached_reparametrization(reparametrization)

        else:
            # If caching is not active, this function just evaluates the parametrization
            return reparametrization()

    def set_original(self, value: torch.Tensor) -> None:
        if torch.jit.is_scripting():
            raise RuntimeError("Parametrization is not working with scripting.")

        self.reparametrizations[tensor_name].right_inverse(value)

    setattr(module.__class__, tensor_name, property(get_parametrized, set_original))
