"""
    @file:              reparametrize.py
    @Author:            Raphael Brodeur, PyTorch

    @Creation Date:     07/2025
    @Last modification: 08/2025

    @Description:       This file contains the 'register_reparametrization()' function which is used to replace a Torch
                        parameter or buffer by a variational posterior 'nn.Module' from which the parameter or buffer is
                        sampled. Adapted from PyTorch's 'nn.utils.parametrize' in order to remove the original parameter
                        from the registered parameters and the state dict.
"""

from copy import deepcopy
from typing import (
    Dict,
    Optional,
    Tuple
)

import torch
from torch import Tensor
from torch.nn import (
    Module,
    ModuleDict,
    Parameter
)


__all__ = ["register_reparametrization"]


_cache_enabled = 0
_cache: Dict[Tuple[int, str], Optional[Tensor]] = {}


class Reparametrization(Module):
    """
    This class wraps the parametrization and handles safety checks for the replacement by the variational posterior and
    its forward call.

    It is the type of 'module.reparametrizations[tensor_name]' when 'module[tensor_name]' has been reparametrized with
    'register_parametrization()'.

    Notes
    -----
    This class is used internally by 'register_parametrization()'. It shall not be instantiated by the user.
    """

    def __init__(
            self,
            parametrization: Module,
            original: Parameter | Tensor,
            unsafe: bool = False
    ) -> None:
        """
        Wraps a reparametrization and checks if the parametrization function fits with original tensor.

        Parameters
        ----------
        parametrization : Module
            The parametrization function; the variational posterior in the context of Bayes-by-backprop.
        original : Parameter | Tensor
            The tensor that is being reparametrized.
        unsafe : bool
            Whether to bypass correctness checks. Defaults to False.

        Raises
        ------
        TypeError
            If unsafe is set to False and the parametrization's forward call does not return a Tensor.
        TypeError
            If unsafe is set to False and the parametrization's forward call does not return a Tensor with the same
            dtype as the original tensor.
        ValueError
            If unsafe is set to False and the parametrization's forward call does not return a Tensor with the same
            shape as the original tensor.
        """
        super().__init__()

        # Correctness checks
        if not unsafe:
            new = parametrization()     # Get output tensor from reparametrization

            # Check if parametrization's forward call returns a Tensor
            if not isinstance(new, Tensor):
                raise TypeError(
                    f"A reparametrization must return a tensor. Got {type(new).__name__}."
                )

            # Check if dtypes match
            if new.dtype != original.dtype:
                raise TypeError(
                    "Reparametrization may not change the dtype of the tensor, unless the unsafe flag is enabled.\n"
                    f"Original tensor has dtype: {original.dtype}\n"
                    f"Reparametrization returns dtype: {new.dtype}"
                )

            # Check if shapes match
            if new.shape != original.shape:
                raise ValueError(
                    "Reparametrization may not change the shape of the tensor, unless the `unsafe` flag is enabled.\n"
                    f"Original tensor has shape: {original.shape}\n"
                    f"Reparametrization returns shape: {new.shape}"
                )

        self.variational_posterior: Module = parametrization

    def forward(self) -> Tensor:
        """
        Wraps the parametrization's forward call. Checks for scripting.
        """
        if torch.jit.is_scripting():
            raise RuntimeError("Reparametrization is not working with scripting.")

        x = self.variational_posterior()

        return x

    def extra_repr(self) -> str:
        """
        Returns the variational posterior put in place.

        Returns
        -------
        extra_repr : str
            The str extra representation of the reparametrization.
        """
        return f"{self.variational_posterior}"

    def __repr__(self) -> str:
        """
        Representation of the reparametrization.

        Overwrites nn.Module.__repr__ in order to supress child modules representation.

        Returns
        -------
        repr : str
            The str representation of the reparametrization.
        """
        return f"{self.__class__.__name__}({self.extra_repr()})"


def register_reparametrization(
    module: Module,
    tensor_name: str,
    parametrization: Module
) -> Module:
    """
    Registers a reparametrization to a tensor (parameter or buffer) in a module.

    Assume that tensor_name="weight" for simplicity. When accessing module.weight, the module will return the
    parametrized version parametrization(module.weight). If the original tensor requires a gradient, the backward pass
    will differentiate through attribute parametrization and the optimizer will update the tensor accordingly. The
    parameters or buffers of the parametrization are registered to the model and state_dict.

    Examples
    --------
        mod = nn.Linear(2, 4)
        # This module has parameters "mod.bias" and "mod.weight"

        register_reparametrization(mod, "weight", bnn.GaussianPosterior)
        # The module "mod" now has parameters "mod.bias", "mod.reparametrizations.weight.GaussianPosterior.mu"
        # and "mod.reparametrizations.weight.GaussianPosterior.rho"
    """
    parametrization.train(module.training)  # Put parametrization in same training mode as module

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
            module.reparametrizations = ModuleDict()

        # Add a property into the class
        _inject_property(module, tensor_name)

        # Add a Reparametrization
        assert isinstance(module.reparametrizations, ModuleDict)  # Make mypy happy
        module.reparametrizations[tensor_name] = reparametrization

    else:
        raise ValueError(
            f"Module '{module}' does not have a parameter, a buffer, or a "
            f"parametrized element with name '{tensor_name}'"
        )

    return module


def is_reparametrized(module: Module, tensor_name: Optional[str] = None) -> bool:
    """
    Whether the module has a reparametrization.

    If tensor_name is specified, checks for the specified parameter or
    buffer, otherwise checks for all parameters and buffers in the module.

    Parameters
    ----------
    module : Module
        The module to check for.
    tensor_name : Optional[str]
        The name of the parameter or buffer to check for within the module. Optional. Defaults to None.

    Returns
    -------
    is_reparametrized : bool
        Whether the module or the tensor is reparametrized.
    """
    reparametrizations = getattr(module, "reparametrizations", None)

    if reparametrizations is None or not isinstance(reparametrizations, ModuleDict):
        return False

    if tensor_name is None:
        # Check that there is at least one parametrized buffer or Parameter
        return len(reparametrizations) > 0

    else:
        return tensor_name in reparametrizations


def _inject_new_class(module: Module) -> None:
    """
    Sets up a module to be reparametrized.

    This works by substituting the class of the module by a class that extends it to be able to inject a property.
    Adds the prefix Reparametrized to the module's name.

    Parameters
    ----------
    module : Module
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

    # We don't allow serialization of parametrized modules but should still allow deep-copying.
    # Default 'deepcopy' function invokes __deepcopy__ method instead of __getstate__ when it exists.
    if not hasattr(cls, "__deepcopy__"):
        dct["__deepcopy__"] = default_deepcopy  # type: ignore[assignment]

    param_cls = type(
        f"Reparametrized{cls.__name__}",
        (cls,),
        dct,
    )

    module.__class__ = param_cls


def _inject_property(module: Module, tensor_name: str) -> None:
    """
    Injects a property into module[tensor_name].

    This function is the core of the reparametrization mechanism.

    It assumes that the class in the module has already been modified from its original one using _inject_new_class and
    that the tensor under :attr:`tensor_name` has already been moved out.

    Same as in PyTorch's implementation but the getter function is modified to get reparametrization instead of
    parametrization, and the setter function is modified to ...

    Parameters
    ----------
    module : Module
        The module into which a property is injected.
    tensor_name : str
        The name of the property to create.
    """
    # We check if an attribute already exists under that name.
    # This should never fire if register_parametrization is correctly implemented
    assert not hasattr(module, tensor_name)

    @torch.jit.unused
    def get_cached_reparametrization(reparametrization) -> Tensor:
        global _cache
        key = (id(module), tensor_name)
        tensor = _cache.get(key)
        if tensor is None:
            tensor = reparametrization()
            _cache[key] = tensor
        return tensor

    def get_parametrized(self) -> Tensor:
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

    def set_original(self, value: Tensor) -> None:
        if torch.jit.is_scripting():
            raise RuntimeError("Parametrization is not working with scripting.")

        self.reparametrizations[tensor_name].right_inverse(value)

    setattr(module.__class__, tensor_name, property(get_parametrized, set_original))
