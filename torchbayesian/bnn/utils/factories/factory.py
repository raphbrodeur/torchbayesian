"""
    @file:              factory.py
    @Author:            Raphael Brodeur

    @Creation Date:     08/2025
    @Last modification: 01/2026

    @Description:       This file contains the 'Factory' base class used to instantiate object factories and register
                        factory functions to said instances.
"""

from typing import (
    Any,
    Callable,
    Dict,
    Tuple
)


__all__ = ["Factory"]


class Factory:
    """
    This class serves as a base for factories.

    This class serves as a dynamic registry of factory functions so that new factory functions can be registered to
    instances of this class with the decorator register_factory_function(). This allows users, for example, to register
    a custom-made 'Prior' to the API and call it through the typical torchbayesian pipeline.

    Examples
    --------
        # Create an instance of 'Factory' to serve as a normalization layer factory
        Norm = Factory()

        # Register to the normalization factory 'Norm' a factory function that gets batch normalization layers:

        @Norm.register_factory_function("batch")
        def batch_norm_factory(dim) -> BatchNorm:
            types = (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)
            return types[dim - 1]

        # Now, one can get from the normalization factory 'Norm' to get a batch normalization layer
        batch_norm_layer = Norm["batch", norm_dim]

        # e.g. batch_norm_layer is nn.BatchNorm3d and not nn.BatchNorm3d()
    """

    def __init__(self) -> None:
        """
        Initializes a factory. Creates a registry of factory functions (functions that get some class without
        instantiating it).
        """
        # Create a registry where factory functions are to be keyed to names
        self._factories: Dict[str, Callable] = {}

    def __getitem__(self, args: str | Tuple) -> Any:
        """
        Gets an object class for a given name or a tuple of a name and factory function arguments.

        Does so by getting the factory function from the registry of factory functions with the name given, and by then
        calling the said factory function using the remaining arguments.

        Parameters
        ----------
        args : str | Tuple
            The arguments needed to get the object class. The arguments specify which factory function to get from the
            registered factory functions and also additional arguments can be used to said factory function.

        Returns
        -------
        object_class : Any
            The object class. That is, not an instance of the object but its actual class. e.g. returns the class
            nn.Conv3d, not an instance nn.Conv3d().
        """
        # If 'args' is a factory name
        if isinstance(args, str):
            factory_name = args
            args = ()               # () so that *args gives empty tuple

        # If 'args' is a factory name with arguments
        else:
            factory_name, *args = args  # Remaining elements of 'args' are unpacked with * into the new list 'args'

        factory_function = self._factories[factory_name.upper()]    # Gets the factory function
        object_class = factory_function(*args)                      # Calls said factory function

        return object_class

    @property
    def factories(self) -> Tuple[str, ...]:
        """
        The names of the factory functions registered to the instance.
        """
        return tuple(self._factories)

    def _add_factory_function(
            self,
            name: str,
            func: Callable
    ) -> None:
        """
        Adds a factory function to the instance under a given name.

        Parameters
        ----------
        name : str
            The name of the factory function.
        func : Callable
            The factory function to add.

        Raises
        ------
        ValueError
            If a factory function is already registered to the instance under the same name.
        """
        if name.upper() in self.factories:
            raise ValueError(f"There is already a factory function registered under the name {name}.")

        self._factories[name.upper()] = func

    def register_factory_function(self, name: str) -> Callable:
        """
        This decorator adds/registers the decorated factory function to the instance under a given name.

        Parameters
        ----------
        name : str
            The name of the factory function

        Returns
        -------
        _wrapper : Callable
            The decorated function.
        """
        def _wrapper(func: Callable) -> Callable:
            self._add_factory_function(name=name, func=func)
            return func

        return _wrapper
