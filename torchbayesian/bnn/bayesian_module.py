"""
    @file:              bayesian_module.py
    @Author:            Raphael Brodeur

    @Creation Date:     07/2025
    @Last modification: 08/2025

    @Description:       This file contains the BayesianModule class which is a Torch nn.Module container used to
                        reparametrize the parameters of any torch model or module with some variational posterior.
"""

from typing import Optional

from torch import Tensor
from torch.nn import Module

from torchbayesian.bnn.utils import register_reparametrization
from torchbayesian.bnn.variational_posteriors import GaussianPosterior, VariationalPosterior


class BayesianModule(Module):
    """
    This class is a Torch nn.Module container used to reparametrize the parameters of any torch model or module with
    some variational posterior like for Bayes-by-Backprop approximate variational inference as in the paper "Weight
    Uncertainty in Neural Networks" by Blundell et al.

    Should be called before weights are registered to optimizer, otherwise, manually register new parameters to
    optimizer.
    """

    def __init__(
            self,
            module: Module,
            variational_posterior: Optional[VariationalPosterior] = None,
            debug: bool = False
    ) -> None:
        """
        Makes the module a bayesian neural network with a variational distribution over each parameter. Reparametrizes
        the parameters of any torch model or module with some variational posterior like for Bayes-by-Backprop
        approximate variational inference as in the paper "Weight Uncertainty in Neural Networks" by Blundell et al.

        Parameters
        ----------
        module: Module
            The module to make bayesian.
        variational_posterior : Optional[VariationalPosterior]
            The variational posterior distribution for the parameters. Defaults to GaussianPosterior.
        debug : bool
            Whether to print debug messages. Defaults to False.
        """
        super().__init__()

        if variational_posterior is None:
            variational_posterior = GaussianPosterior

        # Replace every parameter in the model by an instance of the variational posterior
        for name, param in list(module.named_parameters()):
            # Get name of parameter and name of the module owning it
            if "." in name:
                owner_module_name, param_name = name.rsplit('.', 1)
            else:
                owner_module_name, param_name = ("", name)

            # Register reparametrization to parameter
            if debug:
                print(
                    f"Registering to module {module.get_submodule(owner_module_name)}'s {param_name}..."
                )
            register_reparametrization(
                module=module.get_submodule(owner_module_name),
                tensor_name=param_name,
                parametrization=variational_posterior(param)
            )

        self.module = module

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass of the BNN.

        Parameters
        ----------
        input : Tensor
            The input tensor.

        Returns
        -------
        output : Tensor
            The output tensor.
        """
        return self.module(input)
