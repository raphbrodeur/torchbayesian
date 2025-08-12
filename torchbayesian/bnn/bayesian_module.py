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
import torch.distributions as D
from torch.distributions import Distribution
from torch.nn import Module

from torchbayesian.bnn.priors import GaussianPrior, Prior
from torchbayesian.bnn.utils import register_reparametrization
from torchbayesian.bnn.variational_posteriors import GaussianPosterior, VariationalPosterior


class BayesianModule(Module):
    """
    This class is a Torch nn.Module container used to reparametrize the parameters of any torch model or module with
    some variational posterior like for Bayes-by-Backprop approximate variational inference as in the paper "Weight
    Uncertainty in Neural Networks" by Blundell et al.

    Should be called before weights are registered to optimizer, otherwise, manually register new parameters to
    optimizer.

    Notes
    -----
    By default, a gaussian variational posterior distribution and a gaussian prior distribution are used. This allows
    to evaluate KL divergence between the two in closed form and is somewhat of a standard for BBB variational
    inference, even though the original paper proposes a scale mixture of gaussians as prior.
    """

    def __init__(
            self,
            module: Module,
            variational_posterior: Optional[VariationalPosterior] = None,
            prior: Optional[Prior] = None,
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
        prior : Optional[Prior]
            The distribution for the parameters. Defaults to GaussianPrior.
        debug : bool
            Whether to print debug messages. Defaults to False.

        Notes
        -----
        By default, GaussianPosterior and GaussianPrior are used as the variational posterior distribution and the prior
        distribution for the parameters. This allows simple close form evaluation of the KL divergence between the two.
        """
        super().__init__()

        if variational_posterior is None:
            variational_posterior = GaussianPosterior       # TODO use factory logic
        if prior is None:
            self._prior = GaussianPrior                       # TODO use factory logic

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

    def _MC_approx_kl_divergence(self, num_samples: int) -> Tensor:
        """
        Computes the KL divergence KL(posterior || prior) of the BayesianModule through a Monte Carlo approximation.

        Parameters
        ----------
        num_samples : int
            The number of samples for the MC approximation of the KL divergence. A common value is in the range of TODO

        Returns
        -------
        kl_div : Tensor
            The MC-approximate KL divergence KL(posterior || prior) of the BayesianModule.
        """
        pass

    def _compute_kl_divergence(
            self,
            variational_posterior: Distribution,
            prior: Distribution,
            num_samples: Optional[int] = None
    ) -> Tensor:
        """
        Gets the KL divergence KL(posterior || prior) of a parameter from the BayesianModule.

        Notes
        -----
        If closed-form evaluation is not defined, defaults to MC approximation of the KL divergence by sampling from the
        posterior and prior.

        Parameters
        ----------
        variational_posterior : Distribution
            The variational posterior distribution.
        prior : Distribution
            The prior distribution.
        num_samples : Optional[int]
            The number of samples for the MC approximation of the KL divergence. Defaults to None. A common value is in
            the range of TODO...

        Returns
        -------
        kl_div : Tensor
            The KL divergence KL(posterior || prior) of the BayesianModule.

        Raises
        ------
        ValueError
            TODO if no num samples is specified but its needed
        """
        # If analytical solution exists
        kl = D.kl_divergence(variational_posterior, prior)

        # Else, use MC approximation
        ...

        return kl

    def kl_divergence(self, num_samples: Optional[int] = None) -> Tensor:
        """
        Gets the KL divergence KL(posterior || prior) of all parameters in the BayesianModule.

        Notes
        -----
        If closed-form evaluation is not defined, defaults to MC approximation of the KL divergence.

        Parameters
        ----------
        num_samples : Optional[int]
            The number of samples for the MC approximation of the KL divergence. Defaults to None. A common value is in
            the range of TODO...

        Returns
        -------
        kl_div : Tensor
            The KL divergence KL(posterior || prior) of the BayesianModule.
        """
        for name, module in self.named_modules():
            if isinstance(module, VariationalPosterior):    # Compute KL divergence for BNN modules
                posterior_dist = module.distribution
                prior_dist = self._prior(shape=posterior_dist.batch_shape).distribution

                # Compute KL divergence
                kl = self._compute_kl_divergence(
                    variational_posterior=posterior_dist,
                    prior=prior_dist,
                    num_samples=num_samples
                )

                ...
