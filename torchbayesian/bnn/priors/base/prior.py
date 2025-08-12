"""
    @file:              prior.py
    @Author:            Raphael Brodeur

    @Creation Date:     08/2025
    @Last modification: 08/2025

    @Description:       This file contains the Prior base class for all parameter prior distributions used for
                        Bayes-by-backprop variational inference.
"""

from abc import ABC, abstractmethod

from torch.distributions import Distribution


class Prior(ABC):
    """
    This class serves as a base class for all priors used for Bayes-by-backprop variational inference.

    Base class implemented for consistency in the case of possible future implementation of learnable priors and making
    Prior also a subclass of Module.
    """

    @property
    @abstractmethod
    def distribution(self) -> Distribution:
        """
        A torch.distributions.Distribution corresponding to the prior. This is used for KL computation aligned with
        torch's framework.

        Returns
        -------
        distribution : Distribution
            A torch.Distribution.
        """
        raise NotImplementedError
