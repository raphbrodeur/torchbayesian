"""
    @file:              dropout.py
    @Author:            Raphael Brodeur

    @Creation Date:     01/2026
    @Last modification: 02/2026

    @Description:       This file contains Torch dropout layers for Bayesian neural networks (BNN) via Monte Carlo
                        dropout (MCD), as described in "Dropout as a Bayesian Approximation: Representing Model
                        Uncertainty in Deep Learning" by Y. Gal and Z. Ghahramani.
"""

from torch import Tensor
from torch.nn.modules.dropout import _DropoutNd
import torch.nn.functional as F


__all__ = [
    "BayesianAlphaDropout",
    "BayesianDropout",
    "BayesianDropout1d",
    "BayesianDropout2d",
    "BayesianDropout3d",
    "BayesianFeatureAlphaDropout"
]


class BayesianDropout(_DropoutNd):
    """
    This class is an implementation of 'torch.nn.Dropout' that remains active in eval mode.

    This is used for Bayesian neural networks (BNN) via Monte Carlo dropout (MCD), as described in "Dropout as a
    Bayesian Approximation: Representing Model Uncertainty in Deep Learning" by Y. Gal and Z. Ghahramani.

    Parameters
    ----------
    p : float
        The probability of an element to be zeroed. Defaults to 0.5.
    inplace : bool
        If set to 'True', will do this operation in-place. Defaults to 'False'.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input : Tensor
            The input Tensor.

        Returns
        -------
        output : Tensor
            The output tensor
        """
        return F.dropout(input=input, p=self.p, training=True, inplace=self.inplace)


class BayesianDropout1d(_DropoutNd):
    """
    This class is an implementation of 'torch.nn.Dropout1d' that remains active in eval mode.

    This is used for Bayesian neural networks (BNN) via Monte Carlo dropout (MCD).

    Parameters
    ----------
    p : float
        The probability of an element to be zeroed. Defaults to 0.5.
    inplace : bool
        If set to 'True', will do this operation in-place. Defaults to 'False'.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input : Tensor
            The input Tensor.

        Returns
        -------
        output : Tensor
            The output tensor
        """
        return F.dropout1d(input=input, p=self.p, training=True, inplace=self.inplace)


class BayesianDropout2d(_DropoutNd):
    """
    This class is an implementation of 'torch.nn.Dropout2d' that remains active in eval mode.

    This is used for Bayesian neural networks (BNN) via Monte Carlo dropout (MCD).

    Parameters
    ----------
    p : float
        The probability of an element to be zeroed. Defaults to 0.5.
    inplace : bool
        If set to 'True', will do this operation in-place. Defaults to 'False'.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input : Tensor
            The input Tensor.

        Returns
        -------
        output : Tensor
            The output tensor
        """
        return F.dropout2d(input=input, p=self.p, training=True, inplace=self.inplace)


class BayesianDropout3d(_DropoutNd):
    """
    This class is an implementation of 'torch.nn.Dropout3d' that remains active in eval mode.

    This is used for Bayesian neural networks (BNN) via Monte Carlo dropout (MCD).

    Parameters
    ----------
    p : float
        The probability of an element to be zeroed. Defaults to 0.5.
    inplace : bool
        If set to 'True', will do this operation in-place. Defaults to 'False'.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input : Tensor
            The input Tensor.

        Returns
        -------
        output : Tensor
            The output tensor
        """
        return F.dropout3d(input=input, p=self.p, training=True, inplace=self.inplace)


class BayesianAlphaDropout(_DropoutNd):
    """
    This class is an implementation of 'torch.nn.AlphaDropout' that remains active in eval mode.

    This is used for Bayesian neural networks (BNN) via Monte Carlo dropout (MCD).

    Parameters
    ----------
    p : float
        The probability of an element to be zeroed. Defaults to 0.5.
    inplace : bool
        If set to 'True', will do this operation in-place. Defaults to 'False'.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input : Tensor
            The input Tensor.

        Returns
        -------
        output : Tensor
            The output tensor
        """
        return F.alpha_dropout(input=input, p=self.p, training=True)


class BayesianFeatureAlphaDropout(_DropoutNd):
    """
    This class is an implementation of 'torch.nn.FeatureAlphaDropout' that remains active in eval mode.

    This is used for Bayesian neural networks (BNN) via Monte Carlo dropout (MCD).

    Parameters
    ----------
    p : float
        The probability of an element to be zeroed. Defaults to 0.5.
    inplace : bool
        If set to 'True', will do this operation in-place. Defaults to 'False'.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Forward pass.

        Parameters
        ----------
        input : Tensor
            The input Tensor.

        Returns
        -------
        output : Tensor
            The output tensor
        """
        return F.feature_alpha_dropout(input=input, p=self.p, training=True)
