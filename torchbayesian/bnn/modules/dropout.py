"""
    @file:              dropout.py
    @Author:            Raphael Brodeur

    @Creation Date:     01/2026
    @Last modification: 01/2026

    @Description:       This file contains Bayesian implementations of Torch dropout layers.
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
    This class is a Bayesian (Monte Carlo Dropout) implementation of 'torch.nn.Dropout'. Remains active in eval mode.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.

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
    This class is a Bayesian (Monte Carlo Dropout) implementation of 'torch.nn.Dropout1d'. Remains active in eval mode.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.

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
    This class is a Bayesian (Monte Carlo Dropout) implementation of 'torch.nn.Dropout2d'. Remains active in eval mode.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.

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
    This class is a Bayesian (Monte Carlo Dropout) implementation of 'torch.nn.Dropout3d'. Remains active in eval mode.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.

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
    This class is a Bayesian (Monte Carlo Dropout) implementation of 'torch.nn.AlphaDropout'. Remains active in eval
    mode.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.

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
    This class is a Bayesian (Monte Carlo Dropout) implementation of 'torch.nn.FeatureAlphaDropout'. Remains active in
    eval mode.
    """

    def forward(self, input: Tensor) -> Tensor:
        """
        Runs the forward pass.

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
