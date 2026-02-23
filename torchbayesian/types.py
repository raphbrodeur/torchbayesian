"""
    @file:              types.py
    @Author:            Raphael Brodeur

    @Creation Date:     02/2026
    @Last modification: 02/2026

    @Description:       This file contains type aliases used across torchbayesian. Some types are defined to mirror
                        PyTorch types in order to avoid depending on PyTorch internal types; these may be adopted later
                        on as PyTorch introduces stable public type aliases.
"""

from typing import TypeAlias, Union

from torch import (
    dtype as _dtype,
    Size as Size
)

# Analog to PyTorch internal type '_size'.
# TODO -- Merge later on as PyTorch makes public common internal types.
_size: TypeAlias = Union[Size, list[int], tuple[int, ...]]  # noqa: PYI042,PYI047
