__all__ = ["LambdaLayer"]

import torch
import torch.nn as nn


class LambdaLayer(nn.Module):
    """A layer that applies a given lambda function to its input tensor.

    Parameters
    ----------
    lambd : torch.nn.Module
        A lambda function to be applied to the input tensor.
    """

    def __init__(self, lambd: torch.nn.Module) -> None:
        super().__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)
