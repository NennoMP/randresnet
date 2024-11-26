"""Module containing utils for weight initialization."""
import torch
import torch.nn as nn


def init_weights(m: nn.Module, init: str, scaling: float, act: str = 'tanh') -> None:
    """
    Initialize weights of a nn.Conv2d module according to the specified initialization strategy.
    
    Args:
        m: module to initialize
        init: initialization strategy. Options are 'uniform', 'xavier' and 'orthogonal'.
        scaling: scaling factor for initialization. Only used for 'uniform' initialization.
        act: activation function for computing the gain. Only used for 'xavier' and 'orthogonal' 
            initialization.

    Raises:
        ValueError: if an invalid initialization strategy is provided.
    """
    if isinstance(m, nn.Conv2d):
        if init == 'uniform':
            torch.nn.init.uniform_(m.weight, a=-scaling, b=scaling)
        elif init == 'xavier':
            torch.nn.init.xavier_uniform_(m.weight, gain=torch.nn.init.calculate_gain(act))
        elif init == 'orthogonal':
            torch.nn.init.orthogonal_(m.weight, gain=torch.nn.init.calculate_gain(act))
        else:
            raise ValueError(
                f"Invalid initialization: {init}. Options are 'uniform', 'xavier' and 'orthogonal'!"
            )