"""PyTorch implementation of randomized Residual Neural Networks (randResNet).

The architecture is inspired by the seminal paper on ResNets [1]_. The implementation
and structure of this file is heavily influenced by [1]_ and [2]_.

References
----------
.. [1] He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings
of the IEEE conference on computer vision and pattern recognition. 2016.
.. [2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
.. [3] https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
"""

__all__ = [
    "randResNet",
    "randresnet_",
    "randresnet5k",
    "randresnet19k",
    "randresnet75k",
]
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.layers import LambdaLayer


class ResidualBlock(nn.Module):
    """Residual block of randResNet.

    Consists of (Conv -> BatchNorm -> AvgPool) with residual skip connections. The skip
    connections and non-linear branch are scaled by tunable hyperparameters `alpha` and
    `beta`, respectively.

    Parameters
    ----------
    n_filters : int
        Number of filters in the convolutional layer.
    alpha : float
        Scaling factor for the residual branch.
    beta : float
        Scaling factor for the non-linear branch.
    stride : int
        Stride of the average pooling operator. If skip connections are implemented
        as random convolution (`skip_option`='conv'), this also defines the stride of
        the convolution in the skip connections.
    skip_option : str
        Type of skip connection to use. Options are 'identity' and 'conv'.

    Attributes
    ----------
    block : nn.Sequential
        A sequential container of (Conv -> BatchNorm -> Tanh -> AvgPool).
    alpha : nn.Parameter
        Scaling factor for the residual branch.
    beta : nn.Parameter
        Scaling factor for the non-linear branch.
    shortcut : nn.Module
        Shortcut transformation, either identity or random convolution.
    """

    def __init__(
        self,
        n_filters: int,
        alpha: float,
        beta: float,
        stride: int,
        skip_option: str,
    ) -> None:
        """
        Raises
        ------
        ValueError
            If an invalid `skip_option` is provided.
        """
        super().__init__()

        self.block = self._make_block(n_filters=n_filters, stride=stride)

        # scaling coefficients
        self.register_buffer("alpha", torch.tensor([alpha]))
        self.register_buffer("beta", torch.tensor([beta]))

        # residual skip connections
        if skip_option == "identity":  # identity
            self.shortcut = LambdaLayer(
                lambd=lambda x: F.pad(
                    input=x[:, :, ::stride, ::stride],
                    pad=(0, 0, 0, 0, 0, 0),
                    mode="constant",
                    value=0,
                )
            )
        elif skip_option == "conv":  # random convolution
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    n_filters, n_filters, kernel_size=1, stride=stride, bias=False
                )
            )
        else:
            raise ValueError(
                f"Invalid skip option: {skip_option}. "
                f"Options are 'identity' and 'conv'!"
            )

    def _make_block(self, n_filters: int, stride: int) -> nn.Sequential:
        """Initialize the residual block.

        Parameters
        ----------
        n_filters : int
            Number of filters in the convolutional layer.
        stride : int
            Stride of the average pooling operator.

        Returns
        -------
        nn.Sequential
            A sequential container of (Conv -> BatchNorm -> Tanh -> AvgPool).
        """
        layers = OrderedDict()
        layers.update(
            [
                (
                    "conv",
                    nn.Conv2d(
                        n_filters, n_filters, kernel_size=3, padding=1, bias=False
                    ),
                ),
                ("bn", nn.BatchNorm2d(num_features=n_filters, affine=False)),
                ("activation", nn.Tanh()),
                ("avgpool", nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)),
            ]
        )
        return nn.Sequential(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, n_filters, height (H), width (W)).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_filters, height (H'), width (W')).
        """
        out = self.block(x)
        out = (self.alpha * self.shortcut(x)) + (self.beta * out)
        return out


class randResNet(nn.Module):
    """Randomized Residual Neural Networks (randResNet).

    randResNet is a randomization-based (convolutional) Residual Neural Network
    (ResNet). All parameters are randomly initialized and left untrained, and leveraged
    as a light-weight feature extractor. Residual connections are scaled by positive,
    tunable hyperparameters `alpha` and `beta`, to control the trade-off between
    linear and non-linear dynamics, respectively.

    The architecture consists of one input block, followed by an arbitrary number of
    residual blocks with skip connections, and an adaptive average pooling layer. All
    blocks consist of a convolution, batch normalization, and pooling. The input block
    employs max pooling, while residual blocks employ average pooling.

    The extracted features are typically fed to a simple, linear classifier trained via
    closed-form solutions. This is the only trainable component in the overall model.
    The number of trainable parameters can be estimated as `n_filters` *
    (`adaptive_size` ** 2) * num_classes.

    Parameters
    ----------
    n_layers : int
        Number of residual blocks in the network.
    in_channels : int
        Number of input channels.
    n_filters : int, default=128
        Number of filters in each convolutional layer.
    scaling : float, default=0.1
        Scaling factor for weight initialization.
    alpha : float, default=1
        Scaling factor for the residual branch.
    beta : float, default=1
        Scaling factor for the non-linear branch.
    adaptive_size : int, default=2
        Output size of the final adaptive average pooling layer.
    skip_option : str, default='identity'
        Type of skip connection to use. Options are 'identity' and 'conv'.

    Attributes
    ----------
    in_block : nn.Sequential
        Input block, consisting of (Conv -> BatchNorm -> Tanh -> MaxPool).
    layers : nn.Sequential
        A sequential container of `ResidualBlock` instances, each consisting of (Conv
        -> BatchNorm -> Tanh -> AvgPool).
    adaptive_avgpool : nn.AdaptiveAvgPool2d
        Final adaptive average pooling layer.
    """

    def __init__(
        self,
        n_layers: int,
        in_channels: int,
        n_filters: int = 128,
        scaling: float = 0.1,
        alpha: float = 1,
        beta: float = 1,
        adaptive_size: int = 2,
        skip_option: str = "identity",
    ) -> None:
        super().__init__()

        # Input block (Conv -> BatchNorm -> Tanh -> MaxPool)
        self.in_block = nn.Sequential(
            OrderedDict(
                [
                    (
                        "con",
                        nn.Conv2d(
                            in_channels, n_filters, kernel_size=3, padding=1, bias=False
                        ),
                    ),
                    ("bn", nn.BatchNorm2d(num_features=n_filters)),
                    ("activation", nn.Tanh()),
                    ("maxpool", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        # Residual blocks (Conv -> BatchNorm -> Tanh -> AvgPool)
        self.layers = self._make_layers(
            n_layers=n_layers,
            n_filters=n_filters,
            alpha=alpha,
            beta=beta,
            skip_option=skip_option,
        )

        # Adaptive average pooling layer
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((adaptive_size, adaptive_size))

        self._init_weights(scaling=scaling)

    def _init_weights(self, scaling: float) -> None:
        """Initialize weights of the model.

        Weights are uniformly initialized in the range [-`scaling`, `scaling`].

        Parameters
        ----------
        scaling : float
            Scaling factor for weight initialization.
        """
        for _, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.uniform_(tensor=m.weight, a=-scaling, b=scaling)

    def _make_layers(
        self,
        n_layers: int,
        n_filters: int,
        alpha: int,
        beta: int,
        skip_option: int,
        stride: int = 1,
    ) -> nn.Sequential:
        """Initialize residual blocks.

        Parameters
        ----------
        n_layers : int
            Number of residual blocks to create.
        n_filters : int
            Number of filters in each convolutional layer.
        alpha : float
            Scaling factor for the residual branch.
        beta : float
            Scaling factor for the non-linear branch.
        skip_option : str
            Type of skip connection to use. Options are 'identity' and 'conv'.
        stride : int, default=1
            Stride of the average pooling operator in each residual block.

        Returns
        -------
        nn.Sequential
            A sequential container of `ResidualBlock` instances.
        """
        layers = [
            ResidualBlock(
                n_filters=n_filters,
                alpha=alpha,
                beta=beta,
                stride=stride,
                skip_option=skip_option,
            )
            for _ in range(n_layers)
        ]

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch_size, in_channels, height, width).

        Returns
        -------
        torch.Tensor
            Output tensor of shape (batch_size, n_filters, adaptive_size,
            adaptive_size).
        """
        out = self.in_block(x)
        out = self.layers(out)
        out = self.adaptive_avgpool(out)
        out = out.view(out.size(0), -1)  # flatten
        return out


def randresnet_(hparams: dict) -> randResNet:
    """Initialize randResNet with given hyperparameters."""
    return randResNet(**hparams)


def randresnet5k(hparams: dict) -> torch.nn.Module:
    """Initialize randResNet (5K trainable parameters) with given hyperparameters."""
    hparams.pop("n_filters", None)
    hparams.pop("adaptive_size", 2)
    return randResNet(n_filters=[125], **hparams)


def randresnet19k(hparams: dict) -> torch.nn.Module:
    """Initialize randResNet (19K trainable parameters) with given hyperparameters."""
    hparams.pop("n_filters", None)
    hparams.pop("adaptive_size", 2)
    return randResNet(n_filters=[478], **hparams)


def randresnet75k(hparams: dict) -> torch.nn.Module:
    """Initialize randResNet (75K trainable parameters) with given hyperparameters."""
    hparams.pop("n_filters", None)
    hparams.pop("adaptive_size", 2)
    return randResNet(n_filters=[1875], **hparams)
