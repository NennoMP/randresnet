"""Properly implemented ResNet-s for CIFAR10 as described in paper [1]_.

This implementation has been adapted from [2]_, by Yerlan Idelbayev.

In our paper experiments we considered the following network sizes:

name       | layers   | params
ResNet4    | 4        | 5k
ResNet6    | 6        | 19.1k
ResNet8    | 8        | 75k

References
----------
.. [1] He et al., Deep Residual Learning for Image Recognition (2016), arXiv:1512.03385
.. [2] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""

__all__ = ["ResNet", "resnet_", "resnet4", "resnet6", "resnet8"]


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from src.models.layers import LambdaLayer


class BasicBlock(nn.Module):
    """Basic block for traditional, fully-trainable ResNets.

    Parameters
    ----------
    in_planes : int
        Number of input channels.
    planes : int
        Number of output channels.
    stride : int, optional, default=1
        Stride for the convolutional layer.
    skip_option : str, optional, default='A'
        Option for the shortcut connection. Options are 'A' for identity, 'B' for 1x1
        convolution.

    Attributes
    ----------
    conv1 : nn.Conv2d
        First convolutional layer.
    bn1 : nn.BatchNorm2d
        Batch normalization for the first convolutional layer.
    conv2 : nn.Conv2d
        Second convolutional layer.
    bn2 : nn.BatchNorm2d
        Batch normalization for the second convolutional layer.
    shortcut : nn.Sequential
        Shortcut connection.
    """

    def __init__(
        self, in_planes: int, planes: int, stride: int = 1, skip_option: str = "A"
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if skip_option == "A":
                """For CIFAR10 ResNet paper uses skip option 'A'."""
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(
                        x[:, :, ::2, ::2],
                        (0, 0, 0, 0, planes // 4, planes // 4),
                        "constant",
                        0,
                    )
                )
            elif skip_option == "B":
                self.shortcut = nn.Sequential(
                    nn.Conv2d(
                        in_planes, planes, kernel_size=1, stride=stride, bias=False
                    ),
                    nn.BatchNorm2d(planes),
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    """ResNet model for CIFAR10 using BasicBlock.

    Parameters
    ----------
    in_channels : int, optional, default=3
        Number of input channels.
    in_planes : int, optional, default=16
        Number of input planes for the first convolutional layer.
    block : Type[BasicBlock], optional, default=BasicBlock
        Block type to be used in the ResNet.
    num_blocks : list[int], optional, default=[1, 1, 1]
        Number of blocks in each layer.
    num_classes : int, optional, default=10
        Number of output classes.

    Attributes
    ----------
    conv1 : nn.Conv2d
        First convolutional layer.
    bn1 : nn.BatchNorm2d
        Batch normalization for the first convolutional layer.
    layers : nn.Sequential
        Sequential container of ResNet layers.
    linear : nn.Linear
        Fully connected output layer.
    """

    def __init__(
        self,
        in_channels: int = 3,
        in_planes: int = 16,
        block: type[BasicBlock] = BasicBlock,
        num_blocks: list[int] = [1, 1, 1],
        num_classes: int = 10,
    ) -> None:
        super().__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(
            in_channels, in_planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(in_planes)

        # First basic block
        self.layers = [self._make_layer(block, in_planes, num_blocks[0], stride=1)]

        # Subsequent basic blocks
        for i in range(1, len(num_blocks)):
            in_planes *= 2
            self.layers.append(
                self._make_layer(block, in_planes, num_blocks[i], stride=2)
            )
        self.layers = nn.Sequential(*self.layers)

        # Fully-connected output layer
        self.linear = nn.Linear(in_planes, num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights of the model with Kaiming normal initialization."""
        for _, m in self.named_modules():
            if isinstance(m, nn.Linear | nn.Conv2d):
                init.kaiming_normal_(m.weight)

    def _make_layer(
        self, block: BasicBlock, planes: int, num_blocks: int, stride: int
    ) -> nn.Sequential:
        """Initialize a ResNet layer.

        Parameters
        ----------
        block : BasicBlock
            Block type to be used in the ResNet.
        planes : int
            Number of output channels.
        num_blocks : int
            Number of blocks in the layer.
        stride : int
            Stride for the first block in the layer.

        Returns
        -------
        nn.Sequential
            Sequential container of the ResNet layer.
        """
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes

        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        for layer in self.layers:
            out = layer(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def resnet_(hparams: dict) -> ResNet:
    """Initialize ResNet with given hyperparameters."""
    return ResNet(**hparams)


def resnet4(hparams: dict) -> torch.nn.Module:
    """Initialize ResNet4 (5K trainable parameters) with given hyperparameters."""
    hparams.pop("num_blocks", None)
    return ResNet(num_blocks=[1], **hparams)


def resnet6(hparams: dict) -> torch.nn.Module:
    """Initialzie ResNet6 (19K trainable parameters) with given hyperparameters."""
    hparams.pop("num_blocks", None)
    return ResNet(num_blocks=[1, 1], **hparams)


def resnet8(hparams: dict) -> torch.nn.Module:
    """Initialize ResNet8 (75K trainable parameters) with given hyperparameters."""
    hparams.pop("num_blocks", None)
    return ResNet(num_blocks=[1, 1, 1], **hparams)
