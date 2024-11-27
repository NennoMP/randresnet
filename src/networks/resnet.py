"""Proper implementation of ResNet for CIFAR10 as described in paper [1]. Code heavily adapted from [2].

In our paper experiments we considered the following network sizes:
Name       | #Layers   | #Params
ResNet4    | 4         | 5k
ResNet6    | 6         | 19.1k
ResNet8    | 8         | 75k

References:
[1] He et al., Deep Residual Learning for Image Recognition (2016), arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/main/torchvision/models/resnet.py
"""
from typing import List, Type

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


def _weights_init(m: nn.Module) -> None:
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight)


class LambdaLayer(nn.Module):
    def __init__(self, lambd: torch.nn.Module) -> None:
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)


class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1, option: str = 'A'):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, 
            planes, 
            kernel_size=3, 
            stride=stride, 
            padding=1, 
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """For CIFAR10 ResNet paper uses option A."""
                self.shortcut = LambdaLayer(
                    lambda x: F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0)
                )
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(planes)
                )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self, 
        in_channels: int = 3, 
        in_planes: int = 16,
        block: Type[BasicBlock] = BasicBlock, 
        num_blocks: List[int] = [1, 1, 1], 
        num_classes: int = 10
    ):
        super(ResNet, self).__init__()
        self.in_planes = in_planes

        self.conv1 = nn.Conv2d(in_channels, in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes)
        
        # First basic block
        self.layers = [
            self._make_layer(
                block, 
                in_planes,
                num_blocks[0], 
                stride=1
            )
        ]
        # Other basic blocks
        for i in range(1, len(num_blocks)):
            in_planes *= 2
            self.layers.append(
                self._make_layer(
                    block, 
                    in_planes,
                    num_blocks[i], 
                    stride=2
                )
            )
        self.layers = nn.Sequential(*self.layers)
        self.linear = nn.Linear(in_planes, num_classes)

        self.apply(_weights_init)

    def _make_layer(self, block: BasicBlock, planes: int, num_blocks: int, stride: int):
        strides = [stride] + [1]*(num_blocks-1)
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


def resnet(hparams: dict) -> ResNet:
    return ResNet(**hparams)

def resnet4(hparams: dict) -> torch.nn.Module:
    """ResNet4 - 5K trainable parameters."""
    hparams.pop('num_blocks', None)
    return ResNet(num_blocks=[1], **hparams)

def resnet6(hparams: dict) -> torch.nn.Module:
    """ResNet6 - 19K trainable parameters."""
    hparams.pop('num_blocks', None)
    return ResNet(num_blocks=[1, 1], **hparams)

def resnet8(hparams: dict) -> torch.nn.Module:
    """ResNet8 - 75K trainable parameters."""
    hparams.pop('num_blocks', None)
    return ResNet(num_blocks=[1, 1, 1], **hparams)