"""Implementation of randomized Residual Neural Networks (randResNet) in PyTorch.

The architecture is inspired by the seminal paper [1]. The structure of this file is heavily 
influenced by [2, 3]. In particular, the implementation of LambdaLayer and ResidualBlock is 
adapted from [3].

References:
[1] He et al., Deep Residual Learning for Image Recognition (2016), arXiv:1512.03385
[2] https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
[3] https://github.com/akamaster/pytorch_resnet_cifar10/blob/master/resnet.py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.networks.init_utils import init_weights


class LambdaLayer(nn.Module):
    def __init__(self, lambd: torch.nn.Module) -> None:
        """
        Args:
            lambd: lambda function to apply to the input tensor.
        """
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.lambd(x)


class ResidualBlock(nn.Module):
    def __init__(
        self,
        n_filters: int,
        alpha: float,
        beta: float,
        stride: int = 1,
        skip_option: str = 'identity',
    ) -> None:
        """
        Args:
            n_filters: number of filters in the convolutional layer.
            alpha: scaling factor for residual branch.
            beta: scaling factor for non-linear branch.
            stride: stride of the average pooling operator.
            skip_option: type of skip connection to use. Options are 'identity' and 'conv'.

        Raises:
            ValueError: if an invalid skip_option is provided.
        """
        super().__init__()
        
        self.conv = nn.Conv2d(n_filters, n_filters, kernel_size=3, stride=1, padding=1, bias=False)   
        self.bn = nn.BatchNorm2d(n_filters, affine=False)
        self.avgpool = nn.AvgPool2d(kernel_size=3, stride=stride, padding=1)

        self.alpha = nn.Parameter(torch.tensor([alpha]), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor([beta]), requires_grad=False)
        
        if skip_option == 'identity': # identity residual connection
            self.shortcut = LambdaLayer(
                lambda x: F.pad(x[:, :, ::stride, ::stride], (0, 0, 0, 0, 0, 0), 'constant', 0)
            )
        elif skip_option == 'conv': # random conv residual connection
            self.shortcut = nn.Sequential(
                nn.Conv2d(n_filters, n_filters, kernel_size=1, stride=stride, bias=False)
            )
        else:
            raise ValueError(
                f"Invalid skip option: {skip_option}. Options are 'identity' and 'conv'!"
            )
            
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.avgpool(torch.tanh(self.bn(self.conv(x))))
        out = (self.alpha * self.shortcut(x)) + (self.beta * out)
        return out 


class randResNet(nn.Module):
    def __init__(
        self, 
        in_channels: int = 1,
        n_layers: int = 1,
        n_filters: int = 16,
        init: str = 'uniform',
        scaling: float = 0.1,
        alpha: float = 1,
        beta: float = 1,
        adaptive_size: int = 2,
        skip_option: str = 'identity',
    ) -> None:
        """
        Args:
            in_channels: number of input channels in the first convolutional layer.
            n_layers: number of residual blocks.
            n_filters: number of filters in each convolutional layer.
            init: initialization strategy. Options are 'uniform', 'xavier' and 'orthogonal'.
            scaling: scaling factor for initialization.
            alpha: scaling factor for residual branch.
            beta: scaling factor for non-linear branch.
            adaptive_size: target output size of the adaptive average pooling layer.
            skip_option: type of skip connection to use. Options are 'identity' and 'conv'.
        """
        super().__init__()
        
        self.n_layers = n_layers
        self.n_filters = n_filters
        self.init = init
        self.scaling = scaling
        self.alpha = alpha
        self.beta = beta
        self.skip_option = skip_option
        
        # Input Block
        self.in_conv = nn.Conv2d(in_channels, n_filters, kernel_size=3, stride=1, padding=1, bias=False)
        self.in_bn = nn.BatchNorm2d(n_filters)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        # Residual Blocks
        self.layers = self._make_layers() 
        self.adaptive_avgpool = nn.AdaptiveAvgPool2d((adaptive_size, adaptive_size))
        
        self._init_weights()
            
    def _init_weights(self):
        for _, m in self.named_modules():
            init_weights(m, self.init, self.scaling)

    def _make_layers(self) -> nn.Sequential:
        layers = []
        for _ in range(1, self.n_layers):
            layers.append(
            ResidualBlock(
                n_filters=self.n_filters,
                skip_option=self.skip_option,
                alpha=self.alpha,
                beta=self.beta,
            )
        )
        return nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.maxpool(torch.tanh(self.in_bn(self.in_conv(x))))    
        for layer in self.layers:
            out = layer(out)
        out = self.adaptive_avgpool(out)
        out = out.view(out.size(0), -1)
        return out


def randresnet(hparams: dict) -> randResNet:
    return randResNet(**hparams)

def randresnet5k(hparams: dict) -> torch.nn.Module:
    """randResNet - 5K trainable parameters."""
    hparams.pop('n_filters', None)
    return randResNet(n_filters=[125], **hparams)

def randresnet19k(hparams: dict) -> torch.nn.Module:
    """randResNet - 19K trainable parameters."""
    hparams.pop('n_filters', None)
    return randResNet(n_filters=[478], **hparams)

def randresnet75k(hparams: dict) -> torch.nn.Module:
    """randResNet - 75K trainable parameters."""
    hparams.pop('n_filters', None)
    return randResNet(n_filters=[1875], **hparams)
