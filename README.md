<div align="center">

# Randomized Residual Neural Networks

![code-quality](https://github.com/nennomp/randresnet/actions/workflows/code-quality.yml/badge.svg)
[![license](https://img.shields.io/badge/License-MIT-green.svg?labelColor=gray)](https://github.com/nennomp/randresnet)

</div>

This repository contains the official code for the paper:

```
Randomized Residual Neural Networks,
Matteo Pinna, Andrea Ceni, Claudio Gallicchio.
```

## Abstract
Convolutional Neural Networks (CNNs) can extract relevant features from raw images even with randomly initialized and fixed weights, particularly when combined with appropriate pooling operators that provide robustness to noise and translation. In this paper, we study the architectural bias of randomly weighted Residual Neural Networks (ResNets), a particular type of CNNs with residual connections. Specifically, We introduce a novel class of randomized ResNets, called randomized Residual Neural Networks (randResNets), which use positive scaling coefficients for the residual and non-linear branches to control the trade-off between linear and non-linear dynamics and the proximity to the edge of stability. The convolutional layers are randomly initialized and then left untrained, and are used as a pre-processing step to transform raw images into rich feature maps, making image recognition tasks more easily solvable by a linear classifier. Output features are obtained through a single forward pass, bypassing backpropagation, and the linear classifier can be trained using lightweight techniques with closed-form solutions. Thus, the proposed approach achieves notable computational efficiency, as no backpropagation is involved in the training process. Furthermore, randResNet employs independent scaling coefficients for both the residual and non-linear branches, which are tunable hyperparameters of the model. A thorough mathematical analysis demonstrates the stability of our randomized ResNet architecture for an arbitrarily large number of depths. We demonstrate that the proposed approach can compete with fully-trainable ResNets and outperforms other randomized neural networks, while being more efficient in terms of computations, often by orders of magnitude.

<div align="center">
<img src="assets/figure-1.png?raw=true" alt="Model" title="Model">
</div>

## Setup
To install the required dependencies:
```
conda create -n randresnet python=3.12
conda activate randresnet
pip install -e .
```

## Experiments
See the [experiments guide](./experiments/README.md) for instructions on reproducing the experiments in the paper, or running your own.
