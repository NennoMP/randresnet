# Randomized Convolutional Residual Neural Networks
Code for "Randomized Convolutional Residual Neural Networks" paper.

### Abstract
Convolutional Neural Networks (CNNs) can extract relevant features from raw images even with randomly initialized and fixed weights, particularly when combined with appropriate pooling operators that provide robustness to noise and translation. In this paper, we study the architectural bias of randomly weighted Residual Neural Networks (ResNets), a particular type of CNNs with residual connections. Specifically, we introduce a novel class of randomized ResNets and employ its untrained convolutional operators as a pre-processing to transform raw images into rich feature maps, making image recognition tasks more easily solvable by a linear classifier. A thorough mathematical analysis demonstrates the stability of our randomized ResNet architecture for an arbitrarily large number of depths. We demonstrate that the proposed approach can often compete with fully-trainable ResNets, while being more efficient in terms of computations, often by orders of magnitude.

### Setup
Linux
```
virtualenv venv && source venv/bin/activate && pip install -r requirements.txt
pip install -e .
```
Windows
```
virtualenv venv && venv\scripts\activate && pip install -r requirements.txt
pip install -e .
```

### Run
See the [experiments guide](./experiments/README.md) for instructions on reproducing the experiments in the paper, or running your own.

### Citation
If you use the code in this repository, consider citing our work:
```
PLACEHOLDER
```
