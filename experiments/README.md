# Experiments
Here you can find instructions for reproducing the experiments presented in the paper.

Additionally, in `configs/randresnet_configs.sh` and `configs/resnet_configs.sh` you can find the optimal configurations identified during model selection and used in our experiments, for randResNet and fully-trainable ResNet, respectively.

### randResNet
To test randResNet on the MNIST dataset across $10$ different random initializations, run the following:
- **$5K$ trainable parameters**
```
python test_randresnet.py --dataset mnist --n_filters 125 --n_layers 3 --scaling 1.0 --alpha 0.5 --beta 0.5 --reg 0.0 --batch_size 256 --n_trials 10
```

- **$19K$ trainable parameters**
```
python test_randresnet.py --dataset mnist --n_filters 478 --n_layers 3 --scaling 1.0 --alpha 0.5 --beta 0.5 --reg 0.0 --batch_size 256 --n_trials 10
```

- **$75K$ trainable parameters**
```
python test_randresnet.py --dataset mnist --n_filters 1875 --n_layers 3 --scaling 1.0 --alpha 0.5 --beta 0.5 --reg 0.0 --batch_size 256 --n_trials 10
```

### ResNet
To test fully-trainable ResNet on the MNIST dataset across $10$ different random initializations, run the following:
- **$5K$ trainable parameters**
```
python test_resnet.py --dataset mnist --num_blocks '[1]' --optimizer sgd --lr 0.1 --epochs 50 --patience 10 --batch_size 256 --n_trials 10
```

- **$19K$ trainable parameters**
```
python test_resnet.py --dataset mnist --num_blocks '[1, 1]' --optimizer sgd --lr 0.1 --epochs 50 --patience 10 --batch_size 256 --n_trials 10
```

- **$75K$ trainable parameters**
```
python test_resnet.py --dataset mnist --num_blocks '[1, 1, 1]' --optimizer sgd --lr 0.1 --epochs 50 --patience 10 --batch_size 256 --n_trials 10
```
