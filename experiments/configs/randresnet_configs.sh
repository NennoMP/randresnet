# Best configurations for randResNet

#==============================
# MNIST
#==============================
# 5K params
python test_randresnet.py --dataset mnist --n_filters 125 --n_layers 3 --scaling 1.0 --alpha 0.5 --beta 0.5 --reg 0.0 --batch_size 256 --n_trials 10
# 19K params
python test_randresnet.py --dataset mnist --n_filters 478 --n_layers 3 --scaling 1.0 --alpha 0.5 --beta 0.5 --reg 0.0 --batch_size 256 --n_trials 10
# 75K params
python test_randresnet.py --dataset mnist --n_filters 1875 --n_layers 3 --scaling 1.0 --alpha 0.5 --beta 0.5 --reg 0.0 --batch_size 256 --n_trials 10

#==============================
# Fashion-MNIST
#==============================
# 5K params
python test_randresnet.py --dataset fmnist --n_filters 125 --n_layers 2 --scaling 0.5 --alpha 0.9 --beta 1 --reg 0.1 --batch_size 256 --n_trials 10
# 19K params
python test_randresnet.py --dataset fmnist --n_filters 478 --n_layers 2 --scaling 0.5 --alpha 0.9 --beta 1 --reg 0.1 --batch_size 256 --n_trials 10
# 75K params
python test_randresnet.py --dataset fmnist --n_filters 1875 --n_layers 2 --scaling 0.5 --alpha 0.9 --beta 1 --reg 0.1 --batch_size 256 --n_trials 10

#==============================
# SVHN
#==============================
# 5K params
python test_randresnet.py --dataset svhn --n_filters 125 --n_layers 5 --scaling 0.01 --alpha 0.01 --beta 0.01 --reg 0.0 --batch_size 256 --n_trials 10
# 19K params
python test_randresnet.py --dataset svhn --n_filters 478 --n_layers 5 --scaling 0.01 --alpha 0.01 --beta 0.01 --reg 0.0 --batch_size 256 --n_trials 10
# 75K params
python test_randresnet.py --dataset svhn --n_filters 1875 --n_layers 5 --scaling 0.01 --alpha 0.01 --beta 0.01 --reg 0.0 --batch_size 256 --n_trials 10

#==============================
# CIFAR10
#==============================
# 5K params
python test_randresnet.py --dataset cifar10 --n_filters 125 --n_layers 2 --scaling 1.0 --alpha 0.9 --beta 0.5 --reg 1.0 --batch_size 256 --n_trials 10
# 19K params
python test_randresnet.py --dataset cifar10 --n_filters 478 --n_layers 2 --scaling 1.0 --alpha 0.9 --beta 0.5 --reg 1.0 --batch_size 256 --n_trials 10
# 75K params
python test_randresnet.py --dataset cifar10 --n_filters 1875 --n_layers 2 --scaling 1.0 --alpha 0.9 --beta 0.5 --reg 1.0 --batch_size 256 --n_trials 10
