# Best configurations for fully-trainable ResNet

#==============================
# MNIST
#==============================
# 5K params
python test_resnet.py --dataset mnist --num_blocks '[1]' --optimizer sgd --lr 0.1 --epochs 50 --patience 10 --batch_size 256 --n_trials 10
# 19K params
python test_resnet.py --dataset mnist --num_blocks '[1, 1]' --optimizer sgd --lr 0.1 --epochs 50 --patience 10 --batch_size 256 --n_trials 10
# 75K params
python test_resnet.py --dataset mnist --num_blocks '[1, 1, 1]' --optimizer sgd --lr 0.1 --epochs 50 --patience 10 --batch_size 256 --n_trials 10

#==============================
# Fashion-MNIST
#==============================
# 5K params
python test_resnet.py --dataset fmnist --num_blocks '[1]' --optimizer adam --lr 0.01 --epochs 50 --patience 10 --batch_size 256 --n_trials 10
# 19K params
python test_resnet.py --dataset fmnist --num_blocks '[1, 1]' --optimizer sgd --lr 0.1 --epochs 50 --patience 10 --batch_size 256 --n_trials 10
# 75K params
python test_resnet.py --dataset fmnist --num_blocks '[1, 1, 1]' --optimizer adam --lr 0.01 --epochs 50 --patience 10 --batch_size 256 --n_trials 10

#==============================
# SVHN
#==============================
# 5K params
python test_resnet.py --dataset svhn --num_blocks '[1]' --optimizer adam --lr 0.01 --epochs 200 --patience 20 --batch_size 256 --n_trials 10
# 19K params
python test_resnet.py --dataset svhn --num_blocks '[1, 1]' --optimizer adam --lr 0.01 --epochs 200 --patience 20 --batch_size 256 --n_trials 10
# 75K params
python test_resnet.py --dataset svhn --num_blocks '[1, 1, 1]' --optimizer adam --lr 0.01 --epochs 200 --patience 20 --batch_size 256 --n_trials 10

#==============================
# CIFAR10
#==============================
# 5K params
python test_resnet.py --dataset cifar10 --num_blocks '[1]' --optimizer sgd --lr 0.1 --epochs 200 --patience 20 --batch_size 256 --n_trials 10
# 19K params
python test_resnet.py --dataset cifar10 --num_blocks '[1, 1]' --optimizer sgd --lr 0.1 --epochs 200 --patience 20 --batch_size 256 --n_trials 10
# 75K params
python test_resnet.py --dataset cifar10 --num_blocks '[1, 1, 1]' --optimizer adam --lr 0.01 --epochs 200 --patience 20 --batch_size 256 --n_trials 10
