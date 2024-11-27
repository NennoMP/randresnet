"""Module containing utils for data loading and pre-processing."""
from typing import Tuple

import numpy as np
import torch
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision.datasets.vision import VisionDataset


def get_mnist(
    out_dir: str = 'datasets', 
    val_split: float = 0.1,
    batch_size: int = 256, 
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Get MNIST dataloaders.

    Args:
        - val_split: validation split ratio
        - random_state: random seed for reproducibility
    """
    dataset = torchvision.datasets.MNIST

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.1307,), std=(0.3081,)),
    ])
    full_train_dataset = dataset(out_dir, train=True, transform=transforms, download=True)
    test_dataset = dataset(out_dir, train=False, transform=transforms, download=True)
    targets = full_train_dataset.targets

    return get_dataloaders(
        full_train_dataset, 
        test_dataset, 
        targets, 
        val_split, 
        batch_size, 
        random_state
    )

def get_fmnist(
    out_dir: str = 'datasets', 
    val_split: float = 0.1,
    batch_size: int = 256, 
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Get Fashion-MNIST dataloaders.

    Args:
        - val_split: validation split ratio
        - random_state: random seed for reproducibility
    """
    dataset = torchvision.datasets.FashionMNIST

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.2860,), std=(0.3530,)),
    ])
    full_train_dataset = dataset(out_dir, train=True, transform=transforms, download=True)
    test_dataset = dataset(out_dir, train=False, transform=transforms, download=True)
    targets = full_train_dataset.targets

    return get_dataloaders(
        full_train_dataset, 
        test_dataset, 
        targets, 
        val_split, 
        batch_size, 
        random_state
    )

def get_svhn(
    out_dir: str = 'datasets', 
    val_split: float = 0.1,
    batch_size: int = 256, 
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Get Street View House Numbers (SVHN) dataloaders.

    Args:
        - val_split: validation split ratio
        - random_state: random seed for reproducibility
    """
    dataset = torchvision.datasets.SVHN

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.4377, 0.4438, 0.4728), std=(0.1980, 0.2010, 0.1970)),
    ])
    full_train_dataset = dataset(out_dir, split='train', transform=transforms, download=True)
    test_dataset = dataset(out_dir, split='test', transform=transforms, download=True)
    targets = full_train_dataset.labels

    return get_dataloaders(
        full_train_dataset, 
        test_dataset, 
        targets, 
        val_split, 
        batch_size, 
        random_state
    )

def get_cifar10(
    out_dir: str = 'datasets', 
    val_split: float = 0.1,
    batch_size: int = 256, 
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """Get CIFAR10 dataloaders.

    Args:
        - val_split: validation split ratio
        - random_state: random seed for reproducibility
    """
    dataset = torchvision.datasets.CIFAR10

    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2470, 0.2435, 0.2616)),
    ])

    full_train_dataset = dataset(out_dir, train=True, transform=transforms, download=True)
    test_dataset = dataset(out_dir, train=False, transform=transforms, download=True)
    targets = full_train_dataset.targets

    return get_dataloaders(
        full_train_dataset, 
        test_dataset, 
        targets, 
        val_split, 
        batch_size, 
        random_state
    )

def get_dataloaders(
    full_train_dataset: Dataset, 
    test_dataset: Dataset,
    targets: np.ndarray,
    val_split: float = 0.1,
    batch_size: int = 256, 
    num_workers: int = 1,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
    """
    Perform a stratified train-val split of a given dataset. Then, return the corresponding 
    dataloaders.

    Args:
        - full_train_dataset: the training set to split
        - test_dataset: the test set
        - targets: the target labels of the given training dataset. Used for stratified split.
        - val_split: the ratio of validation split
        - random_state: the random seed for reproducibility

    Returns:
        - full_train_dataloader: original training set dataloader
        - train_dataloader: training set split dataloader
        - val_dataloader: validation set split dataloader
        - test_dataloader: test set dataloader
    """

    # Stratified train-val split
    train_indices, val_indices = train_test_split(
        np.arange(len(full_train_dataset)),
        test_size=val_split,
        random_state=random_state,
        stratify=targets
    )
    train_dataset = Subset(full_train_dataset, train_indices)
    val_dataset = Subset(full_train_dataset, val_indices)

    full_train_dataloader = DataLoader(
        full_train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=num_workers,
    )
    val_dataloader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
    )
    test_dataloader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=num_workers,
    )

    return full_train_dataloader, train_dataloader, val_dataloader, test_dataloader

def compute_mean_std(
    dataset: VisionDataset
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    """Compute channel-wise mean and standard deviation of a given dataset.

    Args:
        dataset: a (torchvision) dataset to compute mean and std
    
    Returns:
        mean: a tuple of mean values for each channel
        std: a tuple of standard deviation values for each channel
    """
    data = torch.stack([dataset[i][0] for i in range(len(dataset))], dim=0).cpu().numpy()

    # Compute mean and std for each channel
    if data.shape[1] == 1: # gray-scale
        mean = (data[:,0,:,:].mean(),)
        std = (data[:,0,:,:].std(),)
    else: # RGB
        mean = data[:,0,:,:].mean(), data[:,1,:,:].mean(), data[:,2,:,:].mean()
        std = data[:,0,:,:].std(), data[:,1,:,:].std(), data[:,2,:,:].std()

    return mean, std