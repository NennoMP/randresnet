"""Dataloaders for vision datasets."""

__all__ = ["VisionDataLoader"]

import os

import numpy as np
import torchvision
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Subset
from torchvision.datasets.vision import VisionDataset


class VisionDataLoader:
    """Loader for vision datasets.

    Supported datasets include MNIST, FashionMNIST, SVHN, and CIFAR10.

    Parameters
    ----------
    path : str
        Path to the directory where datasets will be stored/downloaded.

    Attributes
    ----------
    DATASETS : dict[str, dict[str, Any]]
        A dictionary containing the configuration for each supported dataset, including
        the corresponding torchvision.datasets class (`class`), mean and standard
        deviation for channel-wise normalization (`mean`, `std`).
    """

    DATASETS = {
        "mnist": {
            "class": torchvision.datasets.MNIST,
            "mean": (0.1307,),
            "std": (0.3081,),
            "type": "standard",
        },
        "fmnist": {
            "class": torchvision.datasets.FashionMNIST,
            "mean": (0.2860,),
            "std": (0.3530,),
            "type": "standard",
        },
        "svhn": {
            "class": torchvision.datasets.SVHN,
            "mean": (0.4377, 0.4438, 0.4728),
            "std": (0.1980, 0.2010, 0.1970),
            "type": "svhn",
        },
        "cifar10": {
            "class": torchvision.datasets.CIFAR10,
            "mean": (0.4914, 0.4822, 0.4465),
            "std": (0.2470, 0.2435, 0.2616),
            "type": "standard",
        },
    }

    def __init__(self) -> None:
        self.path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "..", "datasets")
        )

    def _load_standard_dataset(
        self, dataset_class: type, transforms: torchvision.transforms.Compose
    ) -> tuple[VisionDataset, VisionDataset, list]:
        """Load datasets that use train = [True/False] API (MNIST, Fashion-MNIST,
        CIFAR-10).

        Parameters
        ----------
        dataset_class : type
            The torchvision dataset class to instantiate.
        transforms : torchvision.transforms.Compose
            Transformation to be applied to the data.

        Returns
        -------
        tuple[VisionDataset, VisionDataset, list]
            A tuple containing the full training dataset, test dataset, and list of
            targets. The latter are needed for stratified splitting.
        """
        full_train_dataset = dataset_class(
            self.data_dir, train=True, transform=transforms, download=True
        )
        test_dataset = dataset_class(
            self.data_dir, train=False, transform=transforms, download=True
        )
        return (full_train_dataset, test_dataset, full_train_dataset.targets)

    def _load_svhn_dataset(
        self, dataset_class: type, transforms: torchvision.transforms.Compose
    ) -> tuple:
        """Load datasets that use split = ['train'/'test'] API (SVHN).

        Parameters
        ----------
        dataset_class : type
            The torchvision dataset class to instantiate.
        transforms : torchvision.transforms.Compose
            Transformation to be applied to the data.

        Returns
        -------
        tuple[VisionDataset, VisionDataset, list]
            A tuple containing the full training dataset, test dataset, and list of
            targets. The latter are needed for stratified splitting.
        """
        full_train_dataset = dataset_class(
            self.data_dir, split="train", transform=transforms, download=True
        )
        test_dataset = dataset_class(
            self.data_dir, split="test", transform=transforms, download=True
        )
        return (full_train_dataset, test_dataset, full_train_dataset.labels)

    def get_dataloaders(
        self,
        dataset_name: str,
        val_split: float = 0.1,
        batch_size: int = 256,
        num_workers: int = 4,
        random_state: int = 42,
    ) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader]:
        """Instantiate dataloaders for the specified dataset.

        This method applies channel-wise normalization to the given dataset. Then,
        performs a stratified performs a stratified train-validation split on the
        original training set. Finally, returns the dataloaders for the full training
        set, the training and validation splits, and the test set.

        Parameters
        ----------
        dataset_name : str
            Name of the dataset. Options are 'mnist', 'fmnist', 'svhn', and 'cifar10'.
        val_split : float, optional, default=0.1
            Validation split percentage.
        batch_size : int, optional, default=256
            Batch size for dataloaders.
        num_workers : int, optional, default=4
            Number of worker processes for data loading.
        random_state : int, optional, default=42
            Random seed for reproducibility.

        Returns
        -------
        tuple[DataLoader, DataLoader, DataLoader, DataLoader]
            A tuple of dataloaders for the original full training set, the training and
            validation splits, and the test set, respectively.

        Raises
        ------
        ValueError
            If `dataset_name` is not one of the supported datasets.
        """
        if dataset_name not in self.DATASETS:
            raise ValueError(
                f"Unsupported dataset {dataset_name} is not supported. "
                f"Options are {list(self.DATASETS.keys())}."
            )

        config = self.DATASETS[dataset_name]

        # transformations
        transforms = torchvision.transforms.Compose(
            [
                torchvision.transforms.ToTensor(),
                torchvision.transforms.Normalize(
                    mean=config["mean"], std=config["std"]
                ),
            ]
        )

        if config["type"] == "standard":
            full_train_dataset, test_dataset, targets = self._load_standard_dataset(
                dataset_class=config["class"], transforms=transforms
            )
        elif config["type"] == "svhn":
            full_train_dataset, test_dataset, targets = self._load_svhn_dataset(
                dataset_class=config["class"], transforms=transforms
            )

        # stratified train-val split
        train_indices, val_indices = train_test_split(
            np.arange(len(full_train_dataset)),
            test_size=val_split,
            random_state=random_state,
            stratify=targets,
        )
        train_dataset = Subset(full_train_dataset, train_indices)
        val_dataset = Subset(full_train_dataset, val_indices)

        # create dataloaders
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

        return (
            full_train_dataloader,
            train_dataloader,
            val_dataloader,
            test_dataloader,
        )
