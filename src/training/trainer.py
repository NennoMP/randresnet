__all__ = ["Trainer"]

import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
from torch.utils.data import DataLoader
from tqdm import tqdm


class Trainer:
    """Trainer class for fully-trainable models for classification tasks.

    Trains a model via backpropagation with early-stoppin and tracking of carbon data,
    including training time, emissions, and energy consumption.

    Parameters
    ----------
    model : nn.Module
        The model to be trained.
    train_dataloader, val_dataloader, test_dataloader : DataLoader
        DataLoader for the training, validation, and test datasets, respectively.
    optimizer : torch.optim.Optimizer, optional, default=None
        Optimizer for training the model. If None, Adam optimizer is used by default.
    criterion : nn.Module, optional, default=None
        Loss function for training the model. If None, CrossEntropyLoss is used by
        default.
    device : torch.device, optional, default=None
        Device to run the model on. If None, it defaults to CUDA if available,
        otherwise CPU.

    Attributes
    ----------
    best_epoch : int
        The epoch on which the best validation accuracy was achieved.
    best_val_accuracy : float
        The best validation accuracy achieved during training.
    patience_count : int
        Counter for early-stopping patience.
    best_params : dict
        The model parameters corresponding to the best validation accuracy.
    history : dict
        Dictionary containing training and validation accuracy history.
    results : dict
        Dictionary containing final training, validation, and test accuracies, as well
        as carbon tracking data.
    """

    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: torch.optim.Optimizer | None = None,
        criterion: nn.Module | None = None,
        device: torch.device | None = None,
    ) -> None:
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.optimizer = optimizer
        if optimizer is None:
            self.optimizer = torch.optim.Adam(params=self.model.parameters())

        self.criterion = criterion
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()

        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._reset()

    def _reset(self) -> None:
        """Reset tracking data."""
        self.best_epoch = 0
        self.best_val_accuracy = float("-inf")
        self.patience_count = 0
        self.best_params = None
        self.history = {
            "train_accuracy": [],
            "val_accuracy": [],
        }
        self.results = {
            "train_accuracy": None,
            "val_accuracy": None,
            "test_accuracy": None,
            "carbon_data": None,
        }

    def _run_epoch(
        self, dataloader: DataLoader, training: bool, show_progress: bool
    ) -> float:
        """Run a single epoch of training or inference.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader for the dataset to run the epoch on.
        training : bool
            Whether to run the epoch in training mode or evaluation mode.
        show_progress : bool
            Whether to show a progress bar using tqdm or not.

        Returns
        -------
        accuracy : float
            Classification accuracy of the model on the provided dataset.
        """
        self.model.train() if training else self.model.eval()

        correct = total = 0
        for inputs, targets in tqdm(dataloader, disable=not show_progress):
            inputs, targets = inputs.to(self.device), targets.to(self.device)

            if training:
                self.optimizer.zero_grad()

            outputs = self.model(inputs)
            if targets.dim() > 1:
                targets = targets.squeeze(1)

            loss = self.criterion(outputs, targets)

            if training:
                loss.backward()
                self.optimizer.step()

            _, preds = torch.max(outputs, 1)
            correct += torch.sum(preds == targets.data)
            total += targets.size(0)

            accuracy = correct / total
            return accuracy

    def train(
        self, epochs: int = 100, patience: int = 10, show_progress: bool = False
    ) -> dict[str, float | None]:
        """Train the model with early stopping based on validation accuracy.

        Tracks efficiency metrics with respect to the training process, including
        training time (seconds), CO2 emissions (kg) and energy consumed (Wh).

        Parameters
        ----------
        epochs : int, optional, default=100
            Maximum number of epochs to train the model.
        patience : int, optional, default=10
            Number of epochs without improvement on validation accuracy before applying
            early-stopping.
        show_progress : bool, optional, default=False
            Whether to show a progress bar using tqdm or not.

        Returns
        -------
        results : dict[str, Any]
            Dictionary containing training (`train_accuracy`), validation and
            (`val_accuracy`), test (`test_accuracy`) accuracies as floats. Also
            includes carbon tracking data (`carbon_data`) as an object containing
            training time (`duration`), emissions (`emissions`), and energy consumption
            (`energy_consumed`).
        """
        # Start carbon.io tracking
        carbon_tracker = EmissionsTracker(log_level="error", save_to_file=False)
        carbon_tracker.start()

        self.model.to(self.device)

        # Train the model
        epoch = 0
        while epoch < epochs and self.patience_count <= patience:
            train_accuracy = self._run_epoch(
                self.train_dataloader, training=True, show_progress=show_progress
            )
            val_accuracy = self._run_epoch(
                self.val_dataloader, training=False, show_progress=show_progress
            )

            self.history["train_accuracy"].append(train_accuracy)
            self.history["val_accuracy"].append(val_accuracy)
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Accuracy: {train_accuracy:.4f}, "
                f"Val Accuracy: {val_accuracy:.4f}."
            )

            # If validation accuracy improved, update best model parameters
            if val_accuracy >= self.best_val_accuracy:
                self.patience_count = 0
                self.best_epoch = epoch + 1
                self.best_val_accuracy = val_accuracy
                self.best_params = {
                    k: v.cpu().clone() for k, v in self.model.state_dict().items()
                }

            # Check early-stopping
            if val_accuracy <= self.best_val_accuracy:
                self.patience_count += 1
                if self.patience_count > patience:
                    print(f"Early stopping at epoch {epoch + 1}.")

            epoch += 1

        print(f"Restoring best weights from epoch {self.best_epoch}.")
        self.model.load_state_dict(self.best_params)  # restore best parameters

        # End carbon.io tracking
        carbon_data = carbon_tracker._prepare_emissions_data()
        _ = carbon_tracker.stop()

        train_accuracy = self._run_epoch(self.train_dataloader)
        val_accuracy = self._run_epoch(self.val_dataloader)
        test_accuracy = self._run_epoch(self.test_dataloader)
        print(
            f"Train Accuracy: {train_accuracy:.4f} - "
            f"Val Accuracy: {val_accuracy:.4f} - "
            f"Test Accuracy: {test_accuracy:.4f}"
        )
        print(
            f"Training time: {carbon_data.duration:.4f} - "
            f"Emissions: {carbon_data.emissions:.4f} - "
            f"Energy:  {carbon_data.energy_consumed:.4f}"
        )
        self.results["train_accuracy"] = train_accuracy
        self.results["val_accuracy"] = val_accuracy
        self.results["test_accuracy"] = test_accuracy
        self.results["carbon_data"] = carbon_data

        return self.results
