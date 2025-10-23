__all__ = ["TrainerRidge"]

import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader
from tqdm import tqdm


class TrainerRidge:
    """Trainer class for randomized models for classification tasks.

    Extracts features via the forward pass of a randomized model, then trains a readout
    layer on the extracted features. The readout is implemented as a ridge classifier
    and optimized in closed-form solution. Tracks carbon data, including training time,
    emissions, and energy consumption.

    Parameters
    ----------
    model : nn.Module
        The instance of the randomized model to use as a feature extractor.
    train_dataloader, val_dataloader, test_dataloader : DataLoader
        DataLoader for the training, validation, and test datasets, respectively.
    readout : RidgeClassifier, optional, default=None
        An instance of a sklearn trainable readout. If None, a RidgeClassifier with SVD
        solver.
    device : torch.device, optional, default=None
        Device to run the model on. If None, it defaults to CUDA if available,
        otherwise CPU.

    Attributes
    ----------
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
        readout: RidgeClassifier | None,
        device: torch.device | None = None,
    ) -> None:
        """
        Args:
            device: the device to run the model on.
            model: the instance of the randomized model to use as a feature extractor.
            train_dataloader: training data dataloader.
            val_dataloader: validation data dataloader.
            test_dataloader: test data dataloader.
            classifier: linear classifier (readout) to be trained.
            reg: regularization coefficient for the classifier.

        Raises:
            ValueError: if an invalid classifier is provided.
        """
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.readout = readout
        if readout is None:
            self.readout = RidgeClassifier(solver="svd")

        self.device = device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._reset()

    def _reset(self) -> None:
        """Reset tracking data."""
        self.results = {
            "train_accuracy": None,
            "val_accuracy": None,
            "test_accuracy": None,
            "carbon_data": None,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, show_progress: bool) -> float:
        """Evaluate the model on the given dataloader.

        Parameters
        ----------
        dataloader : DataLoader
            DataLoader on which to run evaluation.
        show_progress : bool, optional
            Whether to show tqdm progress bar or not.

        Returns
        -------
        accuracy : float
            Classification accuracy.
        """
        self.model.eval()

        outputs, targets = [], []
        for x, y in tqdm(dataloader, disable=not show_progress):
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            outputs.append(out.cpu())
            targets.append(y.cpu())

        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        preds = self.readout.predict(outputs)

        return accuracy_score(y_true=targets, y_pred=preds)

    @torch.no_grad()
    def train(self, show_progress: bool = False) -> dict[str, float | None]:
        """Fit the readout on top of the features extracted by the randomized model.

        Tracks efficiency metrics with respect to the training process, including
        training time (seconds), CO2 emissions (kg) and energy consumed (Wh).

        Parameters
        ----------
        show_progress : bool, optional, default=False
            Whether to show tqdm progress bar or not.

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
        with EmissionsTracker(log_level="error", save_to_file=False) as tracker:
            self.model.to(self.device)

            outputs, targets = [], []
            for x, y in tqdm(self.train_dataloader, disable=not show_progress):
                x, y = x.to(self.device), y.to(self.device)
                out = self.model(x)
                outputs.append(out.cpu())
                targets.append(y.cpu())

            # Fit the readout
            outputs = torch.cat(outputs, dim=0)
            targets = torch.cat(targets, dim=0)
            self.readout.fit(outputs, targets)

            # Evaluate
            train_accuracy = accuracy_score(
                y_true=targets, y_pred=self.readout.predict(outputs)
            )
            val_accuracy = self.evaluate(
                dataloader=self.val_dataloader, show_progress=show_progress
            )
            test_accuracy = self.evaluate(
                dataloader=self.test_dataloader, show_progress=show_progress
            )

        # Get carbon data from tracker
        carbon_data = tracker._prepare_emissions_data()

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
