"""Module implementing training logic for randomized models on classification tasks.

SolverRidge trains a linear classifier on top of the features extracted through a single forward 
pass of a randomized model. The classifier is trained in closed-form using Ridge regression. 
"""
from tqdm import tqdm
from typing import Dict, Optional

import torch
import torch.nn as nn
from codecarbon import EmissionsTracker
from sklearn.linear_model import RidgeClassifier
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader


class SolverRidge:
    def __init__(
        self, 
        device : torch.device,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        classifier: str = 'svd',
        reg: float = 0,
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

        if classifier == 'svd':
            self.classifier = RidgeClassifier(alpha=reg, solver='svd')
        else:
            raise ValueError(f"Invalid classifier: {classifier}. Options are 'svd'!")
        
        self._reset()
    
    def _reset(self) -> None:
        """Reset tracking data."""
        self.results = {
            'train_accuracy': None,
            'val_accuracy': None,
            'test_accuracy': None,
            'carbon_data': None,
        }

    @torch.no_grad()
    def evaluate(self, dataloader: DataLoader, show_progress: bool = False) -> float:
        """Evaluate the model on the provided dataloader.

        Args:
            dataloader: dataloader to evaluate the model on.
            show_progress: whether to show tqdm progress bar or not

        Returns:
            accuracy: accuracy of model predictions.
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
        accuracy = accuracy_score(targets, self.classifier.predict(outputs))

        return accuracy

    @torch.no_grad()
    def train(self, show_progress: bool = False) -> Dict[str, Optional[float]]:
        """
        Fit the ridge regressor on top of the features extracted by the randomized model. 
        Track efficiency metrics with respect to the training process: training time (seconds), 
        CO2 emissions (kg) and energy consumed (Wh).

        Args:
            show_progress: whether to show tqdm progress bar or not

        Returns:
            results: a dictionary containing accuracies (train, validation, test) and efficiency 
                metrics (training time, emissions and energy consumed).
        """
        # Start carbon.io tracking
        carbon_tracker = EmissionsTracker(log_level='error', save_to_file=False)
        carbon_tracker.start()

        self.model.to(self.device)
        outputs, targets = [], []
        for x, y in tqdm(self.train_dataloader, disable=not show_progress):
            x, y = x.to(self.device), y.to(self.device)
            out = self.model(x)
            outputs.append(out.cpu())
            targets.append(y.cpu())

        outputs = torch.cat(outputs, dim=0)
        targets = torch.cat(targets, dim=0)
        self.classifier.fit(outputs, targets)

        # End carbon.io tracking
        carbon_data = carbon_tracker._prepare_emissions_data()
        _ = carbon_tracker.stop()

        train_accuracy = accuracy_score(targets, self.classifier.predict(outputs))
        val_accuracy = self.evaluate(self.val_dataloader)
        test_accuracy = self.evaluate(self.test_dataloader)

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
        self.results['train_accuracy'] = train_accuracy
        self.results['val_accuracy'] = val_accuracy
        self.results['test_accuracy'] = test_accuracy
        self.results['carbon_data'] = carbon_data
        
        return self.results