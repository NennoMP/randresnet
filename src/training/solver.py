"""Module implementing training logic for fully-trainable models."""
from tqdm import tqdm
from typing import Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from codecarbon import EmissionsTracker


class Solver:
    def __init__(
        self, 
        device : torch.device,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: DataLoader,
        test_dataloader: DataLoader,
        optimizer: str = 'sgd',
        lr: float = 1e-1,
        momentum: float = 0.9,
        weight_decay: float = 1e-4,
    ) -> None:
        """
        Args:
            device: the device to run the model on.
            model: the instance of the randomized model to use as a feature extractor.
            train_dataloader: training data dataloader.
            val_dataloader: validation data dataloader.
            test_dataloader: test data dataloader.
            optimizer: optimizer to use for training. Options are 'sgd' and 'adam'.

        Raises:
            ValueError: if an invalid optimizer is provided.
        """
        self.device = device
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        if optimizer == 'sgd':
            self.optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=lr,
                momentum=momentum,
                weight_decay=weight_decay,
            )
        elif optimizer == 'adam':
            self.optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
            )
        else:
            raise ValueError(f"Invalid optimizer: {optimizer}. Options are 'sgd' and 'adam'!")
        self.criterion = nn.CrossEntropyLoss()
        
        self._reset()
    
    def _reset(self) -> None:
        """Reset tracking data."""
        self.best_epoch = 0
        self.best_weights = None
        self.best_val_accuracy = float('-inf')
        self.patience_count = 0

        self.history = {
            'train_accuracy': [],
            'val_accuracy': [],
        }
        
        self.results = {
            'train_accuracy': None,
            'val_accuracy': None,
            'test_accuracy': None,
            'carbon_data': None,
        }

    def _run_epoch(self, dataloader: DataLoader, training: bool = False) -> float:
        """Run a single epoch of training or validation.
        
        Args:
            dataloader: dataloader to run the epoch on.
            training: if True put the model in training mode, otherwise in evaluation mode.

        Returns:
            accuracy: accuracy of model predictions.
        """
        self.model.train() if training else self.model.eval()

        correct = total = 0
        with torch.set_grad_enabled(training):
            for (inputs, targets) in tqdm(dataloader, disable=not self.show_progress):
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
        
    def train(self, epochs: int = 50, patience: int = 10, show_progress: bool = False) -> Dict[str, Optional[float]]:
        """
        Train the model up to a maximum number of specified number of epochs, and with an early-stopping mechanism determined by the patience value.

        Args:
            epochs: maximum number of epochs to train the model.
            patience: patience value for early stopping.
            show_progress: whether to show tqdm progress bar or not

        Returns:
            results: a dictionary containing accuracies (train, validation, test) and efficiency 
                metrics (training time, emissions and energy consumed).
        """
        self.show_progress = show_progress
        # Start carbon.io tracking
        carbon_tracker = EmissionsTracker(log_level='error', save_to_file=False)
        carbon_tracker.start()
        
        self.model.to(self.device)        
        epoch = 0
        while epoch < epochs and self.patience_count <= patience:
            train_accuracy = self._run_epoch(self.train_dataloader, training=True)
            val_accuracy = self._run_epoch(self.val_dataloader)
            
            self.history['train_accuracy'].append(train_accuracy)
            self.history['val_accuracy'].append(val_accuracy)
            print(
                f"Epoch {epoch + 1}/{epochs} - "
                f"Train Accuracy: {train_accuracy:.4f}, Val Accuracy: {val_accuracy:.4f}"
            )

            # If validation accuracy improved, save the model weights
            if val_accuracy >= self.best_val_accuracy:
                self.patience_count = 0
                self.best_epoch = epoch + 1
                self.best_val_accuracy = val_accuracy
                self.best_params = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}

            # Check for early stopping
            if val_accuracy <= self.best_val_accuracy:
                self.patience_count += 1
                if self.patience_count > patience:
                    print(f"Early stopping at epoch {epoch + 1}.")

            epoch += 1     

        # Restore best model weights
        print(f"Restoring best weights from epoch {self.best_epoch}.")
        self.model.load_state_dict(self.best_params)
        
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
        self.results['train_accuracy'] = train_accuracy
        self.results['val_accuracy'] = val_accuracy
        self.results['test_accuracy'] = test_accuracy
        self.results['carbon_data'] = carbon_data
        
        return self.results