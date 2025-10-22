#!/usr/bin/env python3

import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch
from sklearn.linear_model import RidgeClassifier

from src.data import VisionDataLoader
from src.models import randResNet
from src.training import TrainerRidge

parser = argparse.ArgumentParser(description="hparams")
parser.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    help="Dataset to test (default: mnist). Options: mnist, fmnist, svhn, cifar10.",
)
parser.add_argument(
    "--n_filters", type=int, default=64, help="number of filters (default: 64)"
)
parser.add_argument(
    "--n_layers", type=int, default=2, help="number of layers (default: 2)"
)
parser.add_argument(
    "--scaling",
    type=float,
    default=0.1,
    help="weight scaling for uniform init (default: 0.1)",
)
parser.add_argument(
    "--alpha",
    type=float,
    default=1.0,
    help="residual branch coefficient (default: 1.0)",
)
parser.add_argument(
    "--beta",
    type=float,
    default=1.0,
    help="non-linear branch coefficient (default: 1.0)",
)
parser.add_argument(
    "--reg", type=float, default=0.0, help="regularization coefficient (default: 0.0)"
)
parser.add_argument(
    "--batch_size", type=int, default=256, help="mini-batch size (default: 256)"
)
parser.add_argument(
    "--n_trials", type=int, default=10, help="number of trials (default: 10)"
)
parser.add_argument(
    "--show_progress",
    type=bool,
    default=False,
    help="whether to show tqdm progress or not",
)


def main() -> None:
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load data
    loader = VisionDataLoader()
    full_train_dataloader, _, val_dataloader, test_dataloader = loader.get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
    )

    # Run multiple trials
    metrics = np.zeros((args.n_trials, 4))
    for i in range(args.n_trials):
        print(f"----- TRIAL {i + 1}/{args.n_trials}")

        # Initialize the (randomized) model
        model = randResNet(
            n_layers=args.n_layers,
            in_channels=full_train_dataloader.dataset[0][0].shape[0],
            n_filters=args.n_filters,
            scaling=args.scaling,
            alpha=args.alpha,
            beta=args.beta,
        )
        # Freeze all weights of the (untrained) CNN part
        # Note: @torch.no_grad() decorator is used in the training logic of randResNet
        # thus, this is not strictly necessary
        for param in model.parameters():
            param.requires_grad = False

        # Initialize the readout
        readout = RidgeClassifier(alpha=args.reg, solver="svd")

        # Training
        trainer = TrainerRidge(
            model=model,
            train_dataloader=full_train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            readout=readout,
            device=device,
        )
        results = trainer.train(show_progress=args.show_progress)

        # Save metrics of current trial
        metrics[i, 0] = results["test_accuracy"]
        metrics[i, 1] = results["carbon_data"].duration
        metrics[i, 2] = results["carbon_data"].emissions
        metrics[i, 3] = results["carbon_data"].energy_consumed

    means, stds = np.mean(metrics, axis=0), np.std(metrics, axis=0)
    n_params = trainer.readout.coef_.size + trainer.readout.intercept_.size
    print(
        "##################################################\n"
        f"Mean test accuracy: {means[0]:.4f} ± {stds[0]:.4f}\n"
        f"Mean training time: {means[1]:.2f} ± {stds[1]:.4f} (s)\n"
        f"Mean emissions: {means[2]:.2f} ± {stds[2]:.4f} (kg)\n"
        f"Mean energy consumed: {means[3]:.2f} ± {stds[3]:.4f} (kWh)\n"
        f"Number of trainable parameters: {n_params}\n"
        "##################################################"
    )

    save_experiment(
        model_name=randResNet.__name__.lower(),
        n_params=n_params,
        means=means,
        stds=stds,
        args=args,
    )


def save_experiment(
    model_name: str,
    n_params: int,
    means: np.ndarray,
    stds: np.ndarray,
    args: argparse.Namespace,
) -> None:
    """
    Args:
        model_name: name of the model class
        n_params: number of trainable parameters
        means: tuple of averages on metrics of interest
        stds: tuple of standard deviations on metrics of interest
    """
    output = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_name": model_name,
        "n_params": n_params,
        "config": vars(args),
        "results": {
            "test_accuracy": {"mean": f"{means[0]:.4f}", "std": f"{stds[0]:.4f}"},
            "training_time": {"mean": f"{means[1]:.4f}", "std": f"{stds[1]:.4f}"},
            "emissions": {"mean": f"{means[2]:.4f}", "std": f"{stds[2]:.4f}"},
            "energy_consumed": {"mean": f"{means[3]:.4f}", "std": f"{stds[3]:.4f}"},
        },
    }

    os.makedirs("results", exist_ok=True)
    out_file = f"results/{model_name}_{args.dataset}.csv"
    with open(out_file, "a", newline="") as outf:
        writer = csv.DictWriter(outf, fieldnames=output.keys())
        writer.writerow(output)


if __name__ == "__main__":
    main()
