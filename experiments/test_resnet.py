#!/usr/bin/env python3
import argparse
import csv
import os
from datetime import datetime

import numpy as np
import torch

from src.data import VisionDataLoader
from src.models import ResNet
from src.training import Trainer


def parse_list(value):
    """Parse a string representing a list of integers to an actual list of integers."""
    try:
        value = value.strip("[]").split(",")
        return [int(x.strip()) for x in value]
    except Exception as e:
        raise argparse.ArgumentTypeError(
            f"Could not parse '{value}' as a list of integers."
        ) from e


parser = argparse.ArgumentParser(description="hparams")
parser.add_argument(
    "--dataset",
    type=str,
    default="mnist",
    help="Dataset to test (default: mnist). Options: mnist, fmnist, svhn, cifar10.",
)
parser.add_argument(
    "--num_blocks", type=parse_list, help="Number of blocks (default: '[1]')"
)
parser.add_argument(
    "--optimizer", type=str, default="sgd", help="optimizer (default: sgd)"
)
parser.add_argument(
    "--lr", type=float, default=0.1, help="learning rate (default: 0.1)"
)
parser.add_argument(
    "--epochs", type=int, default=50, help="number of training epochs (default: 50)"
)
parser.add_argument(
    "--patience", type=int, default=10, help="patience epochs (default: 10)"
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
    _, train_dataloader, val_dataloader, test_dataloader = loader.get_dataloaders(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
    )

    # Run multiple trials
    metrics = np.zeros((args.n_trials, 4))
    for i in range(args.n_trials):
        print(f"----- TRIAL {i + 1}/{args.n_trials}")

        # Initialize the model
        model = ResNet(
            in_channels=train_dataloader.dataset[0][0].shape[0],
            num_blocks=args.num_blocks,
        )

        if args.optimizer == "sgd":
            optimizer = torch.optim.SGD(
                params=model.parameters(), lr=args.lr, momentum=0.9, weight_decay=1e-4
            )
        elif args.optimizer == "adam":
            optimizer = torch.optim.Adam(
                params=model.parameters(), lr=args.lr, weight_decay=1e-4
            )
        else:
            raise ValueError(
                f"Optimizer '{args.optimizer}' not recognized. "
                f"Options are 'sgd' and 'adam'."
            )
        criterion = torch.nn.CrossEntropyLoss()
        # Training
        trainer = Trainer(
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        results = trainer.train(
            epochs=args.epochs, patience=args.patience, show_progress=args.show_progress
        )

        # Save metrics of current trial
        metrics[i, 0] = results["test_accuracy"]
        metrics[i, 1] = results["carbon_data"].duration
        metrics[i, 2] = results["carbon_data"].emissions
        metrics[i, 3] = results["carbon_data"].energy_consumed

    means, stds = np.mean(metrics, axis=0), np.std(metrics, axis=0)
    params = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum(np.prod(p.size()) for p in params)
    print(
        "##################################################\n"
        f"Mean test accuracy: {means[0]:.4f} ± {stds[0]:.4f}\n",
        f"Mean training time: {means[1]:.2f} ± {stds[1]:.4f} (s)\n",
        f"Mean emissions: {means[2]:.2f} ± {stds[2]:.4f} (kg)\n",
        f"Mean energy consumed: {means[3]:.2f} ± {stds[3]:.4f} (kWh)\n"
        f"Number of trainable parameters: {n_params}\n"
        "##################################################",
    )

    save_experiment(
        model_name=ResNet.__name__.lower(),
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
        "config": vars(args),
        "n_params": n_params,
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
