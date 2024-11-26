#!/usr/bin/env python3
import argparse
import csv
import os
from datetime import datetime
from typing import List

import numpy as np
import torch
import torch.nn as nn

import src.data.data_utils as data_utils
from src.training.solver import Solver
from src.networks.resnet import ResNet


def parse_list(value):
    """Parse a strig representing a list of integers to an actual list of integers."""
    try:
        value = value.strip('[]').split(',')
        return [int(x.strip()) for x in value]
    except:
        raise argparse.ArgumentTypeError(
            "Must be a list of integers. Examples are '[1]'', '[1, 1]'' and '[1, 1, 1]'"
        )


parser = argparse.ArgumentParser(description='hparams')
parser.add_argument(
    '--dataset', 
    type=str, 
    default='mnist',
    help='Dataset to test (default: mnist). Options: mnist, fmnist, svhn, cifar10.'
)
parser.add_argument('--num_blocks', type=parse_list, help="Number of blocks (default: '[1]')")
parser.add_argument('--optimizer', type=str, default='sgd', help='optimizer (default: sgd)')
parser.add_argument('--lr', type=float, default=0.1, help='learning rate (default: 0.1)')
parser.add_argument('--epochs', type=int, default=50, help='number of training epochs (default: 50)')
parser.add_argument('--patience', type=int, default=10, help='patience epochs (default: 10)')
parser.add_argument('--batch_size', type=int, default=256, help='mini-batch size (default: 256)')
parser.add_argument('--n_trials', type=int, default=10, help='number of trials (default: 10)')
parser.add_argument('--show_progress', type=bool, default=False, help='whether to show tqdm progress or not')


def main() -> None:
    args = parser.parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load data
    if args.dataset == 'mnist':
        _, train_dataloader, val_dataloader, test_dataloader = data_utils.get_mnist(batch_size=args.batch_size)
    elif args.dataset == 'fmnist':
        _, train_dataloader, val_dataloader, test_dataloader = data_utils.get_fmnist(batch_size=args.batch_size)
    elif args.dataset == 'svhn':
        _, train_dataloader, val_dataloader, test_dataloader = data_utils.get_svhn(batch_size=args.batch_size)
    elif args.dataset == 'cifar10':
        _, train_dataloader, val_dataloader, test_dataloader = data_utils.get_cifar10(batch_size=args.batch_size)
    else:
        raise ValueError(
            f"Invalid dataset: {args.dataset}. Options are 'mnist', 'fmnist', 'svhn' and 'cifar10'!"
        )
    
    metrics = np.zeros((args.n_trials, 4))
    for i in range(args.n_trials):
        print(f'----- TRIAL {i+1}/{args.n_trials}')

        # Initialize model
        model = ResNet(
            in_channels=train_dataloader.dataset[0][0].shape[0],
            num_blocks=args.num_blocks, 
        )

        # Training
        solver = Solver(
            device=device,
            model=model,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            optimizer=args.optimizer,
            lr=args.lr,
        )
        results = solver.train(
            epochs=args.epochs,
            patience=args.patience,
            show_progress=args.show_progress
        )

        # Save metrics of current trial
        metrics[i, 0] = results['test_accuracy']
        metrics[i, 1] = results['carbon_data'].duration
        metrics[i, 2] = results['carbon_data'].emissions
        metrics[i, 3] = results['carbon_data'].energy_consumed

    means, stds = np.mean(metrics, axis=0), np.std(metrics, axis=0)
    params = filter(lambda p: p.requires_grad, model.parameters())
    n_params = sum([np.prod(p.size()) for p in params])
    print(
        '##################################################\n'
        f'Mean test accuracy: {means[0]:.4f} ± {stds[0]:.4f}\n',
        f'Mean training time: {means[1]:.2f} ± {stds[1]:.4f} (s)\n',
        f'Mean emissions: {means[2]:.2f} ± {stds[2]:.4f} (kg)\n',
        f'Mean energy consumed: {means[3]:.2f} ± {stds[3]:.4f} (kWh)\n'
        f'Number of trainable parameters: {n_params}\n'
        '##################################################'
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
    args: argparse.Namespace
) -> None:
    """
    Args:
        model_name: name of the model class
        n_params: number of trainable parameters
        means: tuple of averages on metrics of interest
        stds: tuple of standard deviations on metrics of interest
    """
    output = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'config': vars(args),
        'results': {
            'test_accuracy': {'mean': f'{means[0]:.4f}', 'std': f'{stds[0]:.4f}'},
            'training_time': {'mean': f'{means[1]:.4f}', 'std': f'{stds[1]:.4f}'},
            'emissions': {'mean': f'{means[2]:.4f}', 'std': f'{stds[2]:.4f}'},
            'energy_consumed': {'mean': f'{means[3]:.4f}', 'std': f'{stds[3]:.4f}'}
        }
    }
    os.makedirs('results', exist_ok=True)
    out_file = f'results/{model_name}_{args.dataset}.csv'
    with open(out_file, 'a', newline='') as outf:
        writer = csv.DictWriter(outf, fieldnames=output.keys())
        writer.writerow(output)


if __name__ == '__main__':
    main()