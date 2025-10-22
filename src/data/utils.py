"""Utility functions for data pre-processing."""

__all__ = ["compute_channel_mean_std"]

import torch
from torchvision.datasets.vision import VisionDataset


def compute_channel_mean_std(
    dataset: VisionDataset,
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """Computes channel-wise mean and standard deviation for a given vision dataset.

    Parameters
    ----------
    dataset : VisionDataset
        An instance of a torchvision VisionDataset.

    Returns
    -------
    tuple[tuple[float, ...], tuple[float, ...]]
        A tuple of two tuples, containing the mean and standard deviation for each
        channel, respectively.
    """
    data = (
        torch.stack(tensors=[dataset[i][0] for i in range(len(dataset))], dim=0)
        .cpu()
        .numpy()
    )

    # compute mean and std deviation for each channel
    if data.shape[1] == 1:  # gray-scale
        mean = (data[:, 0, :, :].mean(),)
        std = (data[:, 0, :, :].std(),)
    else:  # RGB
        mean = data[:, 0, :, :].mean(), data[:, 1, :, :].mean(), data[:, 2, :, :].mean()
        std = data[:, 0, :, :].std(), data[:, 1, :, :].std(), data[:, 2, :, :].std()

    return (mean, std)
