import torch
import numpy as np


def _to_decibel(num):
    return 10 * np.log10(num)


def rmse(tensor: torch.Tensor, other_tensor: torch.Tensor) -> float:
    """Root-mean-squared-error between `tensor` and `other_tensor`"""
    return np.sqrt(torch.mean(torch.square(tensor - other_tensor)))


def snr(tensor: torch.Tensor, other_tensor: torch.Tensor) -> float:
    """Signal-to-noise ratio of `other_tensor` to `tensor`"""
    return _to_decibel(
        torch.sum(torch.square(tensor)) / torch.sum(torch.square(tensor - other_tensor))
    ).item()
