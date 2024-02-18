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


def make_equal_length(
    audio_tensor: torch.Tensor, other_audio_tensor: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Truncate the longest tensor so that both tensors have the same length

    Args:
        audio_tensor (torch.Tensor): tensor with dimensions [channels, time],
            where each element in the first dimension is a channel, and
            each element in that element (second dimension) is an
            amplitude value at a point in time.
        other_audio_tensor (torch.Tensor): same as `audio_tensor`.

    Raises:
        ValueError: if audio tensors are not of the expected shape.

    Returns:
        tuple[torch.Tensor, torch.Tensor]: [0] `audio_tensor` and [1]
            `other_audio_tensor`, with either being truncated so that both
            are equal in length.

    """
    if len(audio_tensor.shape) != 2 or len(other_audio_tensor.shape) != 2:
        raise ValueError("Expected shape of audio tensors to be [channels, time]")

    trunc_len = min(audio_tensor.shape[-1], other_audio_tensor.shape[-1])
    return audio_tensor[:, :trunc_len], other_audio_tensor[:, :trunc_len]
