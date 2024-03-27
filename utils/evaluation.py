import torch
import numpy as np


def _to_decibel(num):
    return 10 * np.log10(num)


def rmse(tensor: torch.Tensor, other_tensor: torch.Tensor, scale: bool = True) -> float:
    """Root-mean-squared-error between `tensor` and `other_tensor`

    Both tensors' values are scaled to be between 0 and 1 for fair
    comparison. This also allows the RMSE to be expressed as a
    percentage easily (multiply by 100). This can be disabled by
    setting `scale` to False.

    """
    if scale:
        tensor = make_equal_scale(tensor)
        other_tensor = make_equal_scale(other_tensor)

    return np.sqrt(torch.mean(torch.square(tensor - other_tensor)))


def snr(tensor: torch.Tensor, other_tensor: torch.Tensor, scale: bool = True) -> float:
    """Signal-to-noise ratio of `other_tensor` to `tensor`

    Both tensors' values are scaled to be between 0 and 1 for fair
    comparison. This can be disabled by setting `scale` to false.

    """
    if scale:
        tensor = make_equal_scale(tensor)
        other_tensor = make_equal_scale(other_tensor)

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


def make_equal_scale(tensor: torch.Tensor) -> torch.Tensor:
    """Scales tensor to have values in range between 0 and 1

    Args:
        tensor (torch.Tensor): tensor to be scaled, must be of type
            float32, but no specific shape required since all values
            will be scaled.

    Raises:
        ValueError: if `tensor` is not of expected type float32.
            
    Returns:
        torch.Tensor: new tensor with same shape and type as `tensor`,
            but with values scaled to be between 0 and 1.

    """
    if tensor.dtype != torch.float32:
        raise ValueError("Expected type of `tensor` to be float32")

    return (tensor - tensor.min()) / (tensor.max() - tensor.min())
