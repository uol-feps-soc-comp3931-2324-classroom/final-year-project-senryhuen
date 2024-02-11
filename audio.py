from typing import Callable
import os

import torch
import torchaudio
import librosa


def load_audio(filepath: str, multichannel: str = "keep") -> tuple[torch.Tensor, int]:
    """Load audio from a file into a torch.Tensor

    Supported file formats depends on the availability of backends,
    which is inherited from using `torch.load`. Refer to
    "https://pytorch.org/audio/stable/generated/torchaudio.load.html"
    to see how to check supported formats.

    Args:
        filepath (str): filepath to audio source file to load.
        multichannel (str, optional): options for handling multichannel
            audio. "keep" keeps multichannel, "merge" combines channels
            by mean average into single channel, "first" keeps only the
            first channel. Defaults to "keep".

    Returns:
        tuple[torch.Tensor, int]: [0] tensor with dimensions
            [channels, time], where each element in the first dimension
            is a channel, and each element in that element (second
            dimension) is an amplitude value at a point in time. [1]
            sample rate of loaded audio.

    """
    # audio, sample_rate = torchaudio.load(filepath)

    # using librosa.load as a work around to torchaudio.load broken backend
    audio, sample_rate = librosa.load(filepath, sr=None, mono=False)

    # needs conversion to tensor since Librosa loads as numpy array
    audio = torch.from_numpy(audio)
    if len(audio.shape) <= 1:
        audio = torch.unsqueeze(audio, 0)

    # handle multichannels
    if multichannel == "merge":
        audio = audio_to_mono(audio)
    elif multichannel == "first":
        audio = torch.unsqueeze(audio[0, :], 0)

    return audio, sample_rate


def save_audio(save_filepath: str, audio: torch.Tensor, sample_rate: int):
    """Passes through to `torchaudio.save()`

    Refer to "https://pytorch.org/audio/main/generated/torchaudio.save.html".

    """
    save_dir = os.path.dirname(save_filepath)
    if not os.path.exists(save_dir) and save_filepath[:5] == "data/":
        os.makedirs(save_dir)

    torchaudio.save(save_filepath, audio, sample_rate)


def audio_to_spectrogram(
    audio: torch.Tensor,
    n_fft: int = 2048,
    hop_length: int = 512,
    window_fn: Callable = torch.hamming_window,
) -> torch.Tensor:
    """Passes through to `torch.stft()`

    Refer to "https://pytorch.org/docs/stable/generated/torch.stft.html".

    """
    window = window_fn(n_fft)
    return torch.real(torch.stft(audio, n_fft, hop_length, window=window, return_complex=True))


def audio_to_mono(audio: torch.Tensor) -> torch.Tensor:
    """Merge multichannels into single channel that is mean average of all the channels

    Args:
        audio (torch.Tensor): tensor with dimensions [channels, time],
            where each element in the first dimension is a channel, and
            each element in that element (second dimension) is an
            amplitude value at a point in time.

    Returns:
        torch.Tensor: tensor with same dimensions as `audio` parameter,
            but with just one channel.

    """
    # if already mono
    if audio.shape[0] <= 1:
        return audio

    return torch.unsqueeze(torch.mean(audio, 0), 0)


def merge_to_multichannel(*audio: torch.Tensor) -> torch.Tensor:
    """Merge multiple tensors into one multichannel tensor

    All channels need to be of the same length, and it is assumed they
    are of the same sample rate.

    Args:
        *audio (torch.Tensor): tensors with dimensions [channels, time],
            where each element in the first dimension is a channel, and
            each element in that element (second dimension) is an
            amplitude value at a point in time.

    Returns:
        torch.Tensor: tensor with dimensions [channels, time], where
            each element in the first dimension is a channel, and each
            element in that element (second dimension) is an amplitude
            value at a point in time.

    """
    return torch.cat(audio, 0)
