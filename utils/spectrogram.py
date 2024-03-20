import os
import numpy as np
import torch
import librosa
import skimage


def load_spectrogram(filepath: str) -> torch.Tensor:
    """Load a spectrogram from an image to a torch.Tensor

    Supports: .tiff, .png, .jpg, .jpeg

    Args:
        filepath (str): filepath to spectrogram image to load.

    Raises:
        ValueError: if `filepath` is not for a supported image file
            format.

    Returns:
        torch.Tensor: spectrogram in the same format as from
            `torch.stft()`.

    """
    # check filepath is for a supported image format
    supported_exts = [".tiff", ".png", ".jpg", ".jpeg"]
    ext = os.path.splitext(filepath)[1].lower()
    if ext not in supported_exts:
        raise ValueError(f"Invalid filepath '{filepath}', '{ext}' is not a supported image format")

    img = skimage.io.imread(filepath)
    img = img.astype(np.float32)

    if ext != ".tiff":
        img = np.expand_dims(img, 0)
        img = (img - 128) / 3

    img = np.flip(img, axis=1)
    img = librosa.db_to_amplitude(img, ref=1)
    return torch.from_numpy(img)


def save_spectrogram(spectrogram: torch.Tensor, save_path: str):
    """Save a spectrogram (from `torch.stft()`) as an image

    Supports saving as: .tiff, .png, .jpg, .jpeg

    Args:
        spectrogram (torch.Tensor): spectrogram in the same format as
            from `torch.stft()`.
        save_path (str): path to save the spectrogram image.

    Raises:
        ValueError: if `save_path` is not for a supported image format.
        ValueError: if `spectrogram` is not single channel.

    """
    # check save_path is for a supported image format
    supported_exts = [".tiff", ".png", ".jpg", ".jpeg"]
    ext = os.path.splitext(save_path)[1].lower()
    if ext not in supported_exts:
        raise ValueError(f"Invalid save_path '{save_path}', '{ext}' is not a supported image format")

    # each channel should be separate image
    if spectrogram.shape[0] > 1:
        raise ValueError(
            "Spectrogram is multichannel, expected single channel spectrogram"
        )

    save_dir = os.path.dirname(save_path)
    if not os.path.exists(save_dir) and save_path[:5] == "data/":
        os.makedirs(save_dir)

    img = spectrogram.numpy()
    img = librosa.amplitude_to_db(img, ref=1)
    img = np.flip(img, axis=1)  # flip so frequency is increasing instead (along x-axis)

    if ext != ".tiff":
        # pixels are 8bit integer values, scale spectrogram to this range
        img = (( (img - img.min()) / (img.max() - img.min()) ) * 255).astype(np.uint8)

    skimage.io.imsave(save_path, img)


def merge_to_multichannel(*spec: torch.Tensor) -> torch.Tensor:
    """Merge multiple spectrogram tensors into one multichannel tensor

    All channels need to be of the same shape.

    Args:
        *audio (torch.Tensor): spectrograms in the same format as
            from `torch.stft()`.

    Returns:
        torch.Tensor: spectrogram in the same format as
            from `torch.stft()`.

    """
    return torch.cat(spec, 0)


def separate_phase_to_channel(complex_spectrogram: torch.Tensor) -> torch.Tensor:
    """Separates real and imaginary components into separate channels

    Imaginary components are represented as real numbers, placed in a
    separate channel.

    Args:
        complex_spectrogram (torch.Tensor): complex-valued spectrogram
            in the same format as from `torch.stft()`.

    Returns:
        torch.Tensor: real-valued spectrogram in the same format as
            from `torch.stft()`, except channels may represent real or
            imaginary components.

    """
    return merge_to_multichannel(
        torch.real(complex_spectrogram), torch.imag(complex_spectrogram)
    )


def merge_separate_phase_channel(two_channel_spectrogram: torch.Tensor) -> torch.Tensor:
    """Merge two channels into single channel complex spectrogram

    Args:
        two_channel_spectrogram (torch.Tensor): real-valued spectrogram
            in the same format as from `torch.stft()`, with two
            channels, where the first channel represents the real
            component, and second channel represents the imaginary
            component.

    Returns:
        torch.Tensor: single channel complex-valued spectrogram in the
            same format as from `torch.stft()`.

    """
    if two_channel_spectrogram.shape[0] != 2 or len(two_channel_spectrogram.shape) != 3:
        return None  # bad input spectrogram shape

    return torch.complex(
        two_channel_spectrogram[0:1, :, :],
        two_channel_spectrogram[1:, :, :],
    )
