import os
import numpy as np
import torch
import librosa
import skimage


def load_spectrogram_tiff(filepath: str) -> torch.Tensor:
    """Load a spectrogram from a .tiff image to a torch.Tensor

    Args:
        filepath (str): filepath to spectrogram '.tiff' image to load.

    Raises:
        ValueError: if `filepath` is not for a '.tiff' file.

    Returns:
        torch.Tensor: spectrogram in the same format as from
            `torch.stft()`.

    """
    # check filepath is for a tiff image
    if os.path.splitext(filepath)[1].lower() != ".tiff":
        raise ValueError(f"Invalid filepath '{filepath}', expected a '.tiff' file")

    img = skimage.io.imread(filepath)
    img = np.flip(img, axis=1)
    img = librosa.db_to_amplitude(img, ref=1)
    return torch.from_numpy(img)


def save_spectrogram_tiff(spectrogram: torch.Tensor, save_path: str):
    """Save a spectrogram (from `torch.stft()`) as a .tiff image

    Args:
        spectrogram (torch.Tensor): spectrogram in the same format as
            from `torch.stft()`.
        save_path (str): path to save the spectrogram image.

    Raises:
        ValueError: if `save_path` is not for a '.tiff' file.
        ValueError: if `spectrogram` is not single channel.

    """
    # check save_path is for a tiff image
    if os.path.splitext(save_path)[1].lower() != ".tiff":
        raise ValueError(f"Invalid save_path '{save_path}', expected a '.tiff' file")

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
    img = np.flip(img, axis=1)  # flip  so frequency is increasing instead (along x-axis)
    skimage.io.imsave(save_path, img)
