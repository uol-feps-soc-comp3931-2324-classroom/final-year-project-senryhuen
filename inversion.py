import torch
import torchaudio

from audio import STFT_CONFIG


def griffinlim(spectrogram: torch.Tensor, n_iter: int = 100) -> torch.Tensor:
    """Passes through to `torchaudio.functional.griffinlim()`

    Assumes `spectrogram` generated using global config in `audio`. The
    same parameters are passed to `torchaudio.funcitonal.griffinlim()`.

    Refer to "https://pytorch.org/audio/2.0.0/generated/torchaudio.
    functional.griffinlim.html".

    """
    return torchaudio.functional.griffinlim(
        spectrogram,
        window=STFT_CONFIG["window_fn"](STFT_CONFIG["window_length"]),
        n_fft=STFT_CONFIG["n_fft"],
        hop_length=STFT_CONFIG["hop_length"],
        win_length=STFT_CONFIG["window_length"],
        power=1,
        n_iter=n_iter,
        momentum=0,
        length=None,
        rand_init=True,
    )
