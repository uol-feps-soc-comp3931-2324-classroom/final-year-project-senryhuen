import torch

from utils import audio, spectrogram
from .degli_model import DNN
from .griffinlimalg import griffinlim


def deepgriffinlim(spec: torch.Tensor, n_iter: int = 30) -> torch.Tensor:
    """Returns estimated audio signal for a given magnitude spectrogram

    Assumes `spec` generated using global config in `audio`. The same
    parameters are used for inversion.

    Args:
        spec (torch.Tensor): spectrogram in the same format as from
            `torch.stft()`.
        n_iter (int, optional): number of iterations for deepgriffinlim
            phase estimation. Defaults to 30.

    Returns:
        torch.Tensor: tensor with dimensions [channels, time], where
            each element in the first dimension is a channel, and each
            element in that element (second dimension) is an amplitude
            value at a point in time.

    """
    model = DNN()
    model.load_state_dict(torch.load("spectrograminversion/degli_dnn_state.pt"))
    model.eval()

    # create initial estimate
    est_complex_spectrogram = audio.audio_to_spectrogram(
        griffinlim(spec, 1), discard_phase=False
    )

    for _ in range(n_iter):
        # first GLA-inspired layer in DeGLI sub block (P_A)
        amplitude_replaced_spec = est_complex_spectrogram * (
            spec / torch.abs(est_complex_spectrogram)
        )

        # second GLA-inspired layer in DeGLI sub block (P_C)
        griffinlim_1_iter = audio.audio_to_spectrogram(
            audio.complex_spectrogram_to_audio(amplitude_replaced_spec),
            discard_phase=False,
        )

        # merge input spectrograms into multichannel spectrogram, treat phase as separate channels
        input_tensor = spectrogram.separate_phase_to_channel(
            spectrogram.merge_to_multichannel(
                est_complex_spectrogram, amplitude_replaced_spec, griffinlim_1_iter
            )
        )

        # estimate residual with DNN model, subtract from griffinlim estimate
        residual = model(torch.unsqueeze(input_tensor, 0))
        residual = spectrogram.merge_separate_phase_channel(residual[0])
        est_complex_spectrogram = griffinlim_1_iter - residual

    return audio.complex_spectrogram_to_audio(est_complex_spectrogram)
