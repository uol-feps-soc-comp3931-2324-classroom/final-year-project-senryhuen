import argparse

from utils import audio, spectrogram, evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "spectrogram_filepath",
        type=str,
        help="filepath to spectrogram image of original audio",
    )
    parser.add_argument(
        "audio_filepath",
        type=str,
        help="filepath to reconstructed audio file",
    )
    parser.add_argument(
        "-rmse",
        "--rmse",
        action="store_false",
        help="show root-mean-squared-error between spectrogram of original audio and spectrogram of audio reconstruction (default: True)",
    )
    parser.add_argument(
        "-snr",
        "--snr",
        action="store_false",
        help="show signal-to-noise ratio between spectrogram of original audio and spectrogram of audio reconstruction (default: True)",
    )
    args = parser.parse_args()

    spec = spectrogram.load_spectrogram(args.spectrogram_filepath)
    recon_audio_tensor, _ = audio.load_audio(args.audio_filepath)
    other_spec = audio.audio_to_spectrogram(recon_audio_tensor)

    if spec.shape != other_spec.shape:
        raise ValueError(
            f"{args.spectrogram_filepath} and {args.audio_filepath} cannot be compared because they represent different audio lengths"
        )

    # calculate and print rmse/snr depending on args
    if args.rmse:
        print(f"RMSE: {evaluation.rmse(spec, other_spec)}")
    if args.snr:
        print(f"SNR: {evaluation.snr(spec, other_spec)}")


if __name__ == "__main__":
    main()
