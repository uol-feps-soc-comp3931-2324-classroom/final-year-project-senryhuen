import argparse

from utils import audio, spectrogram, evaluation
from spectrograminversion import griffinlim


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_spectrogram_filepath",
        type=str,
        help="filepath to spectrogram image to reconstruct audio for",
    )
    parser.add_argument(
        "output_audio_filepath",
        type=str,
        help="filepath to save output audio file at",
    )
    parser.add_argument(
        "sample_rate",
        type=int,
        help="sample rate of spectrogram/audio",
    )
    parser.add_argument(
        "-m",
        "--method",
        type=str,
        default="griffinlim",
        help="method of audio reconstruction to use, available options: 'griffinlim', 'TBC' (default: 'griffinlim')",
    )
    parser.add_argument(
        "-it",
        "--iterations",
        type=int,
        default=100,
        help="number of iterations of audio reconstruction method if applicable (default: 100)",
    )
    parser.add_argument(
        "-rmse",
        "--rmse",
        action="store_true",
        help="show root-mean-squared-error between input spectrogram and spectrogram of audio reconstruction (default: False)",
    )
    parser.add_argument(
        "-snr",
        "--snr",
        action="store_true",
        help="show signal-to-noise ratio between input spectrogram and spectrogram of audio reconstruction (default: False)",
    )
    args = parser.parse_args()

    spec = spectrogram.load_spectrogram_tiff(args.input_spectrogram_filepath)

    # audio reconstruction using method chosen from args
    if args.method == "griffinlim":
        audio_tensor = griffinlim(spec, args.iterations)
    else:
        audio_tensor = griffinlim(spec, args.iterations)

    # currently only supports expected .tiff type image format
    audio.save_audio(args.output_audio_filepath, audio_tensor, args.sample_rate)

    # calculate and print rmse/snr depending on args
    if args.rmse or args.snr:
        recon_spec = audio.audio_to_spectrogram(audio_tensor)
        if args.rmse:
            print(f"RMSE: {evaluation.rmse(spec, recon_spec)}")
        if args.snr:
            print(f"SNR: {evaluation.snr(spec, recon_spec)}")


if __name__ == "__main__":
    main()
