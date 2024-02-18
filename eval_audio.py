import argparse

from utils import audio, evaluation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "audio_filepath",
        type=str,
        help="filepath to original audio file",
    )
    parser.add_argument(
        "other_audio_filepath",
        type=str,
        help="filepath to reconstructed audio file to compare with the original audio",
    )
    parser.add_argument(
        "-rmse",
        "--rmse",
        action="store_false",
        help="show root-mean-squared-error between original audio and audio reconstruction (default: True)",
    )
    parser.add_argument(
        "-snr",
        "--snr",
        action="store_false",
        help="show signal-to-noise ratio between original audio and audio reconstruction (default: True)",
    )
    args = parser.parse_args()

    audio_tensor, sample_rate = audio.load_audio(args.audio_filepath, "merge")
    other_audio_tensor, other_sample_rate = audio.load_audio(
        args.other_audio_filepath, "merge"
    )

    if sample_rate != other_sample_rate:
        raise ValueError(
            f"{args.audio_filepath} and {args.other_audio_filepath} cannot be compared because they have different sample rates"
        )

    if (
        abs(audio_tensor.shape[1] - other_audio_tensor.shape[1])
        >= audio.STFT_CONFIG["hop_length"]
    ):
        raise ValueError(
            f"{args.audio_filepath} and {args.other_audio_filepath} cannot be compared because they are different lengths"
        )

    audio_tensor, other_audio_tensor = evaluation.make_equal_length(
        audio_tensor, other_audio_tensor
    )

    # calculate and print rmse/snr depending on args
    if args.rmse:
        print(f"RMSE: {evaluation.rmse(audio_tensor, other_audio_tensor)}")
    if args.snr:
        print(f"SNR: {evaluation.snr(audio_tensor, other_audio_tensor)}")


if __name__ == "__main__":
    main()
