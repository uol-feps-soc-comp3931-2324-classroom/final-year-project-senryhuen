import argparse

from utils import audio, spectrogram


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_audio_filepath",
        type=str,
        help="filepath to audio file to generate spectrogram for",
    )
    parser.add_argument(
        "output_spectrogram_filepath",
        type=str,
        help="filepath to save output spectrogram image file at",
    )
    args = parser.parse_args()

    audio_tensor, _ = audio.load_audio(args.input_audio_filepath, multichannel="merge")
    spec = audio.audio_to_spectrogram(audio_tensor)

    # currently can only save as .tiff image
    spectrogram.save_spectrogram_tiff(spec, args.output_spectrogram_filepath)


if __name__ == "__main__":
    main()
