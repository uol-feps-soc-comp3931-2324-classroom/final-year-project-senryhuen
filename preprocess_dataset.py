import os
import torch

import audio


valid_audio_extensions = [".mp3", ".wav", ".flac"]
last_idx = {
    "speech-16000": -1,
    "music-44100": -1,
}


def split_audio(
    audio_tensor: torch.Tensor, sample_rate: int, segment_length: int = 5
) -> list:
    """Split `audio_tensor` into audio tensors of `segment_length`

    If `audio_tensor` does not divide by `segment_length` perfectly,
    the last segment is discarded.

    Args:
        audio_tensor (torch.Tensor): tensor with dimensions
            [channels, time], where each element in the first dimension
            is a channel, and each element in that element (second
            dimension) is an amplitude value at a point in time.
        sample_rate (int): sample rate of `audio_tensor`.
        segment_length (int, optional): desired length of audio
            segments in seconds. Defaults to 5.

    Returns:
        list: list of audio tensors, each being a segment from
            `audio_tensor`.

    """
    segments = []

    if segment_length <= 0 or sample_rate <= 0:
        return []

    increment = segment_length * sample_rate
    start = 0
    end = increment

    while end <= audio_tensor.shape[-1]:
        segments.append(audio_tensor[:, start:end])
        start += increment
        end += increment

    return segments


def check_idx_consecutive(dirpath: str) -> tuple[bool, int]:
    """Check save directory for processed audio is valid

    Existing files should be have consecutive filenames padded to 6
    digits, starting from "000000".

    Args:
        dirpath (str): path to folder where processed audio will be
            saved.

    Returns:
        tuple[bool, int]: [0] `True` if save location is valid, `False`
            otherwise. [1] highest consecutive number for filename in
            `dirpath`.

    """
    files = sorted(os.listdir(dirpath))
    for idx, file in enumerate(files):
        if os.path.splitext(file)[0] != f"{idx:06}":
            return False, idx - 1

    return True, len(files) - 1


def get_last_idx(idx_type: str, sample_rate: int):
    """Add key to last_idx with updated value

    Value for key in last_idx would be the highest consecutive number
    of a filename at the save location, given that the save location is
    valid (all filenames are consecutive and padded to 6 digits,
    starting from "000000").

    Args:
        idx_type (str): type of processed audio clip is either "speech"
            or "music".
        sample_rate (int): sample rate of processed audio clip.

    Raises:
        ValueError: invalid save location "data/audio/`idx_type`/
            samplerate-`sample_rate`", filenames must be consecutive
            and padded to 6 digits and start from "000000".

    """
    idx = f"{idx_type}-{sample_rate}"
    idx_filepath = f"data/audio/{idx_type}/samplerate-{sample_rate}"

    if os.path.exists(idx_filepath):
        is_consecutive, highest_idx = check_idx_consecutive(idx_filepath)
        if not is_consecutive:
            raise ValueError(
                f"Invalid save location, non-consecutive filenames: '{idx_filepath}'"
            )

        last_idx[idx] = highest_idx
    else:
        last_idx[idx] = -1


def get_save_location(sample_rate: int, extension: str, is_speech: bool = False) -> str:
    """Get relative file path to save new processed audio clip

    Args:
        sample_rate (int): sample rate of processed audio clip.
        extension (str): file extension to save as, e.g. ".mp3".
        is_speech (bool, optional): if `True`, saves under speech
            category, otherwise saves under music category. Defaults to
            `False`.

    Returns:
        str: relative file path to save new processed audio clip,
            always in the format "data/audio/{music || speech}/
            samplerate-{sample_rate}/{incrementing index:06}
            {extension}".

    """
    if is_speech:
        idx_type = "speech"
    else:
        idx_type = "music"

    idx = f"{idx_type}-{sample_rate}"
    if idx not in last_idx:
        get_last_idx(idx_type, sample_rate)

    last_idx[idx] += 1
    return (
        f"data/audio/{idx_type}/samplerate-{sample_rate}/{last_idx[idx]:06}{extension}"
    )


def get_spectrogram_save_location(audio_save_location: str, extension: str) -> str:
    """Get relative file path to save spectrogram image

    Spectrogram corresponds to a new processed audio clip.

    Args:
        audio_save_location (str): relative file path of the
            corresponding new processed audio clip.
        extension (str): file extension to save as, e.g. ".png".

    Returns:
        str: relative file path to save new spectrogram image, which
            will have the same filename as the `audio_save_location'
            but with a different file extension and under a different
            root folder.

    """
    return (
        os.path.splitext(
            os.path.normpath(audio_save_location).replace(
                "data/audio", "data/spectrograms"
            )
        )[0]
        + extension
    )


def _preprocess_audio(
    src_path: str = "data/orig_audio",
    src_limit: int = None,
    segment_limit: int = None,
    spec_format: str = ".tiff",
) -> tuple[int, int]:
    """Process audio to be used in a dataset.

    Duplicates will not be detected if run on the same audio again.

    Splits audio into fixed length segments, creates corresponding
    spectrograms for each segment. Segments sorted by type (speech or
    music) and sample rate.

    Args:
        src_path (str, optional): path to directory containing audio
            files to be processed, includes files in subdirectories.
            Defaults to "data/orig_audio".
        src_limit (int, optional): maximum number of audio files to
            process. Defaults to None.
        segment_limit (int, optional): maximum number of processed
            audio clips. Defaults to None.
        spec_format (str, optional): image format extension to save
            spectrograms as. Defaults to ".tiff".

    Returns:
        tuple[int, int]: [0] number of audio files processed, [1]
            number of processed audio clips produced.

    """
    src_count = 0
    segment_count = 0

    for root, _, files in os.walk(src_path):
        for file in files:
            file_extension = os.path.splitext(file)[1].lower()
            if file_extension in valid_audio_extensions:
                orig_audio, sr = audio.load_audio(os.path.join(root, file))
                clips = split_audio(orig_audio, sample_rate=sr)

                for clip in clips:
                    if "data/orig_audio/speech" in os.path.normpath(root):
                        save_path = get_save_location(sr, file_extension, True)
                    elif "data/orig_audio/music" in os.path.normpath(root):
                        save_path = get_save_location(sr, ".flac", False)

                    audio.save_audio(save_path, clip, sr)

                    # spec = audio.audio_to_spectrogram(audio_tensor)
                    # spec_save_path = get_spectrogram_save_location(save_path, spec_format)
                    ## TBC: save spectrogram image

                src_count += 1
                segment_count += len(clips)

                if (src_limit and src_count >= src_limit) or (
                    segment_limit and segment_count >= segment_limit
                ):
                    return src_count, segment_count

    return src_count, segment_count


if __name__ == "__main__":
    # print(torchaudio.list_audio_backends())
    # print(torchaudio.set_audio_backend("ffmpeg"))
    # print(torchaudio.utils.ffmpeg_utils.get_audio_decoders())
    # print(torchaudio.utils.sox_utils.list_read_formats())

    count, new_count = _preprocess_audio(src_path="data/orig_audio", src_limit=1)

    print(f"{count} audio files processed, split into {new_count} clips")