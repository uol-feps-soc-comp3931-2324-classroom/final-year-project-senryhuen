"""
This script will run tests on audio files in "data/eval_audio" and the generate
results are saved to "evaluationdata/evaluation.csv", with graphs to present
interpretations of the data.

The generated results include the SNR, RMSE and execution time of both the
Griffin-Lim and Deep Griffin-Lim methods for various number of iterations with
each of TIFF, PNG and JPG image formats for each test audio file.

The results also include the SNR, RMSE and file size of WAV, FLAC and MP3 audio
compression on each test audio file.

"""

import os
import signal
import time
import shutil
import pandas as pd
import matplotlib.pyplot as plt

from utils import audio, spectrogram, evaluation
from spectrograminversion import griffinlim, deepgriffinlim
from preprocess_dataset import valid_audio_extensions


def raise_timeout(signalnum, stackframe):
    raise TimeoutError


signal.signal(signal.SIGALRM, raise_timeout)


EVAL_DATA_DIR = "data/eval_audio"
TMP_DIR = "tempevaldata"
TMP_JPG_PATH = "tempevaldata/tempspec.jpg"
TMP_PNG_PATH = "tempevaldata/tempspec.png"
TMP_TIFF_PATH = "tempevaldata/tempspec.tiff"
TMP_MP3_PATH = "tempevaldata/tempaudio.mp3"
TMP_FLAC_PATH = "tempevaldata/tempaudio.flac"
TMP_WAV_PATH = "tempevaldata/tempaudio.wav"


# create temp directory for files generated by evaluation script
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)

os.makedirs(TMP_DIR)

if not os.path.exists("evaluationdata"):
    os.makedirs("evaluationdata")

# create pandas dataframe
img_formats = ["jpg", "png", "tiff"]
method_options = ["gla", "degli"]
num_its = [10, 100, 200, 300, 400]
audio_formats = ["mp3", "flac", "wav"]

columns = ["audio_filepath", "is_speech"]
for img_format in img_formats:
    columns.append(f"{img_format}-filesize")

    for method in method_options:
        for num_it in num_its:
            columns.append(f"{img_format}-{method}-{num_it}it-snrSpec")
            columns.append(f"{img_format}-{method}-{num_it}it-rmseSpec")
            columns.append(f"{img_format}-{method}-{num_it}it-time")

for audio_format in audio_formats:
    columns.append(f"{audio_format}-filesize")
    columns.append(f"{audio_format}-snrSpec")
    columns.append(f"{audio_format}-rmseSpec")

df = pd.DataFrame(columns=columns)

eval_count = 0
speech_count = 0
music_count = 0

for root, dirs, files in os.walk(EVAL_DATA_DIR):
    for file in files:
        # check is audio file
        file_splitext = os.path.splitext(file)
        if file_splitext[1] not in valid_audio_extensions:
            continue

        # check if clip is speech or music - expects files to be in "speech" or "music" dirs
        if "/speech" in root:
            is_speech = True
            if speech_count >= 50:
                continue
        elif "/music" in root:
            is_speech = False
            if music_count >= 50:
                continue
        else:
            continue

        # load audio, only use first N samples
        audio_tensor, sr = audio.load_audio(
            os.path.join(root, file), multichannel="merge"
        )
        clips = audio.split_audio_fixed(audio_tensor)

        if len(clips) <= 0:
            continue

        clip = clips[0]

        new_row = {
            "audio_filepath": [os.path.join(root, file)],
            "is_speech": [is_speech],
        }

        orig_spec = audio.audio_to_spectrogram(clip)

        # save audio as each audio format
        audio.save_audio(TMP_MP3_PATH, clip, sr)
        audio.save_audio(TMP_FLAC_PATH, clip, sr)
        audio.save_audio(TMP_WAV_PATH, clip, sr)

        # save spectrogram as each image format
        spectrogram.save_spectrogram(orig_spec, TMP_JPG_PATH)
        spectrogram.save_spectrogram(orig_spec, TMP_PNG_PATH)
        spectrogram.save_spectrogram(orig_spec, TMP_TIFF_PATH)

        for spec_path in [TMP_JPG_PATH, TMP_PNG_PATH, TMP_TIFF_PATH]:
            spec = spectrogram.load_spectrogram(spec_path)
            img_format = os.path.splitext(spec_path)[1][1:]
            filesize = os.path.getsize(spec_path)
            new_row[f"{img_format}-filesize"] = [filesize]

            # flag for when timed out on method/iteration, can skip further iterations
            skip_gla = False
            skip_degli = False

            for n_iter in [10, 100, 200, 300, 400]:
                if skip_gla:
                    recon_audio_gla = None
                else:
                    signal.alarm(15)
                    try:
                        start = time.time()
                        recon_audio_gla = griffinlim(spec, n_iter=n_iter)
                        end = time.time()
                        exec_time = end - start
                    except TimeoutError:
                        recon_audio_gla = None
                        exec_time = None
                        skip_gla = True
                    finally:
                        signal.alarm(0)

                if recon_audio_gla is not None:
                    recon_spec = audio.audio_to_spectrogram(recon_audio_gla)
                    tmp_rmse = evaluation.rmse(orig_spec, recon_spec)
                    tmp_snr = evaluation.snr(orig_spec, recon_spec)
                    new_row[f"{img_format}-gla-{n_iter}it-rmseSpec"] = [tmp_rmse]
                    new_row[f"{img_format}-gla-{n_iter}it-snrSpec"] = [tmp_snr]
                    new_row[f"{img_format}-gla-{n_iter}it-time"] = [exec_time]

                if skip_degli:
                    recon_audio_degli = None
                else:
                    signal.alarm(15)
                    try:
                        start = time.time()
                        recon_audio_degli = deepgriffinlim(spec, n_iter=n_iter)
                        end = time.time()
                        exec_time = end - start
                    except TimeoutError:
                        recon_audio_degli = None
                        exec_time = None
                        skip_degli = True
                    finally:
                        signal.alarm(0)

                if recon_audio_degli is not None:
                    recon_spec = audio.audio_to_spectrogram(recon_audio_degli)
                    tmp_rmse = evaluation.rmse(orig_spec, recon_spec)
                    tmp_snr = evaluation.snr(orig_spec, recon_spec)
                    new_row[f"{img_format}-degli-{n_iter}it-rmseSpec"] = [tmp_rmse]
                    new_row[f"{img_format}-degli-{n_iter}it-snrSpec"] = [tmp_snr]
                    new_row[f"{img_format}-degli-{n_iter}it-time"] = [exec_time]

        for audio_path in [TMP_MP3_PATH, TMP_FLAC_PATH, TMP_WAV_PATH]:
            aud, sr = audio.load_audio(audio_path)
            audio_format = os.path.splitext(audio_path)[1][1:]
            filesize = os.path.getsize(audio_path)
            new_row[f"{audio_format}-filesize"] = [filesize]

            aud, aud2 = evaluation.make_equal_length(aud, clip)
            spec = audio.audio_to_spectrogram(aud)
            spec2 = audio.audio_to_spectrogram(aud2)
            tmp_rmse = evaluation.rmse(spec, spec2)
            tmp_snr = evaluation.snr(spec, spec2)
            new_row[f"{audio_format}-rmseSpec"] = [tmp_rmse]
            new_row[f"{audio_format}-snrSpec"] = [tmp_snr]

        # add new row to df
        df = pd.concat([df, pd.DataFrame(new_row)])

        # continually save to csv
        df.to_csv("evaluationdata/evaluation.csv")

        eval_count += 1
        if is_speech:
            speech_count += 1
        else:
            music_count += 1

        print(f"total clips evaluated: {eval_count}")
        print(f"speech clips evaluated: {speech_count}")
        print(f"music clips evaluated: {music_count}\n")


# delete created folder and contents
if os.path.exists(TMP_DIR):
    shutil.rmtree(TMP_DIR)


df = pd.read_csv("evaluationdata/evaluation.csv")
img_format_colours = [
    ("magenta", "darkmagenta"),
    ("cyan", "darkcyan"),
    ("darkseagreen", "seagreen"),
]
metrics = ["snr", "rmse"]
metrics_better = ["higher", "lower"]


# graph: x = num its, y = SNR/RMSE, per each image format
for metric, better in zip(metrics, metrics_better):
    fig, ax = plt.subplots()

    for img_format, colour in zip(img_formats, img_format_colours):
        y_gla = []
        y_degli = []

        for point in num_its:
            gla_avg = df[f"{img_format}-gla-{point}it-{metric}Spec"].mean()
            degli_avg = df[f"{img_format}-degli-{point}it-{metric}Spec"].mean()

            y_gla.append(gla_avg)

            empty_cells = df[f"{img_format}-degli-{point}it-{metric}Spec"].isna().sum()
            num_cells = len(df[f"{img_format}-degli-{point}it-{metric}Spec"])
            if empty_cells / num_cells > 0.3:
                y_degli.append(None)
            else:
                y_degli.append(degli_avg)

        ax.plot(
            num_its,
            y_gla,
            marker="o",
            color=colour[0],
            linestyle="-",
            label=f"Griffin-Lim ({img_format.upper()})",
        )
        ax.plot(
            num_its,
            y_degli,
            marker="s",
            color=colour[1],
            linestyle="-",
            label=f"Deep Griffin-Lim ({img_format.upper()})",
        )

    ax.set_xlabel("No. Iterations")
    ax.set_ylabel(f"Average {metric.upper()} ({better} is better)")
    ax.set_title(f"Average {metric.upper()} per Method per No. Iterations")
    ax.legend()
    fig.savefig(f"evaluationdata/{metric}-perIts.png")
    fig.clear()


# graph: x = time, y = SNR/RMSE, per each image format
for metric, better in zip(metrics, metrics_better):
    fig, ax = plt.subplots()

    for img_format, colour in zip(img_formats, img_format_colours):
        x_gla = []
        y_gla = []
        x_degli = []
        y_degli = []

        for point in num_its:
            gla_avg = df[f"{img_format}-gla-{point}it-{metric}Spec"].mean()
            degli_avg = df[f"{img_format}-degli-{point}it-{metric}Spec"].mean()
            gla_time_avg = df[f"{img_format}-gla-{point}it-time"].mean()
            degli_time_avg = df[f"{img_format}-degli-{point}it-time"].mean()

            x_gla.append(gla_time_avg)
            y_gla.append(gla_avg)

            empty_cells = df[f"{img_format}-degli-{point}it-{metric}Spec"].isna().sum()
            num_cells = len(df[f"{img_format}-degli-{point}it-{metric}Spec"])
            if empty_cells / num_cells <= 0.3:
                x_degli.append(degli_time_avg)
                y_degli.append(degli_avg)
            else:
                x_degli.append(None)
                y_degli.append(None)

        ax.plot(
            x_gla,
            y_gla,
            marker="o",
            color=colour[0],
            linestyle="-",
            label=f"Griffin-Lim ({img_format.upper()})",
        )
        ax.plot(
            x_degli,
            y_degli,
            marker="s",
            color=colour[1],
            linestyle="-",
            label=f"Deep Griffin-Lim ({img_format.upper()})",
        )

    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel(f"Average {metric.upper()} ({better} is better)")
    ax.set_title(f"Average {metric.upper()} per Method per Time")
    ax.legend()
    fig.savefig(f"evaluationdata/{metric}-perTime.png")
    fig.clear()


# csv: average time per iteration per method
avg_times = {}

gla_avg_times = []
degli_avg_times = []
for img_format, colour in zip(img_formats, img_format_colours):
    gla_avg_times.append(df[f"{img_format}-gla-300it-time"].mean() / 300)
    degli_avg_times.append(df[f"{img_format}-degli-300it-time"].mean() / 300)

avg_times["gla-avg-time-per-it"] = [sum(gla_avg_times) / len(gla_avg_times)]
avg_times["degli-avg-time-per-it"] = [sum(degli_avg_times) / len(degli_avg_times)]

pd.DataFrame(avg_times).to_csv("evaluationdata/avgTimePerIt.csv")


# graph: x = file size, y = SNR/RMSE
for metric, better in zip(metrics, metrics_better):
    fig, ax = plt.subplots()

    for img_format, marker in zip(img_formats, ["o", "s", "^"]):
        filesize_avg = [df[f"{img_format}-filesize"].mean() / 1024]
        gla_avg = df[f"{img_format}-gla-300it-{metric}Spec"].mean()
        degli_avg = df[f"{img_format}-degli-300it-{metric}Spec"].mean()
        spec_avg = [(gla_avg + degli_avg) / 2]

        ax.plot(
            filesize_avg,
            spec_avg,
            marker=marker,
            color=f"darkcyan",
            linestyle="",
            label=f"Spectrogram - {img_format.upper()} (300 its)",
        )

    for audio_format, marker in zip(audio_formats, ["o", "s", "^"]):
        filesize_avg = [df[f"{audio_format}-filesize"].mean() / 1024]
        aud_avg = [df[f"{audio_format}-{metric}Spec"].mean()]

        ax.plot(
            filesize_avg,
            aud_avg,
            marker=marker,
            color="darkmagenta",
            linestyle="",
            label=f"Audio Format - {audio_format.upper()}",
        )

    ax.set_xlabel("Filesize (KiBs)")
    ax.set_ylabel(f"Average {metric.upper()} ({better} is better)")
    ax.set_title(f"Average {metric.upper()} per Method per Filesize")
    ax.legend()
    fig.savefig(f"evaluationdata/{metric}-perFilesize.png")
    fig.clear()
