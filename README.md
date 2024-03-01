[![Open in Codespaces](https://classroom.github.com/assets/launch-codespace-7f7980b617ed060a017424585567c406b6ee15c891e84e1186181d67ecf80aa0.svg)](https://classroom.github.com/open-in-codespaces?assignment_repo_id=12928530)


# CLI commands (use '--help' flag for details about usage)
1. audio2spec.py - convert audio to a spectrogram image (.tiff)
2. spec2audio.py - reconstruct audio from a spectrogram image (.tiff as generated by audio2spec)
3. eval_audio.py - evaluate reconstructed audio against the original audio
4. eval_spec.py - evaluate the spectrogram of the reconstructed audio against the original spectrogram that was reconstructed


# Other scripts
1. preprocess_dataset.py - processes audio files in preparation for use in a dataset (splits into fixed length clips and sorts by type (speech or music) and sample rate, generates attributes csv)
