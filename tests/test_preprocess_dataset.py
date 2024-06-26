from tests import *


TESTDATA_PATH = "tests/testdata_for_preprocess"
CONSEC_PATH = f"{TESTDATA_PATH}/consec"
NONCONSEC_PATH = f"{TESTDATA_PATH}/nonconsec"


class TestPreprocessingAudio(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        # make sure TESTDATA_PATH does not exist
        if os.path.exists(TESTDATA_PATH):
            shutil.rmtree(TESTDATA_PATH)

        os.makedirs(TESTDATA_PATH)

        if not os.path.exists(TEST_FLAC):
            raise FileNotFoundError(f"Required test file not found: {TEST_FLAC}")
        if not os.path.exists(TEST_WAV):
            raise FileNotFoundError(f"Required test file not found: {TEST_WAV}")

        # make dir with files that have consecutive filenames
        os.makedirs(CONSEC_PATH)
        open(f"{CONSEC_PATH}/000000.txt", "w", encoding="utf8").close()
        open(f"{CONSEC_PATH}/000001.txt", "w", encoding="utf8").close()
        open(f"{CONSEC_PATH}/000002.txt", "w", encoding="utf8").close()

        # make dir with files that do not have consecutive filenames
        os.makedirs(NONCONSEC_PATH)
        open(f"{NONCONSEC_PATH}/000000.txt", "w", encoding="utf8").close()
        open(f"{NONCONSEC_PATH}/notanumber.txt", "w", encoding="utf8").close()
        open(f"{NONCONSEC_PATH}/100.txt", "w", encoding="utf8").close()

    @classmethod
    def tearDownClass(cls) -> None:
        # delete created folder and contents
        if os.path.exists(TESTDATA_PATH):
            shutil.rmtree(TESTDATA_PATH)

    def test_check_idx_consecutive_when_true(self):
        is_consec, last_idx = preprocess_dataset.check_idx_consecutive(CONSEC_PATH)
        self.assertTrue(is_consec)
        self.assertEqual(last_idx, 2)

    def test_check_idx_consecutive_when_false(self):
        is_consec, last_idx = preprocess_dataset.check_idx_consecutive(NONCONSEC_PATH)
        self.assertFalse(is_consec)
        self.assertEqual(last_idx, 0)

    def test_get_save_location_flac(self):
        save_location = preprocess_dataset.get_save_location(44100, ".flac")
        expected_pattern = r"data/audio/music/samplerate-44100/\d{6}.flac"
        self.assertRegex(save_location, expected_pattern)

    def test_get_save_location_wav(self):
        save_location = preprocess_dataset.get_save_location(
            16000, ".wav", is_speech=True
        )
        expected_pattern = r"data/audio/speech/samplerate-16000/\d{6}.wav"
        self.assertRegex(save_location, expected_pattern)

    def test_remove_hidden_files(self):
        files = ["0", "0.txt", ".DS_Store", "._0.txt", "a.py"]
        cleaned_files = ["0", "0.txt", "a.py"]
        self.assertEqual(cleaned_files, preprocess_dataset.remove_hidden_files(files))

    def test_remove_hidden_files_from_cleaned_list(self):
        cleaned_files = ["0", "0.txt", "a.py"]
        self.assertEqual(cleaned_files, preprocess_dataset.remove_hidden_files(cleaned_files))

    def test_remove_hidden_files_from_empty_list(self):
        files = []
        self.assertEqual([], preprocess_dataset.remove_hidden_files(files))

    def test_precompute_DNN_dataset_output_shape(self):
        audio_tensor, _ = audio.load_audio(TEST_WAV, multichannel="first")
        spec = audio.audio_to_spectrogram(audio_tensor)
        output = preprocess_dataset.precompute_DNN_dataset(audio_tensor)

        # check returns a tuple
        self.assertEqual(len(output), 2)

        # check first spectrogram has 6 channels, second has 2 channels
        self.assertEqual(output[0].shape[0], 6)
        self.assertEqual(output[1].shape[0], 2)

        # test spectrograms have the same shape with exception of first dimension (channels)
        self.assertEqual(output[0].shape[1:], spec.shape[1:])
        self.assertEqual(output[1].shape[1:], spec.shape[1:])
