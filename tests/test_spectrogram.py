from tests import *


TESTDATA_PATH = "tests/testdata_for_spectrogram"


class TestSpectrogramModule(unittest.TestCase):
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

    @classmethod
    def tearDownClass(cls) -> None:
        # delete created folder and contents
        if os.path.exists(TESTDATA_PATH):
            shutil.rmtree(TESTDATA_PATH)

    def test_save_spectrogram_tiff(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="first")
        spec = audio.audio_to_spectrogram(audio_tensor)

        spec_save_path = f"{TESTDATA_PATH}/test_spec.tiff"
        spectrogram.save_spectrogram_tiff(spec, spec_save_path)

        self.assertTrue(os.path.exists(spec_save_path))
        loaded_spec = spectrogram.load_spectrogram_tiff(spec_save_path)

        self.assertEqual(list(spec.shape), list(loaded_spec.shape))
        self.assertTrue(torch.all(torch.isclose(torch.abs(spec), loaded_spec, atol=2e-2)))


    def test_save_spectrogram_tiff_invalid_save_path(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="first")
        spec = audio.audio_to_spectrogram(audio_tensor)

        spec_save_path = f"{TESTDATA_PATH}/test_spec.invalid"
        self.assertRaises(ValueError, spectrogram.save_spectrogram_tiff, spec, spec_save_path)
        self.assertFalse(os.path.exists(spec_save_path))

    def test_save_spectrogram_tiff_for_multichannel(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="keep")
        spec = audio.audio_to_spectrogram(audio_tensor)

        spec_save_path = f"{TESTDATA_PATH}/test_spec_multi.tiff"
        self.assertRaises(ValueError, spectrogram.save_spectrogram_tiff, spec, spec_save_path)
        self.assertFalse(os.path.exists(spec_save_path))
