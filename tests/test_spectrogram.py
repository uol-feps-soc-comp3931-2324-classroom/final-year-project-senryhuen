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
        self.assertTrue(
            torch.all(torch.isclose(torch.abs(spec), loaded_spec, atol=3e-2))
        )

    def test_save_spectrogram_tiff_invalid_save_path(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="first")
        spec = audio.audio_to_spectrogram(audio_tensor)

        spec_save_path = f"{TESTDATA_PATH}/test_spec.invalid"
        self.assertRaises(
            ValueError, spectrogram.save_spectrogram_tiff, spec, spec_save_path
        )
        self.assertFalse(os.path.exists(spec_save_path))

    def test_save_spectrogram_tiff_for_multichannel(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="keep")
        spec = audio.audio_to_spectrogram(audio_tensor)

        spec_save_path = f"{TESTDATA_PATH}/test_spec_multi.tiff"
        self.assertRaises(
            ValueError, spectrogram.save_spectrogram_tiff, spec, spec_save_path
        )
        self.assertFalse(os.path.exists(spec_save_path))

    def test_merge_multichannel_with_multichannel(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="keep")
        spec_1 = audio.audio_to_spectrogram(audio_tensor)
        spec_2 = audio.audio_to_spectrogram(audio_tensor)

        self.assertEqual(spec_1.shape[0], 2)
        self.assertEqual(spec_2.shape[0], 2)

        spec_merged = spectrogram.merge_to_multichannel(spec_1, spec_2)
        self.assertEqual(spec_merged.shape[0], 4)

    def test_merge_mono_with_multichannel(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="keep")
        audio_tensor_2, _ = audio.load_audio(TEST_FLAC, multichannel="first")
        spec_1 = audio.audio_to_spectrogram(audio_tensor)
        spec_2 = audio.audio_to_spectrogram(audio_tensor_2)

        self.assertEqual(spec_1.shape[0], 2)
        self.assertEqual(spec_2.shape[0], 1)

        spec_merged = spectrogram.merge_to_multichannel(spec_1, spec_2)
        self.assertEqual(spec_merged.shape[0], 3)

    def test_merge_mono_to_mono(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="first")
        spec_1 = audio.audio_to_spectrogram(audio_tensor)
        spec_2 = audio.audio_to_spectrogram(audio_tensor)

        self.assertEqual(spec_1.shape[0], 1)
        self.assertEqual(spec_2.shape[0], 1)

        spec_merged = spectrogram.merge_to_multichannel(spec_1, spec_2)
        self.assertEqual(spec_merged.shape[0], 2)

    def test_separate_phase_to_channel_output_correct_channels(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="first")
        spec = audio.audio_to_spectrogram(audio_tensor, discard_phase=False)
        spec_separated = spectrogram.separate_phase_to_channel(spec)

        # test shape is the same apart from channels dimension
        self.assertEqual(spec.shape[0] * 2, spec_separated.shape[0])
        self.assertEqual(spec.shape[1:], spec_separated.shape[1:])

        # test real and imaginary components are the same in the separated channels
        self.assertTrue(torch.all(torch.eq(torch.real(spec), spec_separated[0, :, :])))
        self.assertTrue(torch.all(torch.eq(torch.imag(spec), spec_separated[1, :, :])))

    def test_separate_phase_to_channel_with_multichannel_specs(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="keep")
        spec = audio.audio_to_spectrogram(audio_tensor, discard_phase=False)
        spec_separated = spectrogram.separate_phase_to_channel(spec)

        # test shape is the same apart from channels dimension
        self.assertEqual(spec.shape[0] * 2, spec_separated.shape[0])
        self.assertEqual(spec.shape[1:], spec_separated.shape[1:])

        # test real and imaginary components are the same in the separated channels
        self.assertTrue(torch.all(torch.eq(torch.real(spec), spec_separated[0:2, :, :])))
        self.assertTrue(torch.all(torch.eq(torch.imag(spec), spec_separated[2:, :, :])))
