from tests import *


TESTDATA_PATH = "tests/testdata_for_audio"


class TestAudioModule(unittest.TestCase):
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

    def test_load_flac_keep_multichannel(self):
        audio_tensor, sample_rate = audio.load_audio(TEST_FLAC, multichannel="keep")

        self.assertEqual(sample_rate, 44100)
        self.assertEqual(list(audio_tensor.shape), [2, 5 * 44100])

    def test_load_flac_merge_multichannel(self):
        audio_tensor, sample_rate = audio.load_audio(TEST_FLAC, multichannel="merge")

        self.assertEqual(sample_rate, 44100)
        self.assertEqual(list(audio_tensor.shape), [1, 5 * 44100])

    def test_load_flac_keep_first_channel(self):
        audio_tensor, sample_rate = audio.load_audio(TEST_FLAC, multichannel="first")

        self.assertEqual(sample_rate, 44100)
        self.assertEqual(list(audio_tensor.shape), [1, 5 * 44100])

    def test_load_wav(self):
        audio_tensor, sample_rate = audio.load_audio(TEST_WAV, multichannel="keep")

        self.assertEqual(sample_rate, 16000)
        self.assertEqual(list(audio_tensor.shape), [1, 5 * 16000])

    def test_save_as_flac(self):
        audio_tensor, sample_rate = audio.load_audio(TEST_WAV, multichannel="keep")

        test_save_path = f"{TESTDATA_PATH}/test_save.flac"
        audio.save_audio(test_save_path, audio_tensor, sample_rate)

        audio_tensor_flac, sample_rate_flac = audio.load_audio(
            test_save_path, multichannel="keep"
        )
        self.assertEqual(sample_rate, sample_rate_flac)
        self.assertEqual(audio_tensor.shape, audio_tensor_flac.shape)

    def test_save_as_wav(self):
        audio_tensor, sample_rate = audio.load_audio(TEST_FLAC, multichannel="keep")

        test_save_path = f"{TESTDATA_PATH}/test_save.wav"
        audio.save_audio(test_save_path, audio_tensor, sample_rate)

        audio_tensor_wav, sample_rate_wav = audio.load_audio(
            test_save_path, multichannel="keep"
        )
        self.assertEqual(sample_rate, sample_rate_wav)
        self.assertTrue(torch.equal(audio_tensor, audio_tensor_wav))

    def test_audio_to_spectrogram(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="keep")
        spec = audio.audio_to_spectrogram(audio_tensor)
        self.assertEqual(list(spec.shape), [2, 1025, 431])

    def test_multichannel_audio_to_mono(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="keep")
        self.assertEqual(audio_tensor.shape[0], 2)

        audio_mono = audio.audio_to_mono(audio_tensor)
        self.assertEqual(audio_mono.shape[0], 1)

    def test_mono_audio_to_mono(self):
        audio_tensor, _ = audio.load_audio(TEST_WAV, multichannel="keep")
        self.assertEqual(audio_tensor.shape[0], 1)

        audio_mono = audio.audio_to_mono(audio_tensor)
        self.assertEqual(audio_mono.shape[0], 1)

    def test_merge_multichannel_with_multichannel(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="keep")
        audio_tensor_2, _ = audio.load_audio(TEST_FLAC, multichannel="keep")

        self.assertEqual(audio_tensor.shape[0], 2)
        self.assertEqual(audio_tensor_2.shape[0], 2)

        audio_tensor_merged = audio.merge_to_multichannel(audio_tensor, audio_tensor_2)
        self.assertEqual(audio_tensor_merged.shape[0], 4)

    def test_merge_mono_with_multichannel(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="keep")
        audio_tensor_2, _ = audio.load_audio(TEST_FLAC, multichannel="first")

        self.assertEqual(audio_tensor.shape[0], 2)
        self.assertEqual(audio_tensor_2.shape[0], 1)

        audio_tensor_merged = audio.merge_to_multichannel(audio_tensor, audio_tensor_2)
        self.assertEqual(audio_tensor_merged.shape[0], 3)

    def test_merge_mono_to_mono(self):
        audio_tensor, _ = audio.load_audio(TEST_WAV, multichannel="keep")
        audio_tensor_2, _ = audio.load_audio(TEST_WAV, multichannel="keep")

        self.assertEqual(audio_tensor.shape[0], 1)
        self.assertEqual(audio_tensor_2.shape[0], 1)

        audio_tensor_merged = audio.merge_to_multichannel(audio_tensor, audio_tensor_2)
        self.assertEqual(audio_tensor_merged.shape[0], 2)
