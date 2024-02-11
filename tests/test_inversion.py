from tests import *


TESTDATA_PATH = "tests/testdata_for_inversion"


class TestInversionModule(unittest.TestCase):
    def test_griffinlim(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="first")
        spec = audio.audio_to_spectrogram(audio_tensor)
        reconstructed_audio_tensor = inversion.griffinlim(spec)

        # test reconstruct has correct shape
        self.assertEqual(len(audio_tensor.shape), 2)
        self.assertEqual(len(reconstructed_audio_tensor.shape), 2)
        self.assertEqual(
            list(audio_tensor.shape)[0], list(reconstructed_audio_tensor.shape)[0]
        )

        # reconstruction length can differ by at most audio.STFT_CONFIG["hop_length"] samples
        # due to the audio not dividing perfectly into windows
        self.assertTrue(
            abs(list(audio_tensor.shape)[1] - list(reconstructed_audio_tensor.shape)[1])
            < audio.STFT_CONFIG["hop_length"]
        )
