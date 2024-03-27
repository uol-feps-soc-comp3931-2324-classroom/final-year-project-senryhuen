from tests import *


TESTDATA_PATH = "tests/testdata_for_evaluation"


class TestEvaluationModule(unittest.TestCase):
    def test_rmse(self):
        spec_1 = torch.tensor([[[1., 2, 3], [4, 5, 6]]])
        spec_2 = torch.tensor([[[1., 2, 4], [4, 5, 6]]])
        self.assertEqual(evaluation.rmse(spec_1, spec_2, False), np.sqrt(1/6))

    def test_snr(self):
        spec_1 = torch.tensor([[[1., 2, 3], [4, 5, 6]]])
        spec_2 = torch.tensor([[[1., 2, 4], [4, 5, 6]]])
        self.assertAlmostEqual(evaluation.snr(spec_1, spec_2, False), 19.6, 1)

    def test_make_equal_length(self):
        audio_tensor, _ = audio.load_audio(TEST_FLAC, multichannel="keep")
        audio_tensor_2, _ = audio.load_audio(TEST_FLAC, multichannel="keep")

        audio_tensor_2 = audio_tensor_2[:, :100000]

        self.assertNotEqual(audio_tensor.shape[1], 100000)
        self.assertEqual(audio_tensor_2.shape[1], 100000)

        audio_tensor, audio_tensor_2 = evaluation.make_equal_length(audio_tensor, audio_tensor_2)

        self.assertEqual(audio_tensor.shape[1], 100000)
        self.assertEqual(audio_tensor_2.shape[1], 100000)

    def test_make_equal_scale_rejects_non_float32(self):
        spec_1 = torch.tensor([[[1, 2, 3], [4, 5, 6]]])
        self.assertFalse(spec_1.dtype == torch.float32)
        self.assertRaises(ValueError, evaluation.make_equal_scale, spec_1)

    def test_make_equal_scale(self):
        spec_1 = torch.tensor([[[1., 2, 3], [4, 5, 6]]])
        self.assertTrue(spec_1.dtype == torch.float32)

        spec_scaled = evaluation.make_equal_scale(spec_1)

        self.assertEqual(spec_scaled.min(), 0)
        self.assertEqual(spec_scaled.max(), 1)
