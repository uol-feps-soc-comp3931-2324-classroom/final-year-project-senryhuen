from tests import *


TESTDATA_PATH = "tests/testdata_for_evaluation"


class TestEvaluationModule(unittest.TestCase):
    def test_rmse(self):
        spec_1 = torch.tensor([[[1., 2, 3], [4, 5, 6]]])
        spec_2 = torch.tensor([[[1., 2, 4], [4, 5, 6]]])
        self.assertEqual(evaluation.rmse(spec_1, spec_2), np.sqrt(1/6))

    def test_snr(self):
        spec_1 = torch.tensor([[[1., 2, 3], [4, 5, 6]]])
        spec_2 = torch.tensor([[[1., 2, 4], [4, 5, 6]]])
        self.assertAlmostEqual(evaluation.snr(spec_1, spec_2), 19.6, 1)
