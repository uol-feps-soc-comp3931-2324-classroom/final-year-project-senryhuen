import unittest
import os
import shutil
import torch

import audio
import preprocess_dataset


TEST_FLAC = "tests/000000.flac"
TEST_WAV = "tests/000000.wav"
TESTDATA_PATH = "tests/testdata"

CONSEC_PATH = f"{TESTDATA_PATH}/consec"
NONCONSEC_PATH = f"{TESTDATA_PATH}/nonconsec"
