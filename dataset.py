import pandas as pd
from torch.utils.data import Dataset

from utils import audio, spectrogram
from preprocess_dataset import DATA_DIR


class AudioDataset(Dataset):
    def __init__(self):
        self.attributes = pd.read_csv(f"{DATA_DIR}/attributes.csv")

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, index):
        audio_path = self.attributes.iloc[index, 0]
        spec_path = self.attributes.iloc[index, 2]
        audio_tensor = audio.load_audio(audio_path, "merge")
        spec = spectrogram.load_spectrogram_tiff(spec_path)
        return audio_tensor, spec
