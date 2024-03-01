import pandas as pd
from torch.utils.data import Dataset, DataLoader

from utils import audio, spectrogram
from preprocess_dataset import DATA_DIR


class AudioDataset(Dataset):
    def __init__(self):
        self.attributes = pd.read_csv(f"{DATA_DIR}/attributes.csv")

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, index):
        audio_path = self.attributes.iloc[index, 1]
        spec_path = self.attributes.iloc[index, 3]
        audio_tensor, _ = audio.load_audio(audio_path, "merge")
        spec = spectrogram.load_spectrogram_tiff(spec_path)
        return audio_tensor, spec


if __name__ == "__main__":
    data = AudioDataset()
    dataloader = DataLoader(data, batch_size=64, shuffle=True)
    audio_tensors, spectrograms = next(iter(dataloader))
    print(f"Feature batch shape: {audio_tensors.shape}")
    print(f"Labels batch shape: {spectrograms.shape}")
