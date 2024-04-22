import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader

from preprocess_dataset import DATA_DIR


class DNNDataset(Dataset):
    def __init__(self):
        self.attributes = pd.read_csv(f"{DATA_DIR}/attributes.csv")

    def __len__(self):
        return len(self.attributes)

    def __getitem__(self, index):
        entry_path = self.attributes.iloc[index, 2]
        inputs, target = torch.load(entry_path)
        return inputs, target


if __name__ == "__main__":
    DNN_data = DNNDataset()
    DNN_dataloader = DataLoader(DNN_data, batch_size=64, shuffle=True)
    input_specs, target_specs = next(iter(DNN_dataloader))
    print(f"DNN input spectrograms batch shape: {input_specs.shape}")
    print(f"DNN target spectrograms batch shape: {target_specs.shape}")
