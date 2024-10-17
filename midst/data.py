import os
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

class MidstDataset(Dataset):
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

        features_path = os.path.join(self.base_dir, "challenge_with_id.csv")
        if not os.path.exists: raise FileNotFoundError(f"Features Path: {features_path} not found.")
        self.features = pd.read_csv(features_path)

        labels_path = os.path.join(self.base_dir, "challenge_label.csv")
        if not os.path.exists: raise FileNotFoundError(f"Labels Path: {labels_path} not found.")
        self.labels = pd.read_csv(labels_path)

        assert len(self.features) == len(self.labels)

    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = self.features.iloc[idx].to_numpy(), self.labels.iloc[idx].to_numpy()
        return torch.from_numpy(x), torch.from_numpy(y)

def get_features_and_labels(base_dir: Path) -> tuple[torch.Tensor, torch.Tensor]:
    dataset = MidstDataset(base_dir)

    data_loader = DataLoader(dataset, batch_size=200)
    features, labels = next(iter(data_loader))

    return features, labels


