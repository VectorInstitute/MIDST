import os
import pandas as pd
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader

class ChallengeDataset(Dataset):
    def __init__(self, base_dir: Path) -> None:
        self.base_dir = base_dir

        challenge_points_path = os.path.join(self.base_dir, "challenge_with_id.csv")
        if not os.path.exists: raise FileNotFoundError(f"Challenge Points Path: {challenge_points_path} not found.")
        self.challenge_points = pd.read_csv(challenge_points_path)

    def __len__(self) -> int:
        return len(self.challenge_points)
    
    def __getitem__(self, idx) -> torch.Tensor:
        x = self.challenge_points.iloc[idx].to_numpy()
        return torch.from_numpy(x)

def get_challenge_points(base_dir: Path) -> torch.Tensor: 
    dataset = ChallengeDataset(base_dir)

    loader = DataLoader(dataset, batch_size=200)
    challenge_points = next(iter(loader))

    return challenge_points


