from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from src.data.preprocess import ben_graham

MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


def build_transforms(train, size=256):
    norm = [A.Normalize(mean=MEAN, std=STD), ToTensorV2()]
    if not train:
        return A.Compose([A.Resize(size, size)] + norm)
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Rotate(limit=180, border_mode=cv2.BORDER_CONSTANT, p=0.9),
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.5),
    ] + norm)


class DRDataset(Dataset):
    def __init__(self, splits_csv, split, img_dir, transform=None, enhance=False):
        df = pd.read_csv(splits_csv)
        self.df = df[df["split"] == split].reset_index(drop=True)
        self.img_dir = Path(img_dir)
        self.transform = transform
        self.enhance = enhance

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        path = self.img_dir / f"{row.id_code}.png"
        img = cv2.imread(str(path))
        if img is None:
            raise FileNotFoundError(f"Could not read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.enhance:
            img = ben_graham(img)
        if self.transform is not None:
            img = self.transform(image=img)["image"]
        return img, int(row.diagnosis)


def class_weights(df, num_classes=5, power=1.0):
    """Inverse-frequency weights raised to `power` (1 = full, 0.5 = square root, 0 = none)."""
    counts = df["diagnosis"].value_counts().reindex(range(num_classes)).fillna(0).values
    w = len(df) / (num_classes * counts)
    return torch.tensor(w ** power, dtype=torch.float32)
