"""PyTorch Dataset for HAM10000, applying the preprocessing pipeline + augmentation."""
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import pandas as pd
import torch
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset

from src.preprocessing.hair_removal import remove_hair
from src.preprocessing.normalization import normalize_image

CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_transforms(image_size: int, train: bool) -> A.Compose:
    if train:
        aug = [
            A.RandomResizedCrop(size=(image_size, image_size), scale=(0.8, 1.0)),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.Rotate(limit=30, p=0.5),
            A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05, p=0.5),
        ]
    else:
        aug = [A.Resize(height=image_size, width=image_size)]
    aug += [A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD), ToTensorV2()]
    return A.Compose(aug)


class HAM10000Dataset(Dataset):
    def __init__(self, split_csv: str, images_dir: str, image_size: int = 224, train: bool = False,
                 hair_removal: bool = True, clahe: bool = True, segmentations_dir: str = None):
        self.metadata = pd.read_csv(split_csv)
        self.images_dir = Path(images_dir)
        self.segmentations_dir = Path(segmentations_dir) if segmentations_dir else None
        self.hair_removal = hair_removal
        self.clahe = clahe
        self.transforms = build_transforms(image_size, train)

    def __len__(self) -> int:
        return len(self.metadata)

    def _load_image(self, image_id: str) -> np.ndarray:
        path = self.images_dir / f"{image_id}.jpg"
        image = cv2.imread(str(path))
        if image is None:
            raise FileNotFoundError(f"Could not read image at {path}")
        return image

    def __getitem__(self, idx: int):
        row = self.metadata.iloc[idx]
        image = self._load_image(row["image_id"])

        if self.hair_removal:
            image = remove_hair(image)
        image = normalize_image(image, use_clahe=self.clahe)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        transformed = self.transforms(image=image)
        tensor = transformed["image"]
        label = CLASS_TO_IDX[row["dx"]]

        sample = {"image": tensor, "label": torch.tensor(label, dtype=torch.long), "image_id": row["image_id"]}

        if self.segmentations_dir is not None:
            mask_path = self.segmentations_dir / f"{row['image_id']}_segmentation.png"
            if mask_path.exists():
                mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
                sample["mask"] = mask

        return sample
