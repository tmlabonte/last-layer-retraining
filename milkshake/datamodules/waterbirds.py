"""Dataset and DataModule for the Waterbirds dataset."""

# Imports Python builtins.
import os.path as osp

# Imports Python packages.
import numpy as np
import pandas as pd
from PIL import Image

# Imports PyTorch packages.
import torch
import torch.nn.functional as F
from torchvision.datasets.utils import download_and_extract_archive
from torchvision.transforms import (
    CenterCrop,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)

# Imports milkshake packages.
from milkshake.datamodules.dataset import Dataset
from milkshake.datamodules.datamodule import DataModule


class WaterbirdsDataset(Dataset):
    """Dataset for the Waterbirds dataset."""

    def __init__(self, *xargs, **kwargs):
        super().__init__(*xargs, **kwargs)

    def download(self):
        waterbirds_dir = osp.join(self.root, "waterbirds")
        if not osp.isdir(waterbirds_dir):
            url = (
                "http://worksheets.codalab.org/rest/bundles/"
                "0x505056d5cdea4e4eaa0e242cbfe2daa4/contents/blob/"
            )

            download_and_extract_archive(
                url,
                waterbirds_dir,
                filename="waterbirds.tar.gz",
            )

    def load_data(self):
        waterbirds_dir = osp.join(self.root, "waterbirds")
        metadata_df = pd.read_csv(osp.join(waterbirds_dir, "metadata.csv"))
        self.data = np.asarray(metadata_df["img_filename"].values)
        self.data = np.asarray([osp.join(waterbirds_dir, d) for d in self.data])

        self.targets = np.asarray(metadata_df["y"].values)
        background = np.asarray(metadata_df["place"].values)
        landbirds = np.argwhere(self.targets == 0).flatten()
        waterbirds = np.argwhere(self.targets == 1).flatten()
        land = np.argwhere(background == 0).flatten()
        water = np.argwhere(background == 1).flatten()

        self.groups = [
            np.intersect1d(landbirds, land),
            np.intersect1d(landbirds, water),
            np.intersect1d(waterbirds, land),
            np.intersect1d(waterbirds, water),
        ]
        
        split = np.asarray(metadata_df["split"].values)
        self.train_indices = np.argwhere(split == 0).flatten()
        self.val_indices = np.argwhere(split == 1).flatten()
        self.test_indices = np.argwhere(split == 2).flatten()

        # Adds group indices into targets for metrics.
        targets = []
        for j, t in enumerate(self.targets):
            g = [k for k, group in enumerate(self.groups) if j in group][0]
            targets.append([t, g])
        self.targets = np.asarray(targets)

class Waterbirds(DataModule):
    """DataModule for the Waterbirds dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, WaterbirdsDataset, 2, 4, **kwargs)

    def augmented_transforms(self):
        transform = Compose([
            RandomResizedCrop((224, 224), scale=(0.7, 1.0)),
            RandomHorizontalFlip(),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        return transform

    def default_transforms(self):
        transform = Compose([
            Resize((256, 256)),
            CenterCrop((224, 224)),
            ToTensor(),
            Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])

        return transform

