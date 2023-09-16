"""Dataset and DataModule for the MultiNLI dataset."""

# Imports Python builtins.
import os
import os.path as osp
import sys

# Imports Python packages.
import numpy as np
import pandas as pd
import wget

# Imports PyTorch packages.
import torch
from torchvision.datasets.utils import (
    extract_archive,
)

# Imports milkshake packages.
from milkshake.datamodules.dataset import Dataset
from milkshake.datamodules.datamodule import DataModule


class MultiNLIDataset(Dataset):
    """Dataset for the MultiNLI dataset."""

    def __init__(self, *xargs, **kwargs):
        super().__init__(*xargs, **kwargs)

    def download(self):
        multinli_dir = osp.join(self.root, "multinli")
        if not osp.isdir(multinli_dir):
            os.makedirs(multinli_dir)

            url = (
                "https://github.com/kohpangwei/group_DRO/raw/"
                "f7eae929bf4f9b3c381fae6b1b53ab4c6c911a0e/"
                "dataset_metadata/multinli/metadata_random.csv"
            )
            wget.download(url, out=multinli_dir)

            url = "https://nlp.stanford.edu/data/dro/multinli_bert_features.tar.gz"
            wget.download(url, out=multinli_dir)
            extract_archive(osp.join(multinli_dir, "multinli_bert_features.tar.gz"))

            url = (
                "https://raw.githubusercontent.com/izmailovpavel/"
                "spurious_feature_learning/6d098440c697a1175de6a24"
                "d7a46ddf91786804c/dataset_files/utils_glue.py"
            )
            wget.download(url, out=multinli_dir)
    
    def load_data(self):
        multinli_dir = osp.join(self.root, "multinli")
        sys.path.append(multinli_dir)
        metadata_path = osp.join(multinli_dir, "metadata_random.csv")
        metadata_df = pd.read_csv(metadata_path)

        bert_filenames = [
            "cached_train_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli",
            "cached_dev_bert-base-uncased_128_mnli-mm",
        ]

        features_array = sum([torch.load(osp.join(multinli_dir, name))
                              for name in bert_filenames], start=[])

        all_input_ids = torch.tensor([
            f.input_ids for f in features_array
        ]).long()
        all_input_masks = torch.tensor([
            f.input_mask for f in features_array
        ]).long()
        all_segment_ids = torch.tensor([
            f.segment_ids for f in features_array
        ]).long()

        self.data = torch.stack((
            all_input_ids,
            all_input_masks,
            all_segment_ids,
        ), dim=2)
        self.targets = np.asarray(metadata_df["gold_label"].values)

        spurious = np.asarray(metadata_df["sentence2_has_negation"].values)
        no_negation = np.argwhere(spurious == 0).flatten()
        negation = np.argwhere(spurious == 1).flatten()
        contradiction = np.argwhere(self.targets == 0).flatten()
        entailment = np.argwhere(self.targets == 1).flatten()
        neutral = np.argwhere(self.targets == 2).flatten()

        self.groups = [
            np.intersect1d(contradiction, no_negation),
            np.intersect1d(contradiction, negation),
            np.intersect1d(entailment, no_negation),
            np.intersect1d(entailment, negation),
            np.intersect1d(neutral, no_negation),
            np.intersect1d(neutral, negation),
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

class MultiNLI(DataModule):
    """DataModule for the MultiNLI dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, MultiNLIDataset, 3, 6, **kwargs)

    def augmented_transforms(self):
        return None

    def default_transforms(self):
        return None
