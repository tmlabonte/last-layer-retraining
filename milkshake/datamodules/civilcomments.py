"""Dataset and DataModule for the CivilComments dataset."""

# Imports Python packages.
import numpy as np
import os.path as osp
import pickle
from transformers import BertTokenizer
import wilds

# Imports PyTorch packages.
import torch

# Imports milkshake packages.
from milkshake.datamodules.dataset import Dataset
from milkshake.datamodules.datamodule import DataModule
from milkshake.utils import to_np


class CivilCommentsDataset(Dataset):
    """Dataset for the CivilComments dataset."""

    def __init__(self, *xargs, **kwargs):
        super().__init__(*xargs, **kwargs)

    def download(self):
        pass

    def load_data(self):
        dataset = wilds.get_dataset(
            dataset="civilcomments",
            download=True,
            root_dir=self.root,
        )

        spurious_names = ["male", "female", "LGBTQ", "black", "white",
                          "christian", "muslim", "other_religions"]
        column_names = dataset.metadata_fields
        spurious_cols = [column_names.index(name) for name in spurious_names]
        spurious = to_np(dataset._metadata_array[:, spurious_cols].sum(-1).clip(max=1))

        prefix = osp.join(self.root, "civilcomments_v1.0")
        data_file = osp.join(prefix, "civilcomments_token_data.pt")
        targets_file = osp.join(prefix, "civilcomments_token_targets.pt")

        if not osp.isfile(data_file):
            tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
            def tokenize(text):
                tokens = tokenizer(
                    text,
                    padding="max_length",
                    truncation=True,
                    max_length=220,
                    return_tensors="pt",
                )

                return torch.squeeze(torch.stack((
                    tokens["input_ids"], tokens["attention_mask"], 
                    tokens["token_type_ids"]), dim=2), dim=0)

            data = []
            targets = []
            ln = len(dataset)
            for j, d in enumerate(dataset):
                print(f"Caching {j}/{ln}")
                data.append(tokenize(d[0]))
                targets.append(d[1])
            data = torch.stack(data)
            targets = torch.stack(targets)
            torch.save(data, data_file)
            torch.save(targets, targets_file)

        self.data = torch.load(data_file).numpy()
        self.targets = torch.load(targets_file).numpy()

        self.groups = [
            np.intersect1d((~self.targets+2).nonzero()[0], (~spurious+2).nonzero()[0]),
            np.intersect1d((~self.targets+2).nonzero()[0], spurious.nonzero()[0]),
            np.intersect1d(self.targets.nonzero()[0], (~spurious+2).nonzero()[0]),
            np.intersect1d(self.targets.nonzero()[0], spurious.nonzero()[0]),
        ]
        
        split = dataset._split_array
        self.train_indices = np.argwhere(split == 0).flatten()
        self.val_indices = np.argwhere(split == 1).flatten()
        self.test_indices = np.argwhere(split == 2).flatten()

        # Adds group indices into targets for metrics.
        targets = []
        for j, t in enumerate(self.targets):
            g = [k for k, group in enumerate(self.groups) if j in group][0]
            targets.append([t, g])
        self.targets = np.asarray(targets)

class CivilComments(DataModule):
    """DataModule for the CivilComments dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, CivilCommentsDataset, 2, 4, **kwargs)

    def augmented_transforms(self):
        return None

    def default_transforms(self):
        return None
