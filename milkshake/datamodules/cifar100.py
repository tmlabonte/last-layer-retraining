"""DataModule for the CIFAR-100 dataset."""

# Imports Python packages.
import numpy as np
import os.path as osp
import pickle

# Imports PyTorch packages.
from pl_bolts.transforms.dataset_normalizations import cifar10_normalization
from torchvision.datasets import CIFAR100 as TorchvisionCIFAR100
from torchvision.transforms import Compose, RandomCrop, RandomHorizontalFlip, ToTensor

# Imports milkshake packages.
from milkshake.datamodules.dataset import Dataset
from milkshake.datamodules.datamodule import DataModule


class CIFAR100Dataset(Dataset, TorchvisionCIFAR100):
    """Dataset for the CIFAR-100 dataset."""

    def __init__(self, *xargs, **kwargs):
        Dataset.__init__(self, *xargs, **kwargs)

    def download(self):
        return TorchvisionCIFAR100.download(self)

    def load_data(self):
        if not self._check_integrity():
            raise ValueError("Use download=True to re-download the dataset.")

        downloaded_list = self.train_list if self.train else self.test_list

        self.data = []
        self.targets = []

        for file_name, checksum in downloaded_list:
            file_path = osp.join(self.root, self.base_folder, file_name)
            with open(file_path, "rb") as f:
                entry = pickle.load(f, encoding="latin1")
                self.data.append(entry["data"])
                if "labels" in entry:
                    self.targets.extend(entry["labels"])
                else:
                    self.targets.extend(entry["fine_labels"])

        self.data = np.vstack(self.data).reshape(-1, 3, 32, 32)
        self.data = self.data.transpose((0, 2, 3, 1))  # convert to HWC
        self.targets = np.asarray(self.targets)

        self._load_meta()

class CIFAR100(DataModule):
    """DataModule for the CIFAR-100 dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, CIFAR100Dataset, 100, 0, **kwargs)

    def augmented_transforms(self):
        transforms = Compose([
            RandomCrop(32, padding=4),
            RandomHorizontalFlip(),
            ToTensor(),
            cifar10_normalization(),
        ])

        return transforms

    def default_transforms(self):
        transforms = Compose([
            ToTensor(),
            cifar10_normalization()
        ])

        return transforms

