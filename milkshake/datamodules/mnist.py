"""DataModule for the MNIST dataset."""

# Imports PyTorch packages.
import torch
from torchvision.datasets import MNIST as TorchVisionMNIST
from torchvision.transforms import Compose, Normalize, ToTensor

# Imports milkshake packages.
from milkshake.datamodules.datamodule import DataModule
from milkshake.datamodules.dataset import Dataset


class MNISTDataset(Dataset, TorchVisionMNIST):
    """Dataset for the MNIST dataset."""

    def __init__(self, *xargs, **kwargs):
        Dataset.__init__(self, *xargs, **kwargs)

    def download(self):
        return TorchVisionMNIST.download(self)

    def load_data(self):
        data, self.targets = TorchVisionMNIST._load_data(self)
        self.data = torch.unsqueeze(data.float(), 1) # CHW format for Tensors

class MNIST(DataModule):
    """DataModule for the MNIST dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, MNISTDataset, 10, 0, **kwargs)

    def augmented_transforms(self):
        return self.default_transforms()

    def default_transforms(self):
        return Normalize(mean=(0.5,), std=(0.5,))

