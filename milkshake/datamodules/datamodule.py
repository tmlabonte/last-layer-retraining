"""Parent class for a classification datamodule."""

# Imports Python builtins.
from abc import abstractmethod
from copy import deepcopy
import random

# Imports Python packages.
import numpy as np
from numpy.random import default_rng

# Imports PyTorch packages.
from torch import Generator, randperm
from torch.utils.data import DataLoader, WeightedRandomSampler
from pl_bolts.datamodules.vision_datamodule import VisionDataModule

# Imports milkshake packages.
from milkshake.datamodules.dataset import Subset
from milkshake.utils import random_split


class DataModule(VisionDataModule):
    """Parent class for a classification datamodule.

    Extends the basic PL.VisionDataModule to play nice with
    torchvision.datasets.VisionDataset and adds some custom functionality
    such as label noise, balanced sampling, fixed splits, etc.

    It inherits from PL.VisionDataModule, but milkshake.DataModule is more general
    and can be used for vision, language, or other applications.

    Attributes:
        dataset_class: A milkshake.datamodules.datasets.Dataset class.
        num_classes: The number of classes.
        num_groups: The number of groups.
        balanced_sampler: Whether to use a class-balanced random sampler during training.
        data_augmentation: Whether to use data augmentation during training.
        label_noise: Whether to add label noise during training.
        dataset_train: A milkshake.datamodules.dataset.Subset for training
        dataset_val: A milkshake.datamodules.dataset.Subset for validation.
        dataset_test: A milkshake.datamodules.dataset.Subset for testing.
        train_transforms: Preprocessing for the train set data.
        val_transforms: Preprocessing for the validation set data.
        test_transforms: Preprocessing for the test set data.
        train_transforms: Preprocessing for the train set targets.
        val_target_transforms: Preprocessing for the validation set targets.
        test_target_transforms: Preprocessing for the test set targets.
    """

    def __init__(self, args, dataset_class, num_classes, num_groups):
        """Initializes a DataModule and sets transforms.

        Args:
            args: The configuration dictionary.
            dataset_class: A milkshake.datamodules.datasets.Dataset class.
            num_classes: The number of classes.
            num_groups: The number of groups.
        """

        super().__init__(
            batch_size=args.batch_size,
            data_dir=args.data_dir,
            drop_last=False,
            normalize=True,
            num_workers=args.num_workers,
            pin_memory=True,
            seed=args.seed,
            shuffle=True,
            val_split=args.val_split,
        )

        self.persistent_workers = args.persistent_workers

        self.dataset_class = dataset_class
        self.num_classes = num_classes
        self.num_groups = num_groups
        
        self.balanced_sampler = args.balanced_sampler
        self.data_augmentation = args.data_augmentation
        self.label_noise = args.label_noise
         
        self.train_transforms = self.default_transforms()
        self.val_transforms = self.default_transforms()
        self.test_transforms = self.default_transforms()

        self.train_target_transforms = None
        self.val_target_transforms = None
        self.test_target_transforms = None

        if self.data_augmentation:
            self.train_transforms = self.augmented_transforms()

    @abstractmethod
    def augmented_transforms(self):
        """Returns torchvision.transforms for use with data augmentation."""

    @abstractmethod
    def default_transforms(self):
        """Returns default torchvision.transforms."""
 
    def prepare_data(self):
        """Downloads datasets to disk if necessary."""

        self.dataset_class(self.data_dir, train=True, download=True)
        self.dataset_class(self.data_dir, train=False, download=True)

    def load_msg(self):
        """Returns a descriptive message about the DataModule configuration."""

        msg = f"Loading {type(self).__name__}"
 
        if hasattr(self, "dataset_test"):
            msg = msg + " with test split."
        elif hasattr(self, "dataset_val"):
            if self.dataset_val.val_indices is not None:
                msg = msg + " with preset val split."
            else:
                msg = msg + f" with {int(self.val_split * 100)}% random val split."

            if self.data_augmentation:
                msg = msg[:-1] + " and data augmentation."
            if self.label_noise:
                msg = msg[:-1] + f" and {int(self.label_noise * 100)}% label noise."
            if self.balanced_sampler:
                msg = msg[:-1] + " and a balanced sampler."

        return msg

    def train_preprocess(self, dataset_train):
        """Preprocesses training dataset. Here, injects label noise if desired.

        Args:
            dataset_train: A milkshake.datamodules.dataset.Subset for training.

        Returns:
            The modified training dataset.
        """

        if self.label_noise:
            # Shuffle training data in unison.
            num_samples = len(dataset_train.targets)
            p = default_rng(seed=self.seed).permutation(num_samples)
            dataset_train.train_indices = dataset_train.train_indices[p]
            dataset_train.targets = dataset_train.targets[p]

            # Injects label noise into the training dataset.
            num_noised_labels = int(self.label_noise * num_samples)
            for i, target in enumerate(dataset_train.targets[:num_noised_labels]):
                if target.ndim > 0:
                    labels = [j for j in range(self.num_classes) if j != target[0]]
                    dataset_train.targets[i][0] = random.choice(labels)
                else:
                    labels = [j for j in range(self.num_classes) if j != target]
                    dataset_train.targets[i] = random.choice(labels)

        return dataset_train

    def val_preprocess(self, dataset_val):
        """Preprocesses validation dataset. Does nothing here, but can be overriden.

        Args:
            dataset_val: A milkshake.datamodules.dataset.Subset for validation.

        Returns:
            The modified validation dataset.
        """

        return dataset_val

    def test_preprocess(self, dataset_test):
        """Preprocesses test dataset. Does nothing here, but can be overidden.

        Args:
            dataset_test: A milkshake.datamodules.dataset.Subset for testing.

        Returns:
            The modified test dataset.
        """

        return dataset_test

    def setup(self, stage=None):
        """Instantiates and preprocesses datasets.
        
        Args:
            stage: "train", "validate", or "test".
        """

        if stage == "test":
            dataset_test = self.dataset_class(
                self.data_dir,
                train=False,
                transform=self.test_transforms,
                target_transform=self.test_target_transforms,
            )

            dataset_test = self.test_preprocess(dataset_test)
            self.dataset_test = dataset_test
        else:
            dataset_train = self.dataset_class(
                self.data_dir,
                train=True,
                transform=self.train_transforms,
                target_transform=self.train_target_transforms,
            )

            dataset_val = self.dataset_class(
                self.data_dir,
                train=True,
                transform=self.val_transforms,
                target_transform=self.val_target_transforms,
            )

            
            dataset_train = self._split_dataset(dataset_train)
            self.dataset_train = self.train_preprocess(dataset_train)

            dataset_val = self._split_dataset(dataset_val, val=True)
            self.dataset_val = self.val_preprocess(dataset_val)

        print(self.load_msg())

    def _split_dataset(self, dataset, val=False):
        """Splits dataset into training and validation subsets.

        Checks for a preset split, then returns a random split if it does not exist.
        
        Args:
            dataset: A milkshake.datamodules.dataset.Dataset.
            val: Whether to return the validation set (otherwise returns the training set).

        Returns:
            A milkshake.datamodules.dataset.Subset of the given dataset with the desired split.
        """

        if dataset.train_indices is None or dataset.val_indices is None:
            len_dataset = len(dataset)
            splits = self._get_splits(len_dataset)

            # Computes a random split with proportion based on self.split.
            dataset_train, dataset_val = random_split(
                dataset,
                splits,
                generator=Generator().manual_seed(self.seed),
            )
        else:
            dataset_train = Subset(dataset, dataset.train_indices)
            dataset_val = Subset(dataset, dataset.val_indices)
                        
        if val:
            return dataset_val
        return dataset_train

    def train_dataloader(self):
        """Returns DataLoader for the train dataset."""

        # Instantiates class-balanced random sampler.
        if self.balanced_sampler:
            indices = self.dataset_train.train_indices
            targets = self.dataset_train.targets[indices]

            # Removes group dimension if necessary.
            if targets[0].ndim > 0:
                targets = targets[:, 0]

            counts = np.bincount(targets)
            label_weights = 1. / counts
            weights = label_weights[targets]
            sampler = WeightedRandomSampler(weights, len(weights))

            return self._data_loader(self.dataset_train, sampler=sampler)

        return self._data_loader(self.dataset_train, shuffle=self.shuffle)

    def val_dataloader(self):
        """Returns DataLoader(s) for the val dataset."""

        return self._data_loader(self.dataset_val)

    def test_dataloader(self):
        """Returns DataLoader(s) for the test dataset."""

        return self._data_loader(self.dataset_test)

    def _data_loader(self, dataset, shuffle=False, sampler=None):
        """Instantiates DataLoader with the given dataset.

        Args:
            dataset: A milkshake.datamodules.dataset.Dataset.
            shuffle: Whether to shuffle the data indices.
            sampler: A torch.utils.data.Sampler for selecting data indices.

        Returns:
            A torch.utils.data.DataLoader with the given configuration.
        """

        return DataLoader(
            dataset,
            batch_size=self.batch_size,
            drop_last=self.drop_last,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            sampler=sampler,
            shuffle=shuffle,
        )

