"""Utility functions for milkshake."""

# Imports Python packages.
import os
import os.path as osp
from glob import glob
import math
import numpy as np
import warnings

# Imports PyTorch packages.
import torch
from torch._utils import _accumulate 

# Imports milkshake packages.
from milkshake.datamodules.dataset import Subset


def get_weights(args, version, best=None, idx=None):
    """Returns weights path given model version and checkpoint index.
    
    Args:
        args: The configuration dictionary.
        version: The model's PyTorch Lightning version.
        best: Whether to return the best model checkpoint.
        idx: The model's checkpoint index (-1 selects the latest checkpoint).
    
    Returns:
        The filepath of the desired model weights.
    """

    ckpt_path = ""
    if args.wandb:
        ckpt_path = args.wandb_dir
    
    # Selects the right naming convention for PL versions based on
    # whether the version input is an int or a string.
    if type(version) == int:
        ckpt_path = osp.join(ckpt_path, f"lightning_logs/version_{version}/checkpoints/*")
    else:
        ckpt_path = osp.join(ckpt_path, f"lightning_logs/{version}/checkpoints/*")

    list_of_weights = glob(osp.join(os.getcwd(), ckpt_path))
    
    if best:
        return [w for w in list_of_weights if "best" in w][0]
    else:
        list_of_weights = sorted([w for w in list_of_weights if "best" not in w])
        return list_of_weights[idx]

def compute_accuracy(probs, targets, num_classes, num_groups):
    """Computes top-1 and top-5 accuracies.
    
    Computes top-1 and top-5 accuracies by total and by class, and also
    includes the number of correct predictions and number of samples.
    The latter is useful for, e.g., collating metrics over an epoch.
    If groups are provided (as a second column in targets), then
    also returns group accuracies.

    Args:
        probs: A torch.Tensor of prediction probabilities.
        targets: A torch.Tensor of classification targets.
        num_classes: The total number of classes.
        num_groups: The total number of groups.

    Returns:
        A dictionary of metrics including top-1 and top-5 accuracies, number of
        correct predictions, and number of samples, each by total, class, and group.
    """

    # TODO: Clean up group metrics.

    # Splits apart targets and groups if necessary.
    groups = None
    if num_groups:
        groups = targets[:, 1]
        targets = targets[:, 0]

    if num_classes == 1:
        preds1 = (probs >= 0.5).int()
    else:
        preds1 = torch.argmax(probs, dim=1)

    correct1 = (preds1 == targets).float()
    
    acc1_by_class = []
    correct1_by_class = []
    total_by_class = []
    for j in range(num_classes):
        correct_nums = correct1[targets == j]
        acc1_by_class.append(correct_nums.mean())
        correct1_by_class.append(correct_nums.sum())
        total_by_class.append((targets == j).sum())

    acc1_by_group = [torch.tensor(1., device=targets.device)]
    correct1_by_group = [torch.tensor(1., device=targets.device)]
    total_by_group = [torch.tensor(1., device=targets.device)]
    if num_groups:
        acc1_by_group = []
        correct1_by_group = []
        total_by_group = []
        for j in range(num_groups):
            correct_nums = correct1[groups == j]
            acc1_by_group.append(correct_nums.mean())
            correct1_by_group.append(correct_nums.sum())
            total_by_group.append((groups == j).sum())

    correct5 = torch.tensor([1.] * len(targets), device=targets.device)
    acc5_by_class = [torch.tensor(1., device=targets.device)] * num_classes
    correct5_by_class = total_by_class
    if num_classes > 5:
        _, preds5 = torch.topk(probs, k=5, dim=1)
        correct5 = torch.tensor([t in preds5[j] for j, t in enumerate(targets)])
        correct5 = correct5.float()

        acc5_by_class = []
        correct5_by_class = []
        for j in range(num_classes):
            correct_nums = correct5[targets == j]
            acc5_by_class.append(correct_nums.mean())
            correct5_by_class.append(correct_nums.sum())

    acc5_by_group = [torch.tensor(1., device=targets.device)]
    correct5_by_group = [torch.tensor(1., device=targets.device)]
    if num_groups:
        acc5_by_group = [torch.tensor(1., device=targets.device)] * num_groups
        correct5_by_group = total_by_group
        if num_classes > 5:
            acc5_by_group = []
            correct5_by_group = []
            for j in range(num_groups):
                correct_nums = correct5[groups == j]
                acc5_by_group.append(correct_nums.mean())
                correct5_by_group.append(correct_nums.sum())

    accs = {
        "acc": correct1.mean(),
        "acc5": correct5.mean(),
        "acc_by_class": torch.stack(acc1_by_class),
        "acc5_by_class": torch.stack(acc5_by_class),
        "acc_by_group": torch.stack(acc1_by_group),
        "acc5_by_group": torch.stack(acc5_by_group),
        "correct": correct1.sum(),
        "correct5": correct5.sum(),
        "correct_by_class": torch.stack(correct1_by_class),
        "correct5_by_class": torch.stack(correct5_by_class),
        "correct_by_group": torch.stack(correct1_by_group),
        "correct5_by_group": torch.stack(correct5_by_group),
        "total": len(targets),
        "total_by_class": torch.stack(total_by_class),
        "total_by_group": torch.stack(total_by_group),
    }
    
    return accs

def _to_np(x):
    """Converts torch.Tensor input to numpy array."""

    return x.cpu().detach().numpy()

def to_np(x):
    """Converts input to numpy array.

    Args:
        x: A torch.Tensor, np.ndarray, or list.

    Returns:
        The input converted to a numpy array.

    Raises:
        ValueError: The input cannot be converted to a numpy array.
    """

    if not len(x):
        return np.array([])
    elif isinstance(x, torch.Tensor):
        return _to_np(x)
    elif isinstance(x, (np.ndarray, list)):
        if isinstance(x[0], torch.Tensor):
            return _to_np(torch.tensor(x))
        else:
            return np.asarray(x)
    else:
        raise ValueError("Input cannot be converted to numpy array.")

def random_split(dataset, lengths, generator):
    """Random split function from PyTorch adjusted for milkshake.Subset.
    
    Args:
        dataset: The milkshake.Dataset to be randomly split.
        lengths: The lengths or fractions of splits to be produced.
        generator: The generator used for the random permutation.
    
    Returns:
        A list of milkshake.Subsets with the desired splits.
    
    Raises:
        ValueError: The sum of input lengths does not equal the length of the input dataset.
    """

    # Handles the case where lengths is a list of fractions.
    if math.isclose(sum(lengths), 1) and sum(lengths) <= 1:
        subset_lengths = []
        for i, frac in enumerate(lengths):
            if frac < 0 or frac > 1:
                raise ValueError(f"Fraction at index {i} is not between 0 and 1")
            n_items_in_split = int(
                math.floor(len(dataset) * frac)
            )
            subset_lengths.append(n_items_in_split)
        remainder = len(dataset) - sum(subset_lengths)

        # Adds 1 to all the lengths in round-robin fashion until the remainder is 0.
        for i in range(remainder):
            idx_to_add_at = i % len(subset_lengths)
            subset_lengths[idx_to_add_at] += 1
        lengths = subset_lengths
        for i, length in enumerate(lengths):
            if length == 0:
                print(f"Length of split at index {i} is 0. "
                      f"This might result in an empty dataset.")

    if sum(lengths) != len(dataset):    # type: ignore[arg-type]
        raise ValueError("Sum of input lengths does not equal the length of the input dataset!")

    indices = torch.randperm(sum(lengths), generator=generator).tolist() 
    return [Subset(dataset, indices[offset - length : offset])
            for offset, length in zip(_accumulate(lengths), lengths)]

def ignore_warnings():
    """Adds nuisance warnings to ignore list.

    Should be called before import pytorch_lightning.
    """

    warnings.filterwarnings(
        "ignore",
        message=r"The feature ([^\s]+) is currently marked under review",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"In the future ([^\s]+)",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"Lazy modules ([^\s]+)",
    )

    warnings.filterwarnings(
        "ignore",
        message=r"There is a wandb run already in progress ([^\s]+)",
    )

    original_filterwarnings = warnings.filterwarnings
    def _filterwarnings(*xargs, **kwargs):
        return original_filterwarnings(*xargs, **{**kwargs, "append": True})
    warnings.filterwarnings = _filterwarnings

