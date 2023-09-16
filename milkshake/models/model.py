"""Parent class and training logic for a classification model."""

# Imports Python builtins.
from abc import abstractmethod
import numpy as np

# Imports PyTorch packages.
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import is_lazy
from torch.optim import Adam, AdamW, SGD
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, MultiStepLR
import torchvision.models as models

# Imports milkshake packages.
from milkshake.utils import compute_accuracy, to_np

# TODO: Organize logging code in a separate file.
# TODO: Implement after_n_steps logging similarly to after_epoch.
# TODO: Clean up collate_metrics.


class Model(pl.LightningModule):
    """Parent class and training logic for a classification model.

    Attributes:
        self.hparams: The configuration dictionary.
        self.model: A torch.nn.Module.
        self.optimizer: A torch.optim optimizer.
    """

    def __init__(self, args):
        """Initializes a Model.

        Args:
            args: The configuration dictionary.
        """

        super().__init__()

        # Saves args into self.hparams.
        self.save_hyperparameters(args)
        print(self.load_msg())
        
        self.model = None
        
        optimizers = {"adam": Adam, "adamw": AdamW, "sgd": SGD}
        self.optimizer = optimizers[args.optimizer]

    @abstractmethod
    def load_msg(self):
        """Returns a descriptive message about the Model configuration."""

    def has_uninitialized_params(self):
        """Returns whether the model has uninitialized parameters."""

        for param in self.model.parameters():
            if is_lazy(param):
                return True
        return False

    def setup(self, stage):
        """Initializes model parameters if necessary.

        Args:
            stage: "train", "val", or "test".
        """

        if self.has_uninitialized_params():
            match stage:
                case "fit":
                    dataloader = self.trainer.datamodule.train_dataloader()
                case "validate":
                    dataloader = self.trainer.datamodule.val_dataloader()
                case "test":
                    dataloader = self.trainer.datamodule.test_dataloader()
                case "predict":
                    dataloader = self.trainer.datamodule.predict_dataloader()
            dummy_batch = next(iter(dataloader))
            self.forward(dummy_batch[0])

    def forward(self, inputs):
        """Predicts using the model.

        Args:
            inputs: A torch.Tensor of model inputs.
        
        Returns:
            The model prediction as a torch.Tensor.
        """
        return torch.squeeze(self.model(inputs), dim=-1)

    def configure_optimizers(self):
        """Returns the optimizer and learning rate scheduler."""

        optimizer = self.optimizer(
            self.model.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )

        if isinstance(optimizer, SGD):
            optimizer.momentum = self.hparams.momentum

        if self.hparams.lr_scheduler == "cosine":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max=self.hparams.max_epochs,
            )
        elif self.hparams.lr_scheduler == "cosine_warmup":
            scheduler = LinearWarmupCosineAnnealingLR(
                optimizer,
                self.hparams.lr_warmup_epochs,
                self.hparams.max_epochs,
            )
        elif self.hparams.lr_scheduler == "linear":
            scheduler = LinearLR(
                optimizer,
                start_factor=1,
                end_factor=self.hparams.lr_drop,
                total_iters=self.hparams.max_epochs,
            )
        elif self.hparams.lr_scheduler == "step":
            scheduler = MultiStepLR(
                optimizer,
                self.hparams.lr_steps,
                gamma=self.hparams.lr_drop,
            )

        return [optimizer], [scheduler]
    
    def log_helper(self, names, values, add_dataloader_idx=False):
        """Compresses calls to self.log.

        Args:
            names: A list of metric names to log.
            values: A list of metric values to log.
            add_dataloader_idx: Whether to include the dataloader index in the name.
        """

        for idx, (name, value) in enumerate(zip(names, values)):
            self.log(
                name,
                value,
                on_step=(name in ("loss", "train_loss")),
                on_epoch=(name not in ("loss", "train_loss")),
                prog_bar=(name in ("train_acc", "val_loss", "val_acc")),
                sync_dist=True,
                add_dataloader_idx=add_dataloader_idx,
            )
    
    def log_helper2(self, names, values, dataloader_idx):
        """Calls log_helper as necessary for each DataLoader.
        
        Args:
           names: A list of metric names to log.
           values: A list of metric values to log.
           dataloader_idx: The index of the current DataLoader.
        """

        if dataloader_idx == 0:
            self.log_helper(names, values)

        try:
            # Errors if there is only 1 dataloader -- this means we
            # already logged it, so just pass.
            self.log_helper(names, values, add_dataloader_idx=True)
        except:
            pass

    def log_metrics(self, result, stage, dataloader_idx):
        """Logs metrics using the step results.
        
        Args:
            result: The output of self.step.
            stage: "train", "val", or "test".
            dataloader_idx: The index of the current dataloader.
        """

        names = []
        values = []
        for name in self.hparams.metrics:
            if name in result:
                # Class and group metrics must be manually collated.
                if "by_class" not in name and "by_group" not in name:
                    names.append(f"{stage}_{name}")
                    values.append(result[name])

        names.extend(["epoch", "step"])
        values.extend([float(self.current_epoch), float(self.trainer.global_step)])
        
        self.log_helper2(names, values, dataloader_idx)
    
    def collate_metrics(self, step_results, stage):
        """Collates and logs metrics by class and group.
        
        This is necessary because the logger does not utilize the info of how many samples
        from each class/group are in each batch, so logging class/group metrics on_epoch as
        usual will weight each batch the same instead of adjusting for totals.
        
        Args:
            step_results: List of dictionary results of self.validation_step or self.test_step.
            stage: "val" or "test".
        """
        
        def collate_and_sum(name):
            stacked = torch.stack([result[name] for result in step_results])
            return torch.sum(stacked, 0)

        dataloader_idx = step_results[0]["dataloader_idx"]

        if "acc_by_class" in self.hparams.metrics or \
          "acc5_by_class" in self.hparams.metrics or \
          "acc_by_group" in self.hparams.metrics or \
          "acc5_by_group" in self.hparams.metrics:
            names = []
            values = []
            total_by_class = collate_and_sum("total_by_class")
            total_by_group = collate_and_sum("total_by_group")

            if "acc_by_class" in self.hparams.metrics:
                acc_by_class = collate_and_sum("correct_by_class") / total_by_class
                names.extend([f"{stage}_acc_class{j}"
                              for j in range(len(acc_by_class))])
                values.extend(list(acc_by_class))
            if "acc5_by_class" in self.hparams.metrics:
                acc5_by_class = collate_and_sum("correct5_by_class") / total_by_class
                names.extend([f"{stage}_acc5_class{j}"
                              for j in range(len(acc5_by_class))])
                values.extend(list(acc5_by_class))
            if "acc_by_group" in self.hparams.metrics:
                acc_by_group = collate_and_sum("correct_by_group") / total_by_group
                names.extend([f"{stage}_acc_group{j}"
                              for j in range(len(acc_by_group))])
                values.extend(list(acc_by_group))
            if "acc5_by_group" in self.hparams.metrics:
                acc5_by_group = collate_and_sum("correct5_by_group") / total_by_group
                names.extend([f"{stage}_acc5_group{j}"
                              for j in range(len(acc5_by_group))])
                values.extend(list(acc5_by_group))

            self.log_helper2(names, values, dataloader_idx)
        
    def add_metrics_to_result(self, result, accs, dataloader_idx):
        """Adds dataloader_idx and metrics from compute_accuracy to result dict.
        
        Args:
            result: A dictionary containing the loss, prediction probabilities, and targets.
            accs: The output of compute_accuracy.
            dataloader_idx: The index of the current DataLoader.
        """
        
        result["dataloader_idx"] = dataloader_idx
        result["acc"] = accs["acc"]
        result["acc5"] = accs["acc5"]

        for k in ["class", "group"]:
            result[f"acc_by_{k}"] = accs[f"acc_by_{k}"]
            result[f"acc5_by_{k}"] = accs[f"acc5_by_{k}"]
            result[f"correct_by_{k}"] = accs[f"correct_by_{k}"]
            result[f"correct5_by_{k}"] = accs[f"correct5_by_{k}"]
            result[f"total_by_{k}"] = accs[f"total_by_{k}"]
    
    def step(self, batch, idx):
        """Performs a single step of prediction and loss calculation.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.

        Returns:
            A dictionary containing the loss, prediction probabilities, and targets.
        
        Raises:
            ValueError: Class weights are specified with MSE loss, or MSE loss
            is specified for a multiclass classification task.
        """

        inputs, orig_targets = batch

        # Removes extra targets (e.g., group index used for metrics).
        if orig_targets[0].ndim > 0:
            targets = orig_targets[:, 0]
        else:
            targets = orig_targets

        logits = self(inputs)

        # Ensures logits is a torch.Tensor.
        if isinstance(logits, (tuple, list)):
            logits = torch.squeeze(logits[0], dim=-1)

        # Initializes class weights if desired.
        weights = torch.ones(self.hparams.num_classes, device=logits.device)
        if self.hparams.class_weights:
            if self.hparams.loss == "mse":
                raise ValueError("Cannot use class weights with MSE.")
            weights = torch.Tensor(self.hparams.class_weights, device=logits.device)

        # Computes loss and prediction probabilities.
        if self.hparams.loss == "cross_entropy":
            if self.hparams.num_classes == 1:
                loss = F.binary_cross_entropy_with_logits(logits, targets, weight=weights)
                probs = torch.sigmoid(logits)
            else:
                loss = F.cross_entropy(logits, targets, weight=weights)
                probs = F.softmax(logits, dim=1)
        elif self.hparams.loss == "mse":
            if self.hparams.num_classes == 1:
                loss = F.mse_loss(logits, targets.float())
            elif self.hparams.num_classes == 2:
                loss = F.mse_loss(logits[:, 0], targets.float())
            else:
                raise ValueError("MSE is only an option for binary classification.")

        return {"loss": loss, "probs": probs, "targets": orig_targets}
    
    def step_and_log_metrics(self, batch, idx, dataloader_idx, stage):
        """Performs a step, then computes and logs metrics.
        
        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.
            stage: "train", "val", or "test".

        Returns:
            A dictionary containing the loss, prediction probabilities, targets, and metrics.
        """
        
        result = self.step(batch, idx)
        
        accs = compute_accuracy(
            result["probs"],
            result["targets"],
            self.hparams.num_classes,
            self.hparams.num_groups,
        )
        
        self.add_metrics_to_result(result, accs, dataloader_idx)

        self.log_metrics(result, stage, dataloader_idx)

        return result

    def training_step(self, batch, idx, dataloader_idx=0):
        """Performs a single training step.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.

        Returns:
            A dictionary containing the loss, prediction probabilities, targets, and metrics.
        """
        
        return self.step_and_log_metrics(batch, idx, dataloader_idx, "train")

    def training_epoch_end(self, training_step_outputs):
        """Collates metrics upon completion of the training epoch.
        
        Args:
            training_step_outputs: List of dictionary outputs of self.training_step.
        """
        
        self.collate_metrics(training_step_outputs, "train")

    
    def validation_step(self, batch, idx, dataloader_idx=0):
        """Performs a single validation step.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.

        Returns:
            A dictionary containing the loss, prediction probabilities, targets, and metrics.
        """

        return self.step_and_log_metrics(batch, idx, dataloader_idx, "val")
    
    def validation_epoch_end(self, validation_step_outputs):
        """Collates metrics upon completion of the validation epoch.
        
        Args:
            validation_step_outputs: List of dictionary outputs of self.validation_step.
        """
        
        self.collate_metrics(validation_step_outputs, "val")

    def test_step(self, batch, idx, dataloader_idx=0):
        """Performs a single test step.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.

        Returns:
            A dictionary containing the loss, prediction probabilities, targets, and metrics.
        """

        return self.step_and_log_metrics(batch, idx, dataloader_idx, "test")
    
    def test_epoch_end(self, test_step_outputs):
        """Collates metrics upon completion of the test epoch.
        
        Args:
            test_step_outputs: List of dictionary outputs of self.test_step.
        """
        
        self.collate_metrics(test_step_outputs, "test")

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        """Performs a single prediction step.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.
            dataloader_idx: The index of the current dataloader.
        """

        logits = self(batch)

        if self.hparams.num_classes == 1:
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int()
        else:
            probs = F.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

        return preds

