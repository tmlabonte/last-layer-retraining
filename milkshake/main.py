"""Main script for training, validation, and testing."""

# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

# Imports Python builtins.
from inspect import isclass
import os
import os.path as osp
import resource

# Imports Python packages.
import wandb
from PIL import ImageFile

# Imports PyTorch packages.
from lightning_lite.utilities.seed import seed_everything
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
import torch

# Imports milkshake packages.
from milkshake.args import parse_args
from milkshake.imports import valid_models_and_datamodules

# Sets matmul precision to take advantage of Tensor Cores.
torch.set_float32_matmul_precision("high")

# Prevents PIL from throwing invalid error on large image files.
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Prevents DataLoader memory error.
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8192, rlimit[1]))

# Silences wandb printouts.
os.environ["WANDB_SILENT"] = "true"


def load_weights(args, model):
    """Loads model weights or resumes training checkpoint.

    Args:
        args: The configuration dictionary.
        model: A model which inherits from milkshake.models.Model.

    Returns:
        The input model, possibly with the state dict loaded.
    """

    args.ckpt_path = None
    if args.weights:
        if args.resume_training:
            # Resumes training state (weights, optimizer, epoch, etc.) from args.weights.
            args.ckpt_path = args.weights
            print(f"Resuming training state from {args.weights}.")
        else:
            # Loads just the weights from args.weights.
            checkpoint = torch.load(args.weights, map_location="cpu")
            state_dict = checkpoint["state_dict"]
            model.load_state_dict(state_dict, strict=False)
            print(f"Weights loaded from {args.weights}.")     

    return model

def load_trainer(args, addtl_callbacks=None):
    """Loads PL Trainer for training and validation.

    Args:
        args: The configuration dictionary.
        addtl_callbacks: Desired callbacks besides ModelCheckpoint and TQDMProgressBar.

    Returns:
        An instance of pytorch_lightning.Trainer parameterized by args.
        
    Raises:
        ValueError: addtl_callbacks is not None and not a list.
    """

    # Checking validation by epochs and by steps are mutually exclusive.
    if args.val_check_interval:
        args.check_val_every_n_epoch = None

    # Checkpointing by epochs and by steps are mutually exclusive.
    if args.ckpt_every_n_steps:
        args.ckpt_every_n_epochs = None

    # Checkpoints model at the specified number of epochs.
    checkpointer1 = ModelCheckpoint(
        filename="{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}",
        save_top_k=-1,
        every_n_epochs=args.ckpt_every_n_epochs,
        every_n_train_steps=args.ckpt_every_n_steps,
    )

    # Checkpoints model with respect to validation loss.
    checkpointer2 = ModelCheckpoint(
        filename="best-{epoch:02d}-{val_loss:.3f}-{val_acc:.3f}",
        monitor="val_loss",
        every_n_epochs=args.ckpt_every_n_epochs,
        every_n_train_steps=args.ckpt_every_n_steps,
    )

    progress_bar = RichProgressBar(refresh_rate=args.refresh_rate)
 
    callbacks = [checkpointer1, checkpointer2, progress_bar]
    if addtl_callbacks is not None:
        if not isinstance(addtl_callbacks, list):
            raise ValueError("addtl_callbacks should be None or a list.")
        callbacks.extend(addtl_callbacks)

    logger = True # Activates TensorBoardLogger by default.
    if args.wandb:
        os.makedirs(args.wandb_dir, exist_ok=True)
        logger = WandbLogger(save_dir=args.wandb_dir, log_model="all")

    # Sets DDP strategy for multi-GPU training.
    # Important note: running multiple fit/validate/test loops using ddp (like
    # the Milkshake example experiments) is an anti-pattern in PyTorch Lightning
    # (see https://github.com/Lightning-AI/lightning/issues/8375). However, ddp
    # is much faster than the alternative, ddp_spawn. Therefore, we recommend
    # using ddp for single training jobs and ddp_spawn for experiments.
    args.strategy = args.strategy if args.devices > 1 else None

    trainer = Trainer.from_argparse_args(
        args,
        callbacks=callbacks,
        deterministic=True,
        logger=logger,
    )

    return trainer

def main(
    args,
    model_or_model_class,
    datamodule_or_datamodule_class,
    callbacks=None,
    model_hooks=None,
    verbose=True,
):
    """Main method for training and validation.

    Args:
        args: The configuration dictionary.
        model_or_model_class: A model or class which inherits from milkshake.models.Model.
        datamodule_or_datamodule_class: A datamodule or class which inherits from milkshake.datamodules.DataModule.
        callbacks: Any desired callbacks besides ModelCheckpoint and TQDMProgressBar.
        model_hooks: Any desired functions to run on the model before training.
        verbose: Whether to print the validation and test metrics.

    Returns:
        The trained model with its validation and test metrics.
    """

    os.makedirs(args.out_dir, exist_ok=True)
    args.devices = int(args.devices)

    # Sets global seed for reproducibility.
    seed_everything(seed=args.seed, workers=True)

    datamodule = datamodule_or_datamodule_class
    if isclass(datamodule):
        datamodule = datamodule(args)

    # Sets persistent workers when using ddp_spawn. Must be set here because a
    # DataModule may be passed as an arg to main prior to ddp initialization.
    if args.devices > 1 and \
       "ddp_spawn" in args.strategy and \
       not datamodule.persistent_workers:
        datamodule.persistent_workers = True

    args.num_classes = datamodule.num_classes
    args.num_groups = datamodule.num_groups

    model = model_or_model_class
    if isclass(model):
        model = model(args)
        
    model = load_weights(args, model)

    if model_hooks:
        for hook in model_hooks:
            hook(model)

    trainer = load_trainer(args, addtl_callbacks=callbacks)
    if not args.eval_only:
        trainer.fit(model, datamodule=datamodule, ckpt_path=args.ckpt_path)

    # Reloads trainer for validation and testing. Must use a single device.
    # Note that this procedure resets the "epoch" and "step" trainer metrics.
    orig_devices = args.devices
    orig_strategy = args.strategy
    if args.devices > 1 and trainer.is_global_zero:
        if "ddp_spawn" not in args.strategy:
            torch.distributed.destroy_process_group()
        args.devices = 1
        trainer = load_trainer(args, addtl_callbacks=callbacks)

    val_metrics = trainer.validate(model, datamodule=datamodule, verbose=verbose)
    test_metrics = trainer.test(model, datamodule=datamodule, verbose=verbose)
    args.devices = orig_devices
    args.strategy = orig_strategy

    # Closes wanbd instance (for experiments which run main() many times).
    # Also cleans up wandb cache by deleting files down to a 10GB limit.
    if args.wandb:
        wandb.finish()
        cache = wandb.sdk.artifacts.artifacts_cache.get_artifacts_cache()
        cache.cleanup(int(10e10))
    
    return model, val_metrics, test_metrics


if __name__ == "__main__":
    args = parse_args()
    
    models, datamodules = valid_models_and_datamodules()

    main(args, models[args.model], datamodules[args.datamodule])

