"""Defines configuration parameters for models and datamodules."""

# Imports Python packages.
from configargparse import Parser
from distutils.util import strtobool

# Import PyTorch packages.
from pytorch_lightning import Trainer

# Imports milkshake packages.
from milkshake.imports import valid_model_and_datamodule_names


def parse_args():
    """Reads configuration file and returns configuration dictionary."""

    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.set_defaults(strategy="ddp_find_unused_parameters_false")

    args = parser.parse_args()

    return args

def add_input_args(parser):
    """Loads configuration parameters into given configargparse.Parser."""

    model_names, datamodule_names = valid_model_and_datamodule_names()

    parser.add("--balanced_sampler", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether to use a class-balanced random sampler for training.")
    parser.add("--batch_size", type=int,
               help="The number of samples to include per batch.")
    parser.add("--bias", default=True, type=lambda x: bool(strtobool(x)),
               help="Whether to add a bias parameter to each model weight.")
    parser.add("--check_val_every_n_epochs", default=1, dest="check_val_every_n_epoch", type=int,
               help="The number of epochs after which to run validation.")
    parser.add("--check_val_every_n_steps", dest="val_check_interval", type=int,
               help="The number of steps after which to run validation.")
    parser.add("--ckpt_every_n_epochs", default=1, type=int,
               help="The number of epochs after which a model checkpoint will be saved.")
    parser.add("--ckpt_every_n_steps", type=int,
               help="The number of steps after which a model checkpoint will be saved.")
    parser.add("--class_weights", default=[], nargs="*", type=float,
               help="The amount by which to weight each class in the loss function.")
    parser.add("--data_augmentation", default=True, type=lambda x: bool(strtobool(x)),
               help="Whether to use data augmentation as specified in the DataModule.")
    parser.add("--data_dir", default="data",
               help="The name of the directory where data will be saved.")
    parser.add("--datamodule", choices=datamodule_names,
               help="The name of the DataModule to utilize.")
    parser.add("--dropout_prob", default=0, type=float,
               help="The probability by which a node will be dropped out.")
    parser.add("--eval_only", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether to skip training and only evaluate the model.")
    parser.add("--input_channels", default=3, type=int,
               help="The number of channels for image inputs.")
    parser.add("--label_noise", default=0, type=float,
               help="The probability by which class labels will be flipped.")
    parser.add("--loss", choices=["cross_entropy", "mse"], default="cross_entropy",
               help="The name of the loss function to utilize for optimization.")
    parser.add("--lr", type=float,
               help="The learning rate to utilize for optimization.") 
    parser.add("--lr_drop", default=0.1, type=float,
               help="The factor by which to drop the LR when using the step scheduler.")
    parser.add("--lr_scheduler", choices=["cosine", "cosine_warmup", "linear", "step"], default="step",
               help="The name of the LR scheduler to utilize.")
    parser.add("--lr_steps", default=[], nargs="*", type=int,
               help="A list of epochs to drop the LR when using the step scheduler.")
    parser.add("--lr_warmup_epochs", default=0, type=int,
               help="The number of epochs to warm up using certain schedulers.")
    parser.add("--metrics", nargs="+", default=["loss", "acc", "acc_by_class"],
               help=("Which metrics to compute and log. Choices include loss, acc, acc5, acc_by_class"
                     "acc5_by_class, and any custom metrics implemented in a DataModule."))
    parser.add("--model", choices=model_names,
               help="The name of the Model to utilize.")
    parser.add("--momentum", default=0.9, type=float,
               help="The momentum value to utilize with the SGD optimizer.")
    parser.add("--no_test", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether to skip the test phase (only train and val)")
    parser.add("--num_workers", default=4, type=int,
               help="The number of sub-processes to use for data loading.")
    parser.add("--optimizer", choices=["adam", "adamw", "sgd"], default="sgd",
               help="The name of the optimizer to utilize.")
    parser.add("--out_dir", default="out",
               help="The name of the directory where outputs will be saved.")
    parser.add("--persistent_workers", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether to use persistent workers (typically for ddp_spawn).")
    parser.add("--refresh_rate", default=1, type=int,
               help="The number of batches after which to update the progress bar.")
    parser.add("--resnet_small_input", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether to reduce the size of the first ResNet layer for small inputs.")
    parser.add("--resnet_l1_regularization", default=0, type=float,
               help="Whether to apply l1-norm regularization to the ResNet last layer.")
    parser.add("--resnet_pretrained", default=True, type=lambda x: bool(strtobool(x)),
               help="Whether to use pretrained weights in the ResNet.")
    parser.add("--resnet_version", choices=[18, 34, 50, 101, 152], default=18, type=int,
               help="The ResNet version to utilize.")
    parser.add("--resume_training", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether to resume the training state from the given checkpoint.")
    parser.add("--resume_weights", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether to load the weights from the given checkpoint.")
    parser.add("--seed", default=1, type=int,
               help="The random seed to utilize.")
    parser.add("--train_fc_only", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether to freeze all parameters except the last layer.")
    parser.add("--val_split", default=0.2, type=float,
               help="The proportion of training data to reserve for validation.")
    parser.add("--wandb", default=False, type=lambda x: bool(strtobool(x)),
               help="Whether to log with Weights and Biases (otherwise uses TensorBoard).")
    parser.add("--wandb_dir", default="wandb",
               help="The name of the directory where wandb outputs will be saved.")
    parser.add("--weights", default="",
               help="The filepath of the model checkpoint to load or resume.")
    parser.add("--weight_decay", default=1e-4, type=float,
               help="The l2-norm regularization to utilize during optimization.")
    
    return parser
