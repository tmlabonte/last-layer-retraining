"""Main file for model distillation experiments."""

# Ignores nuisance pl_bolts warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

# Imports Python packages.
from configargparse import Parser

# Imports PyTorch packages.
from pytorch_lightning import Trainer
import torch
from torch import nn
import torch.nn.functional as F

# Imports milkshake packages.
from milkshake.args import add_input_args
from milkshake.datamodules.cifar10 import CIFAR10
from milkshake.main import main
from milkshake.models.cnn import CNN
from milkshake.models.resnet import ResNet
from milkshake.utils import get_weights


def check_type(logits):
    """Ensures logits is a torch.Tensor."""
    
    if isinstance(logits, (tuple, list)):
        return torch.squeeze(logits[0], dim=-1)
    return logits

class StudentCNN(CNN):
    """Class for a CNN which learns via distillation from the logits of a teacher ResNet."""
    
    def __init__(self, args):
        super().__init__(args)

        # Loads teacher ResNet and freezes parameters.
        self.teacher = ResNet(args)
        state_dict = torch.load(args.teacher_weights, map_location="cpu")["state_dict"]
        self.teacher.load_state_dict(state_dict, strict=False)

        self.teacher.eval()
        for p in self.teacher.parameters():
            p.requires_grad = False

    def load_msg(self):
        return super().load_msg()[:-1] + " with distillation from a teacher ResNet."
            
    def step(self, batch, idx):
        """Performs a single step of prediction and loss calculation.
        
        For distillation, we compute the MSE loss between the student and teacher logits.

        Args:
            batch: A tuple containing the inputs and targets as torch.Tensor.
            idx: The index of the given batch.

        Returns:
            A dictionary containing the loss, prediction probabilities, and targets.
        """

        inputs, targets = batch

        logits = check_type(self(inputs))

        with torch.no_grad():
            teacher_logits = check_type(self.teacher(inputs))

        # Computes MSE loss between student and teacher logits.
        loss = F.mse_loss(logits, teacher_logits)

        probs = F.softmax(logits, dim=1)

        return {"loss": loss, "probs": probs, "targets": targets}

def pct(x):
    return round(x, 3) * 100

def experiment(args):
    cifar10 = CIFAR10(args)

    # Trains a teacher ResNet.
    model, _, resnet_metrics  = main(args, ResNet, cifar10)

    teacher_version = model.trainer.logger.version
    args.teacher_weights = get_weights(args, teacher_version, idx=-1)
    del model

    args.max_epochs = args.cnn_epochs
    args.lr = args.cnn_lr
    args.check_val_every_n_epoch = args.cnn_epochs // 10
    args.ckpt_every_n_epoch = args.cnn_epochs

    # Trains a CNN baseline.
    _, _, cnn_metrics = main(args, CNN, cifar10)

    args.max_epochs = args.distillation_epochs
    args.lr = args.distillation_lr

    # Trains a CNN with distillation from the ResNet.
    _, _, dist_metrics = main(args, StudentCNN, cifar10)
    
    print("---Experiment Results---")
    print("\nTeacher ResNet")
    print(f"Accuracy: {pct(resnet_metrics[0]['test_acc'])}")
    print("\nCNN Baseline")
    print(f"Accuracy: {pct(cnn_metrics[0]['test_acc'])}")
    print("\nCNN with ResNet distillation")
    print(f"Accuracy: {pct(dist_metrics[0]['test_acc'])}")


if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add("--cnn_epochs", default=100, type=int)
    parser.add("--cnn_lr", default=0.1, type=float)
    parser.add("--distillation_epochs", default=100, type=float)
    parser.add("--distillation_lr", default=0.02, type=float)

    args = parser.parse_args()
    experiment(args)

