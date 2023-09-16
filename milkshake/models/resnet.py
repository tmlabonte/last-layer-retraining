"""ResNet model implementation."""

# Imports PyTorch packages.
import torch
from torch import nn
import torchvision.models as models

# Imports milkshake packages.
from milkshake.models.model import Model


class ResNet(Model):
    """ResNet model implementation."""

    def __init__(self, args):
        """Initializes a ResNet model.

        Args:
            args: The configuration dictionary.
        """

        super().__init__(args)

        resnets = {
            18: models.resnet18,
            34: models.resnet34,
            50: models.resnet50,
            101: models.resnet101,
            152: models.resnet152,
        }

        # TODO: Include different ResNet pretraining options.
        if args.resnet_pretrained:
            self.model = resnets[args.resnet_version](weights="IMAGENET1K_V1")
        else:
            self.model = resnets[args.resnet_version](weights=None)

        # Reduces the kernel size and stride for smaller inputs, e.g., CIFAR-10
        # images (32 x 32) instead of ImageNet images (224 x 224).
        if args.resnet_small_input:
            self.model.conv1 = nn.Conv2d(
                args.input_channels,
                64,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            )
            self.model.maxpool = nn.Identity()
        elif args.input_channels != 3:
            self.model.conv1 = nn.Conv2d(
                args.input_channels,
                64,
                kernel_size=7,
                stride=2,
                padding=3,
                bias=False,
            )

        self.model.fc = nn.Sequential(
            nn.Dropout(p=args.dropout_prob),
            nn.Linear(self.model.fc.in_features, args.num_classes),
        )

        # Freezes all parameters except those in the last layer.
        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.fc.parameters():
                p.requires_grad = True

    def load_msg(self):
        if self.hparams.resnet_pretrained:
            return f"Loading ImageNet1K-pretrained ResNet{self.hparams.resnet_version}."
        else:
            return f"Loading ResNet{self.hparams.resnet_version} with no pretraining."

    def step(self, batch, idx):
        """Performs a single step of prediction and loss calculation.

        Args:
            batch: A tuple containing the inputs and targets as torch.tensor.
            idx: The index of the given batch.

        Returns:
            A dictionary containing the loss, prediction probabilities, and targets.
            The probs and targets are moved to CPU to free up GPU memory.
        """

        result = super().step(batch, idx)

        # Optionally adds regularization penalizing the l1 norm of model parameters.
        if self.hparams.resnet_l1_regularization:
            if self.hparams.train_fc_only:
                params = torch.cat([param.view(-1) for param in self.model.fc.parameters()])
            else:
                params = torch.cat([param.view(-1) for param in self.model.parameters()])

            param_l1_norm = torch.linalg.vector_norm(params, ord=1)
            result["loss"] += self.hparams.resnet_l1_regularization * param_l1_norm

        return result

