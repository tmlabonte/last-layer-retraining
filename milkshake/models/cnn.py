"""Convolutional neural network (CNN) model implementation."""

# Imports PyTorch packages.
from torch import nn

# Imports milkshake packages.
from milkshake.models.model import Model


class CNN(Model):
    """CNN model implementation.
    
    This version has conv layers of doubling width followed by a linear layer.
    """

    def __init__(self, args):
        """Initializes a CNN model.
        
        Args:
            args: The configuration dictionary.
        """

        super().__init__(args)
        
        self.model = nn.Sequential()
        
        doubles = [2 ** i for i in range(args.cnn_num_layers - 1)]
        h = [args.cnn_initial_width * j for j in doubles]
        
        channels = zip([args.input_channels] + h[:-1], h)
        for j, (n, k) in enumerate(channels):
            self.model.append(nn.Conv2d(
                n, k,
                args.cnn_kernel_size,
                bias=args.bias,
                padding=args.cnn_padding,
            ))

            if args.cnn_batchnorm:
                self.model.append(nn.BatchNorm2d(k))

            self.model.append(nn.ReLU())
            
            if j != 0:
                self.model.append(nn.MaxPool2d(2))
        
        self.model.append(nn.MaxPool2d(4))
        self.model.append(nn.Flatten())
        self.model.append(nn.LazyLinear(args.num_classes, bias=args.bias))

        # Freezes all parameters except those in the last layer.
        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model[-1].parameters():
                p.requires_grad = True

    def load_msg(self):
        return (
            f"Loading CNN with {self.hparams.cnn_num_layers} layers"
            f" and initial width {self.hparams.cnn_initial_width}."
        )

