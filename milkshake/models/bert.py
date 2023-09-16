"""BERT model implementation."""

# Imports Python builtins.
import types

# Imports Python packages.
from transformers import BertForSequenceClassification, get_scheduler

# Imports PyTorch packages.
from torch.optim import SGD

# Imports milkshake packages.
from milkshake.models.model import Model


class BERT(Model):
    """BERT model implementation."""

    def __init__(self, args):
        """Initializes a BERT model.

        Args:
            args: The configuration dictionary.
        """

        super().__init__(args)

        self.model = BertForSequenceClassification.from_pretrained(
            "bert-base-uncased",
            num_labels=args.num_classes,
        )

        self.model.fc = self.model.classifier
        delattr(self.model, "classifier")

        def classifier(self, x):
            return self.fc(x)
        
        self.model.classifier = types.MethodType(classifier, self.model)
        self.model.base_forward = self.model.forward

        def forward(self, x):
            return self.base_forward(
                input_ids=x[:, :, 0],
                attention_mask=x[:, :, 1],
                token_type_ids=x[:, :, 2]).logits

        self.model.forward = types.MethodType(forward, self.model)

        # Freezes all parameters except those in the last layer.
        if args.train_fc_only:
            for p in self.model.parameters():
                p.requires_grad = False
            for p in self.model.fc.parameters():
                p.requires_grad = True

    def load_msg(self):
        return f"Loading BERT Base Uncased pretrained on Book Corpus and English Wikipedia."

    def configure_optimizers(self):
        """Returns the optimizer and learning rate scheduler."""

        no_decay = ["bias", "LayerNorm.weight"]
        decay_params = []
        nodecay_params = []
        for n, p in self.model.named_parameters():
            if not any(nd in n for nd in no_decay):
                decay_params.append(p)
            else:
                nodecay_params.append(p)

        optimizer_grouped_parameters = [
            {
                "params": decay_params,
                "weight_decay": self.hparams.weight_decay,
            },
            {
                "params": nodecay_params,
                "weight_decay": 0.0,
            },
        ]

        optimizer = self.optimizer(
            optimizer_grouped_parameters,
            lr=self.hparams.lr,
        )
        if isinstance(optimizer, SGD):
            optimizer.momentum = self.hparams.momentum

        # Only the linear scheduler and no scheduler are currently implemented.
        if self.hparams.lr_scheduler == "linear":
            scheduler = get_scheduler(
                "linear",
                optimizer=optimizer,
                num_warmup_steps=self.hparams.lr_warmup_epochs,
                num_training_steps=self.hparams.max_epochs,
            )

            return [optimizer], [scheduler]
        elif self.hparams.lr_scheduler == "step" and self.hparams.lr_steps == []:
            return optimizer
        else:
            raise NotImplementedError()

