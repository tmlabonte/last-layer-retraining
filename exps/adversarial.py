"""Main file for adversarial training experiments with PGD attacks."""

# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

# Imports Python packages.
from configargparse import Parser

# Imports PyTorch packages.
from pytorch_lightning import Trainer
import torch
import torch.nn.functional as F

# Imports milkshake packages.
from milkshake.args import add_input_args
from milkshake.datamodules.cifar10 import CIFAR10
from milkshake.main import main
from milkshake.models.resnet import ResNet
from milkshake.utils import compute_accuracy


class PGDAttack():
    """Class for computing adversarial examples with PGD."""
    
    def __init__(self, args, model):
        self.model = model

        self.alpha = args.alpha
        self.epsilon = args.epsilon
        self.pgd_steps = args.pgd_steps

    def perturb(self, x_natural, y):
        """Computes adversarial example via projected gradient descent.
        
        Uses the parameters of self.model to maximally increase the loss
        on the given datapoint. Runs PGD for as many steps as specified.
        
        Args:
            x_natural: The original input.
            y: The original label.
        
        Returns:
            The PGD adversarial example.
        """
        
        with torch.inference_mode(False):
            y = torch.clone(y)
            x = x_natural.detach()
            x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)

            # Performs gradient steps to adversarially perturb input.
            for i in range(self.pgd_steps):
                x.requires_grad_()
                with torch.enable_grad():
                    logits = self.model(x)
                    loss = F.cross_entropy(logits, y)
                    loss.requires_grad_()

                grad = torch.autograd.grad(loss, [x])[0]
                x = x.detach() + self.alpha * torch.sign(grad.detach())
                x = torch.min(
                    torch.max(x, x_natural - self.epsilon),
                    x_natural + self.epsilon,
                )
                x = torch.clamp(x, 0, 1)

        return x

class AdversarialResNet(ResNet):
    """Class for ResNet adversarial training."""
    
    def __init__(self, args):
        super().__init__(args)

    def adv_step(self, batch, idx):
        inputs, targets = batch

        # Computes adversarial examples.
        adversary = PGDAttack(self.hparams, self.model)
        adv = adversary.perturb(inputs, targets)

        # Computes loss on adversarial examples.
        adv_outputs = self.model(adv)
        loss = F.cross_entropy(adv_outputs, targets)
        probs = F.softmax(adv_outputs, dim=1)

        return {"loss": loss, "probs": probs, "targets": targets}
    
    def step_and_log_metrics(self, batch, idx, dataloader_idx, stage):
        if stage == "train":
            if self.hparams.adversarial_training:
                result = None
                adv_result = self.adv_step(batch, idx)
            else:
                result = self.step(batch, idx)
                adv_result = None
        else:
            # Runs both regular and adversarial steps
            # for validation and testing only.
            result = self.step(batch, idx)
            adv_result = self.adv_step(batch, idx)
        
        # Computes and logs relevant metrics.
        if result:
            accs = compute_accuracy(
                result["probs"],
                result["targets"],
                self.hparams.num_classes,
                self.hparams.num_groups,
            )
        
        if adv_result:
            adv_accs = compute_accuracy(
                adv_result["probs"],
                adv_result["targets"],
                self.hparams.num_classes,
                self.hparams.num_groups,
            )
            
        if result:
            result["acc"] = accs["acc"]
            result["dataloader_idx"] = dataloader_idx
            if adv_result:
                result["adv_loss"] = adv_result["loss"]
                result["adv_acc"] = adv_accs["acc"]
        else:
            result = adv_result
            result["dataloader_idx"] = dataloader_idx
            result["acc"] = adv_accs["acc"]
            result["adv_loss"] = result["loss"]
            result["adv_acc"] = adv_accs["acc"]

        self.log_metrics(result, stage, dataloader_idx)

        return result

def pct(x):
    return round(x, 3) * 100

def experiment(args):
    cifar10 = CIFAR10(args)

    # Trains ERM baseline.
    args.adversarial_training = False
    _, _, baseline_metrics = main(args, AdversarialResNet, cifar10)

    # Trains ResNet with adversarial training.
    args.adversarial_training = True
    _, _, adv_metrics = main(args, AdversarialResNet, cifar10)
    
    print("---Experiment Results---")
    print("\nERM Baseline")
    print(f"Standard Accuracy: {pct(baseline_metrics[0]['test_acc'])}")
    print(f"Adversarial Accuracy: {pct(baseline_metrics[0]['test_adv_acc'])}")
    print("\nAdversarial Training")
    print(f"Standard Accuracy: {pct(adv_metrics[0]['test_acc'])}")
    print(f"Adversarial Accuracy: {pct(adv_metrics[0]['test_adv_acc'])}")


if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    parser.add("--alpha", default=0.01, type=float)
    parser.add("--epsilon", default=0.03, type=float)
    parser.add("--pgd_steps", default=7, type=int)

    args = parser.parse_args()
    experiment(args)

