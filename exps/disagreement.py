"""Main file for last-layer retraining experimentation."""

# Ignores nuisance warnings. Must be called first.
from milkshake.utils import ignore_warnings
ignore_warnings()

# Imports Python builtins.
from copy import deepcopy
from distutils.util import strtobool
import os
import os.path as osp
import pickle
import sys

# Imports Python packages.
from configargparse import Parser
import numpy as np

# Imports PyTorch packages.
from pytorch_lightning import Trainer

# Imports milkshake packages.
from milkshake.args import add_input_args
from milkshake.datamodules.celeba import CelebA
from milkshake.datamodules.civilcomments import CivilComments
from milkshake.datamodules.disagreement import Disagreement
from milkshake.datamodules.multinli import MultiNLI
from milkshake.datamodules.waterbirds import Waterbirds
from milkshake.imports import valid_models_and_datamodules
from milkshake.main import load_weights, main
from milkshake.utils import get_weights


class WaterbirdsDisagreement(Waterbirds, Disagreement):
    """DataModule for the WaterbirdsDisagreement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

class CelebADisagreement(CelebA, Disagreement):
    """DataModule for the CelebADisagreeement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

class CivilCommentsDisagreement(CivilComments, Disagreement):
    """DataModule for the CivilCommentsDisagreement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)
        
class MultiNLIDisagreement(MultiNLI, Disagreement):
    """DataModule for the MultiNLIDisagreement dataset."""

    def __init__(self, args, **kwargs):
        super().__init__(args, **kwargs)

def set_training_parameters(args):
    if args.datamodule == "waterbirds":
        args.datamodule_class = WaterbirdsDisagreement
        args.num_classes = 2
        args.retrain_epochs = 100
        args.finetune_steps = 250
        args.finetune_lrs = [1e-4, 1e-3, 1e-2]
    elif args.datamodule == "celeba":
        args.datamodule_class = CelebADisagreement
        args.num_classes = 2
        args.retrain_epochs = 100
        args.finetune_steps = 250
        args.finetune_lrs = [1e-4, 1e-3, 1e-2]
    elif args.datamodule == "civilcomments":
        args.datamodule_class = CivilCommentsDisagreement
        args.num_classes = 2
        args.retrain_epochs = 10
        args.finetune_steps = 500
        args.finetune_lrs = [1e-6, 1e-5, 1e-4]
    elif args.datamodule == "multinli":
        args.datamodule_class = MultiNLIDisagreement
        args.num_classes = 3
        args.retrain_epochs = 10
        args.finetune_steps = 500
        args.finetune_lrs = [1e-6, 1e-5, 1e-4]
    else:
        raise ValueError(f"DataModule {args.datamodule} not supported.")

    args.finetune_num_datas = [20, 100, 500]
    args.dropout_probs = [0.5, 0.7, 0.9]
    args.early_stop_nums = [1, 2, 5]

def load_erm():
    if osp.isfile("erm.pkl"):
        with open("erm.pkl", "rb") as f:
            erm = pickle.load(f)
    else: 
        datasets = ["waterbirds", "celeba", "civilcomments", "multinli"]
        seeds = [1, 2, 3]
        balance = [True, False]
        split = ["train", "combined"]
        train_pct = [80, 85, 90, 95, 100]

        erm = {}
        for d in datasets:
            erm[d] = {}
            for s in seeds:
                erm[d][s] = {}
                for b in balance:
                    erm[d][s][b] = {}
                    for p in split:
                        erm[d][s][b][p] = {}
                        for t in train_pct:
                            erm[d][s][b][p][t] = {"version": -1, "metrics": []}

        with open("erm.pkl", "wb") as f:
            pickle.dump(erm, f)

    return erm

def dump_erm(args, curr_erm):
    erm = load_erm()
    erm[args.datamodule][args.seed][args.balance_erm][args.split][args.train_pct] = curr_erm

    with open("erm.pkl", "wb") as f:
        pickle.dump(erm, f)

def reset_fc_hook(model):
    try:
        for layer in model.model.fc:
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
    except:
        model.model.fc.reset_parameters()

def print_metrics(metrics):
    val_metrics, test_metrics = metrics

    val_group_accs = [
        acc for name, acc in val_metrics[0].items() if "group" in name
    ]
    val_avg_acc = val_metrics[0]["val_acc"]
    val_avg_acc = round(val_avg_acc * 100, 1)
    val_worst_group_acc = min(val_group_accs) 
    val_worst_group_acc = round(val_worst_group_acc * 100, 1)
    print(f"Val Average Acc: {val_avg_acc}")
    print(f"Val Worst Group Acc: {val_worst_group_acc}")

    if test_metrics:
        test_group_accs = [
            acc for name, acc in test_metrics[0].items() if "group" in name
        ]
        test_avg_acc = test_metrics[0]["test_acc"]
        test_avg_acc = round(test_avg_acc * 100, 1)
        test_worst_group_acc = min(test_group_accs) 
        test_worst_group_acc = round(test_worst_group_acc * 100, 1)
        print(f"Test Average Acc: {test_avg_acc}")
        print(f"Test Worst Group Acc: {test_worst_group_acc}")

    print()

def print_results(erm_metrics, results, keys):
    print("---Experiment Results---")
    print("\nERM")
    print_metrics(erm_metrics)

    for key in keys:
        print(key.title())
        if "self" in key:
            print(f"Best params: {results[key]['params']}")
        print_metrics(results[key]["metrics"])

def finetune_last_layer(
    args,
    finetune_type,
    model_class,
    dropout_prob=0,
    early_stop_weights=None,
    finetune_num_data=None,
    worst_group_pct=None,
):
    disagreement_args = deepcopy(args)
    disagreement_args.finetune_type = finetune_type
    disagreement_args.dropout_prob = dropout_prob

    finetune_args = deepcopy(args)
    finetune_args.train_fc_only = True
    finetune_args.lr_scheduler = "step"
    finetune_args.lr_steps = []
    finetune_args.lr = args.finetune_lr

    # Sets parameters for finetuning (first) or retraining (second).
    if finetune_num_data:
        finetune_args.max_epochs = None
        finetune_args.max_steps = args.finetune_steps
        reset_fc = False
    else:
        finetune_args.max_epochs = args.retrain_epochs
        finetune_args.max_steps = -1
        reset_fc = True

    # Don't save the model (we save manually if it is the best).
    finetune_args.val_check_interval = args.finetune_steps + 1
    finetune_args.ckpt_every_n_steps = args.finetune_steps + 1

    model = model_class(disagreement_args)
    load_weights(disagreement_args, model)

    early_stop_model = None
    if early_stop_weights:
        early_args = deepcopy(disagreement_args)
        early_args.weights = early_stop_weights
        early_stop_model = model_class(early_args)
        load_weights(early_args, early_stop_model)

    datamodule = args.datamodule_class(
        disagreement_args,
        early_stop_model=early_stop_model,
        model=model,
        num_data=finetune_num_data,
        worst_group_pct=worst_group_pct,
    )

    model_hooks = [reset_fc_hook] if reset_fc else None
    model, val_metrics, test_metrics = main(
        finetune_args,
        model_class,
        datamodule,
        model_hooks=model_hooks,
    )

    return model, val_metrics, test_metrics

def experiment(args, model_class):
    os.makedirs("out", exist_ok=True)

    # Loads ERM paths and metrics from pickle file.
    erm = load_erm()
    
    # Adds experiment-specific parameters to args.
    set_training_parameters(args)

    # Trains ERM model.
    curr_erm = erm[args.datamodule][args.seed][args.balance_erm][args.split][args.train_pct]
    erm_version = curr_erm["version"]
    erm_metrics = curr_erm["metrics"]
    if erm_version == -1:
        args.balanced_sampler = True if args.balance_erm else False
        model, erm_val_metrics, erm_test_metrics = main(args, model_class, args.datamodule_class)
        args.balanced_sampler = False

        erm_version = model.trainer.logger.version
        erm_metrics = [erm_val_metrics, erm_test_metrics]

        curr_erm["version"] = erm_version
        curr_erm["metrics"] = erm_metrics
        dump_erm(args, curr_erm)
        del model

    elif not erm_metrics:
        args.weights = get_weights(args, erm_version, idx=-1)
        args.eval_only = True
        _, erm_val_metrics, erm_test_metrics = main(args, model_class, args.datamodule_class)
        args.eval_only = False

        erm_metrics = [erm_val_metrics, erm_test_metrics]
        curr_erm["metrics"] = erm_metrics
        dump_erm(args, curr_erm)

    def print_results2(results, keys):
        return print_results(erm_metrics, results, keys)

    print("ERM")
    print_metrics(erm_metrics)

    # When these two arguments are passed, the entire held-out set
    # (except the actual validation set) is used for training. So,
    # there is no data left for last-layer retraining, and we return.
    if args.split == "combined" and args.train_pct == 100:
        return

    # Gets last-epoch ERM weights.
    args.weights = get_weights(args, erm_version, idx=-1)

    # Sets finetune types. Note that "group-unbalanced retraining" will be
    # either class-unbalanced or class-balanced based on the value
    # of args.balanced_sampler.
    finetune_types = [
        "group-unbalanced retraining",
        "group-balanced retraining",
        "random self",
        "misclassification self",
        "early-stop misclassification self",
        "dropout disagreement self",
        "early-stop disagreement self",
    ]

    # Prepares results dictionary.
    results = {f: {"val_worst_group_acc": -1, "metrics": [], "params": []}
               for f in finetune_types}

    def finetune_helper(
        finetune_type,
        dropout_prob=0,
        early_stop_num=None,
        finetune_lr=None,
        finetune_num_data=None,
        worst_group_pct=None,
    ):
        args.finetune_lr = finetune_lr if finetune_lr else args.lr

        param_str = ""
        if finetune_num_data:
            param_str += f" Num Data {finetune_num_data}"
        param_str += f" LR {args.finetune_lr}"
        if dropout_prob:
            param_str += f" Dropout {dropout_prob}"
        if early_stop_num:
            param_str += f" Early Stop {early_stop_num}"
        if worst_group_pct:
            param_str += f" Worst-group Pct {worst_group_pct}"
        print(f"{finetune_type.title()}{param_str}")

        early_stop_weights = None
        if early_stop_num:
            early_stop_weights = get_weights(args, erm_version, idx=early_stop_num-1)
        model, val_metrics, test_metrics = finetune_last_layer(
            args,
            finetune_type,
            model_class,
            dropout_prob=dropout_prob,
            early_stop_weights=early_stop_weights,
            finetune_num_data=finetune_num_data,
            worst_group_pct=worst_group_pct,
        )

        print_metrics([val_metrics, test_metrics])

        val_worst_group_acc = min([
            acc for name, acc in val_metrics[0].items() if "group" in name
        ])

        if val_worst_group_acc >= results[finetune_type]["val_worst_group_acc"]:
            results[finetune_type]["val_worst_group_acc"] = val_worst_group_acc
            results[finetune_type]["metrics"] = [val_metrics, test_metrics]

            params = []
            if finetune_num_data:
                params.append(finetune_num_data)
            if finetune_lr:
                params.append(finetune_lr)
            if dropout_prob:
                params.append(dropout_prob)
            if early_stop_num:
                params.append(early_stop_num)
            results[finetune_type]["params"] = params

            display_type = finetune_type.replace(" ", "_").replace("-", "_")
            best_path = f"out/best_{args.datamodule}{args.seed}_{display_type}.ckpt"
            model.trainer.save_checkpoint(best_path)

            return params
        
        return None

    # Performs worst-group data ablation.
    if args.worst_group_ablation:
        for worst_group_pct in [2.5, 5] + [12.5 * j for j in range(1, 9)]:
            finetune_helper("class-balanced retraining", worst_group_pct=worst_group_pct)
        return

    # Performs early-stop disagreement demo.
    if args.demo:
        finetune_helper(
            "early-stop disagreement self",
            early_stop_num=5,
            finetune_lr=1e-2,
            finetune_num_data=500,
        )
        print_results2(results, finetune_types[-1:])
        return

    # Performs last-layer retraining.
    finetune_helper("group-unbalanced retraining")
    finetune_helper("group-balanced retraining")

    if args.no_self:
        print_results2(results, finetune_types[:2])
        return

    # Perform SELF hyperparameter search using worst-group validation accuracy.
    for finetune_type in finetune_types[3:]:
        best_params = None
        args.no_test = True

        for finetune_num_data in args.finetune_num_datas:
            for finetune_lr in args.finetune_lrs:
                if finetune_type == "random self":
                    params = finetune_helper(
                        "random self",
                        finetune_lr=finetune_lr,
                        finetune_num_data=finetune_num_data,
                    )
                    if params is not None:
                        best_params = params
                if finetune_type == "misclassification self":
                    params = finetune_helper(
                        "misclassification self",
                        finetune_lr=finetune_lr,
                        finetune_num_data=finetune_num_data,
                    )
                    if params is not None:
                        best_params = params
                elif finetune_type == "early-stop misclassification self":
                    for early_stop_num in args.early_stop_nums:
                        params = finetune_helper(
                            "early-stop misclassification self",
                            early_stop_num=early_stop_num,
                            finetune_lr=finetune_lr,
                            finetune_num_data=finetune_num_data,
                        )
                        if params is not None:
                            best_params = params
                elif finetune_type == "dropout disagreement self":
                    for dropout_prob in args.dropout_probs:
                        params = finetune_helper(
                            "dropout disagreement self",
                            dropout_prob=dropout_prob,
                            finetune_lr=finetune_lr,
                            finetune_num_data=finetune_num_data,
                        )
                        if params is not None:
                            best_params = params
                elif finetune_type == "early-stop disagreement self":
                    for early_stop_num in args.early_stop_nums:
                        params = finetune_helper(
                            "early-stop disagreement self",
                            early_stop_num=early_stop_num,
                            finetune_lr=finetune_lr,
                            finetune_num_data=finetune_num_data,
                        )
                        if params is not None:
                            best_params = params

        args.eval_only = True
        args.no_test = False
        display_type = finetune_type.replace(" ", "_").replace("-", "_")
        args.weights = f"out/best_{args.datamodule}{args.seed}_{display_type}.ckpt"
        finetune_helper(
            finetune_type,
            dropout_prob=best_params[2] if "dropout" in finetune_type else 0,
            early_stop_num=best_params[2] if "early-stop" in finetune_type else None,
            finetune_lr=best_params[1],
            finetune_num_data=best_params[0],
        )

        if not args.save_self_model:
            os.remove(args.weights)
        args.weights = get_weights(args, erm_version, idx=-1)
        args.eval_only = False
        args.no_test = True

    print_results2(results, finetune_types)

    
if __name__ == "__main__":
    parser = Parser(
        args_for_setting_config_path=["-c", "--cfg", "--config"],
        config_arg_is_required=True,
    )

    parser = add_input_args(parser)
    parser = Trainer.add_argparse_args(parser)
    
    parser.add("--balance_erm", default=True, type=lambda x: bool(strtobool(x)),
               help="Whether to perform class-balancing during ERM training.")
    parser.add("--balance_finetune", choices=["sampler", "subset", "none"], default="sampler",
               help="Which type of class-balancing to perform during finetuning.")
    parser.add("--demo", action="store_true",
               help="Whether to run a quick demo of early-stop disagreement SELF.")
    parser.add("--no_self", action="store_true",
               help="Whether to perform only ERM and last-layer retraining, no SELF.")
    parser.add("--save_self_model", action="store_true",
               help="Whether to save the SELF model outputs.")
    parser.add("--split", choices=["combined", "train"], default="train",
               help="The split to train on; either the train set or the combined train and held-out set.")
    parser.add("--train_pct", default=100, type=int,
               help="The percentage of the train set to utilize (for ablations)")
    parser.add("--val_pct", default=100, type=int,
               help="The percentage of the val set to utilize (for ablations)")
    parser.add("--worst_group_ablation", action="store_true",
               help="Whether to perform an ablation on the amount of worst-group data used during retraining.")

    args = parser.parse_args()

    models, _ = valid_models_and_datamodules()
        
    experiment(args, models[args.model])

