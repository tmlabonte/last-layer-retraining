# Milkshake
## Quick and extendable experimentation with classification models

Milkshake is my core repository for ML research designed around rapid prototyping and simple implementations of research necessities. It features a modular and object-oriented structure which decouples the training workflow from the research code, enabling complex experiments to be written in under 200 lines of code. Milkshake is written in PyTorch 1.12 with PyTorch Lightning 1.9, and to me represents the fulfillment of the Lightning 1.x vision of rapid research without the boilerplate.

Note that milkshake is a GitHub template: it is intended for use as a foundation which makes it easy to add your own research code. Currently, there are 8 datasets and 5 models implemented across both vision and language tasks; contributions of models and datasets are welcomed!

### Installation
```
conda update -n base -c defaults conda
conda create -n milkshake python==3.10
conda activate milkshake
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
python -m pip install -e .
```

### Instructions
The `milkshake` folder contains the base code and is written in PyTorch 1.12 with PyTorch Lightning 1.9. Three important files are `milkshake/main.py`, which runs experiments, `milkshake/datamodules/datamodule.py`, which includes data processing and loading, and `milkshake/models/model.py`, which includes model training and inference. These files typically should not need to be modified for experimentation, unless a new basic functionality is being added.

The `cfgs` folder contains configuration files in the `yaml` language which specify training and model parameters. In addition to the options in `milkshake/args.py`, all [PyTorch Lightning 1.9 Trainer flags](https://pytorch-lightning.readthedocs.io/en/1.9.5/common/trainer.html#trainer-flags) are valid config parameters. Use `python milkshake/main.py -h` to see all options and their descriptions.

The `exps` folder contains experiment code and is where most new code should go. Each experiment in `exps` should call the `main` method from `milkshake/main.py` to train the model, and the standard workflow is to subclass Models or DataModules as required. This codebase includes two examples: `adversarial.py` implements adversarial training, while `distillation.py` implements model distillation.

Logging is integrated with [Weights and Biases](https://docs.wandb.ai/guides); use `wandb login` to sync your account. By default, downloaded data will go in the `data` folder and outputs (e.g., plots) will go in the `out` folder; these can be changed by setting `data_dir` and `out_dir` respectively. The model outputs will be saved to `lightning_logs` and the wandb outputs to `wandb`. I like to redirect my experiment output from `stdout` to a file in the `logs` folder, but this isn't strictly necessary.

### Running experiments

To run an experiment, pass the main file (either `milkshake/main.py` or some file in `exps`) and specify the config with `-c`. For example,

`python milkshake/main.py -c cfgs/mnist.yaml`

To change parameters, one can either write a new config or pass variables on the command line:

`python milkshake/main.py -c cfgs/mnist.yaml --lr 0.05`

By convention, boolean arguments are passed with `True` or `False`:

`python milkshake/main.py -c cfgs/mnist.yaml --balanced_sampler True`

To run the adversarial training example, use:

`python exps/adversarial.py -c cfgs/adversarial.yaml`

and similarly for the distillation example.

### FAQs and gotchas
1. The variable `args` is used for the configuration dictionary, but this unfortunately collides with `*args`, the typical Python variable for an unspecified number of positional arguments. Therefore, `*xargs` is used for positional arguments instead.
2. While PyTorch Lightning handles setting many random seeds, one should still use `np.random.default_rng(seed=args.seed)` or `Generator().manual_seed(args.seed)`, especially in DataModules. This is especially important when splitting the dataset so that the random split remains constant even when running multiple training loops.
3. All new datasets should inherit from `milkshake.datamodules.dataset.Dataset`, and all new models should inherit from `milkshake.models.model.Model`. This may require writing a new class for the dataset/model, even if you are importing it from somewhere else (see `milkshake/datamodules/mnist.py` for an example).
4. When launching multiple jobs in parallel, take care that each job initializes before the next one begins. Otherwise, the version numbers will overwrite and the model checkpoints will not be saved.
5. Currently, the only supported format for `targets`, i.e., the class labels, are digits in the range `[0, num_classes]`. If your targets are in some other format, you can add a preprocessing step in the `DataModule` to map them to this range.
6. When implementing custom logging, you must log some attribute named "loss"; that is, you cannot rename the "loss" attribute to something else.

### Citation and License
Milkshake is distributed under the MIT License. If you use this code in your research, please consider using the following citation:
```
@misc{labonte23,
  author={Tyler LaBonte},
  title={Milkshake: Quick and extendable experimentation with classification models},
  howpublished={\url{http://github.com/tmlabonte/milkshake}},
  year={2023},
}
```
