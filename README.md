# Towards Last-layer Retraining for Group Robustness with Fewer Annotations
### Official codebase for the NeurIPS 2023 paper: https://arxiv.org/abs/2309.08534.

### Installation
```
conda update -n base -c defaults conda
conda create -n milkshake python==3.10
conda activate milkshake
conda install pytorch==1.12.1 torchvision==0.13.1 cudatoolkit=11.3 -c pytorch
python -m pip install -e .
```

### Instructions

To run an experiment, specify the config with `-c`. For example,

`python exps/disagreement.py -c cfgs/celeba.yaml`

To run a demo of early-stop disagreement SELF, use

`python exps/disagreement.py -c cfgs/waterbirds/yaml --demo`

By default, the program will run ERM training, last-layer retraining, and SELF with model selection.

### Citation and License
This codebase uses [Milkshake](https://github.com/tmlabonte/milkshake) as a template and inherits its MIT License. Please consider using the following citation:
```
@inproceedings{labonte23towards,
  author={Tyler LaBonte and Vidya Muthukumar and Abhishek Kumar},
  title={Towards Last-layer Retraining for Group Robustness with Fewer Annotations},
  booktitle={Conference on Neural Information Processing Systems (NeurIPS)},
  year={2023},
}
```
