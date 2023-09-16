"""Import functions for milkshake."""

# Imports milkshake packages.
import milkshake
from milkshake.datamodules import *
from milkshake.models import *


def valid_model_and_datamodule_names():
    """Returns valid input names for models and datamodules."""

    model_names = [n for n in milkshake.models.__all__ if n != "model"]
    datamodule_names = [n for n in milkshake.datamodules.__all__ if n not in ("dataset", "datamodule")]

    return model_names, datamodule_names

def valid_models_and_datamodules():
    """Returns {name: class} dict for valid models and datamodules."""

    model_names, datamodule_names = valid_model_and_datamodule_names()

    models = [milkshake.models.__dict__[name].__dict__ for name in model_names]
    models = [dict((k.lower(), v) for k, v in d.items()) for d in models]
    models = {name: models[j][name.replace("_", "")] for j, name in enumerate(model_names)} 

    datamodules = [milkshake.datamodules.__dict__[name].__dict__ for name in datamodule_names]
    datamodules = [dict((k.lower(), v) for k, v in d.items()) for d in datamodules]
    datamodules = {name: datamodules[j][name.replace("_", "")] for j, name in enumerate(datamodule_names)} 

    return models, datamodules

