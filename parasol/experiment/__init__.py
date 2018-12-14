import tensorflow as tf
from parasol.util import json
from .common import Experiment
from .vae import TrainVAE

gfile = tf.gfile

__all__ = ['Experiment', 'TrainVAE', 'from_json']

EXPERIMENTS = [TrainVAE]

EXPERIMENT_MAP = {}
for experiment in EXPERIMENTS:
    EXPERIMENT_MAP[experiment.experiment_type] = experiment

def from_json(fp):
    if isinstance(fp, str):
        with gfile.GFile(fp, 'r') as f:
            params = json.load(f)
    else:
        params = json.load(fp)
    return EXPERIMENT_MAP[params['experiment_type']].from_dict(params)
