import pickle
import tensorflow as tf
from deepx import T
import numpy as np
import random

import parasol.gym as gym
import parasol.control
import parasol.model

from .common import Experiment

gfile = tf.gfile

class Solar(Experiment):

    experiment_type = 'solar'

    def __init__(self, experiment_name, env, control, model,
                 seed=0,
                 **kwargs):
        super(Solar, self).__init__(experiment_name, **kwargs)
        self.env_params = env
        self.env = gym.from_config(env)
        self.control_params = control
        self.horizon = control['horizon']
        self.model_path = model
        self.seed = seed
        self.initialize()

    def initialize(self):
        if self.model_path is not None:
            with gfile.GFile(self.model_path, 'rb') as fp:
                self.model = pickle.load(fp)
        self.control = parasol.control.from_config(self.model, self.control_params)

    def to_dict(self):
        return {
            'env': self.env_params,
            'seed': self.seed,
            'env': self.env_params.copy(),
            'control': self.control_params.copy(),
            'model': self.model_path,
            'out_dir': self.out_dir,
        }

    @classmethod
    def from_dict(cls, params):
        return Solar(
            params['experiment_name'],
            params['env'],
            params['control'],
            params['model'],
            out_dir=params['out_dir'],
            seed=params['seed'],
        )

    def run_experiment(self, out_dir):

        T.core.set_random_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        initial_rollouts = self.env.rollouts(100, self.horizon)

        model = self.model
        control = self.control

        control.train(initial_rollouts)
        with self.env.video(out_dir / 'out.mp4'):
            self.env.rollout(self.horizon, policy=control.act, render=True, show_progress=True)
