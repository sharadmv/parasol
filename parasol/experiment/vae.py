from path import Path
import pickle
import random
import tqdm
import numpy as np
import deepx
from deepx import T, stats
import parasol.gym as gym
import parasol.util as util
import tensorflow.gfile as gfile

from parasol.model import VAE

from .common import Experiment

class TrainVAE(Experiment):

    experiment_type = "train_vae"

    def __init__(self,
                 experiment_name,
                 env,
                 model,
                 data={},
                 train={},
                 seed=0,
                 num_rollouts=100,
                 init_std=1.,
                 num_epochs=1000,
                 learning_rate=1e-4,
                 batch_size=20,
                 prior=None,
                 dump_every=None,
                 dump_data=True,
                 summary_every=1000,
                 beta_start=0.0, beta_rate=1e-4,
                 beta_end=1.0, **kwargs):
        super(TrainVAE, self).__init__(experiment_name, **kwargs)
        self.env_params = env
        self.model_params = model
        self.train_params = train
        self.data_params = data
        self.seed = seed
        self.dump_data = dump_data
        self.horizon = horizon = model['horizon']

        if 'load_model' in self.model_params:
            with gfile.GFile(self.model_params['load_model'], 'rb') as fp:
                self.model = pickle.load(fp)
        else:
            self.model = VAE(
                **model
            )
        self.env = gym.from_config(self.env_params)
        self.model.make_summaries(self.env)

    def initialize(self, out_dir):
        if not gfile.Exists(out_dir / "tb"):
            gfile.MakeDirs(out_dir / "tb")
        if not gfile.Exists(out_dir / "weights"):
            gfile.MakeDirs(out_dir / "weights")
        if not gfile.Exists(out_dir / "weights"):
            gfile.MakeDirs(out_dir / "weights")
        if not gfile.Exists(out_dir / "data"):
            gfile.MakeDirs(out_dir / "data")

    def to_dict(self):
        return {
            "seed": self.seed,
            "out_dir": self.out_dir,
            "env": self.env_params,
            "experiment_name": self.experiment_name,
            "experiment_type": self.experiment_type,
            "dump_data": self.dump_data,
            "model": self.model_params.copy(),
            "data": self.data_params.copy(),
            "train": self.train_params.copy(),
        }

    @classmethod
    def from_dict(cls, params):
        return TrainVAE(
            params['experiment_name'],
            params['env'],
            params['model'],
            data=params['data'],
            train=params['train'],
            seed=params['seed'],
            dump_data=params['dump_data'],
            out_dir=params['out_dir']
        )

    def run_experiment(self, out_dir):
        out_dir = Path(out_dir)

        T.core.set_random_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        if 'load_data' in self.data_params:
            with gfile.GFile(self.data_params['load_data'], 'rb') as fp:
                rollouts = pickle.load(fp)
        else:
            env = self.env
            num_rollouts, init_std = self.data_params['num_rollouts'], self.data_params['init_std']

            def policy(x, _, noise=None):
                action = np.random.multivariate_normal(mean=np.zeros(env.get_action_dim()),
                                          cov=np.eye(env.get_action_dim()) *
                                          (init_std**2))
                return action
            rollouts = env.rollouts(num_rollouts, self.horizon, policy=policy, show_progress=True)
        if self.dump_data:
            with gfile.GFile(out_dir / "data" / "rollouts.pkl", 'wb') as fp:
                pickle.dump(rollouts, fp)
        self.model.train(rollouts, out_dir=out_dir, **self.train_params)
