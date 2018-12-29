import pickle
import tensorflow as tf
from deepx import T
import numpy as np
import random

import parasol.util as util
import parasol.gym as gym
import parasol.control
import parasol.model

from .common import Experiment

gfile = tf.gfile

class Solar(Experiment):

    experiment_type = 'solar'

    def __init__(self, experiment_name, env, control, model,
                 seed=0,
                 horizon=50,
                 num_videos=1,
                 rollouts_per_iter=100,
                 num_iters=10,
                 buffer_size=None,
                 model_train={},
                 **kwargs):
        super(Solar, self).__init__(experiment_name, **kwargs)
        self.env_params = env
        self.env = gym.from_config(env)
        self.control_params = control
        self.horizon = horizon
        self.model_path = model
        self.seed = seed
        self.num_videos = num_videos
        self.rollouts_per_iter = rollouts_per_iter
        self.buffer_size = buffer_size if buffer_size is not None else rollouts_per_iter
        self.num_iters = num_iters
        self.model_train_params = model_train

    def initialize(self, out_dir):
        if not gfile.Exists(out_dir / "tb"):
            gfile.MakeDirs(out_dir / "tb")
        if not gfile.Exists(out_dir / "weights"):
            gfile.MakeDirs(out_dir / "weights")
        if not gfile.Exists(out_dir / "policy"):
            gfile.MakeDirs(out_dir / "policy")
        if not gfile.Exists(out_dir / "videos"):
            gfile.MakeDirs(out_dir / "videos")
        self.initialize_params()

    def initialize_params(self):
        if self.model_path is not None:
            with gfile.GFile(self.model_path, 'rb') as fp:
                self.model = pickle.load(fp)
        else:
            self.model = parasol.model.NoModel(self.env.get_state_dim(),
                                               self.env.get_action_dim(), self.horizon)
        self.control = parasol.control.from_config(self.model, self.control_params, self.env)

    def to_dict(self):
        return {
            'experiment_name': self.experiment_name,
            'experiment_type': self.experiment_type,
            'env': self.env_params,
            'seed': self.seed,
            'env': self.env_params.copy(),
            'control': self.control_params.copy(),
            'horizon': self.horizon,
            'model': self.model_path,
            'model_train': self.model_train_params,
            'out_dir': self.out_dir,
            'buffer_size': self.buffer_size,
            'rollouts_per_iter': self.rollouts_per_iter,
            'num_videos': self.num_videos,
            'num_iters': self.num_iters,
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
            num_videos=params['num_videos'],
            num_iters=params['num_iters'],
            buffer_size=params['buffer_size'],
            model_train=params['model_train'],
            horizon=params['horizon'],
            rollouts_per_iter=params['rollouts_per_iter']
        )

    def run_experiment(self, out_dir):

        T.core.set_random_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        def noise_function():
            n = util.generate_noise((self.horizon, self.control.da),
                                    smooth=False)
            return n

        replay_buffer = None

        for i in range(self.num_iters):
            def video_callback(j):
                if j < self.num_videos:
                    return self.env.video(out_dir / 'videos' / 'iter-{}-{}.mp4'.format(i + 1, j + 1))
            print("Iteration {}: ==================================================".format(i))
            with self.env.logging(out_dir / 'results.csv', verbose=True):
                rollouts = self.env.rollouts(self.rollouts_per_iter, self.horizon,
                                            policy=self.control.act,
                                            callback=video_callback,
                                            noise=noise_function,
                                            show_progress=True)
            if replay_buffer is None:
                replay_buffer = rollouts
            else:
                replay_buffer = tuple(map(np.concatenate, zip(replay_buffer, rollouts)))
            replay_buffer = tuple(r[-self.buffer_size:] for r in replay_buffer)
            self.control.train(replay_buffer, i, out_dir=out_dir)
            self.model.train(replay_buffer, out_dir=out_dir, **self.model_train_params)

        def video_callback(j):
            if j < self.num_videos:
                return self.env.video(out_dir / 'videos' / 'final-{}.mp4'.format(j + 1))
        with self.env.logging(out_dir / 'results.csv', verbose=True):
            rollouts = self.env.rollouts(self.rollouts_per_iter, self.horizon,
                                        policy=self.control.act,
                                        callback=video_callback,
                                        noise=noise_function,
                                        show_progress=True)
