import os

import cv2
import numpy as np
import scipy.misc

from deepx import T
from gym import utils
from gym.envs.mujoco import mujoco_env

from ..gym_wrapper import GymWrapper


__all__ = ['Cartpole']


class GymCartpole(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        utils.EzPickle.__init__(self)
        if self.image:
            self.prev_obs = None
        assets_dir = os.path.join(os.path.dirname(__file__), "assets", "inverted_pendulum.xml")
        mujoco_env.MujocoEnv.__init__(self, assets_dir, 2)

    def reward(self, ob, a):
        reward_dist = -(ob[1] ** 2)
        reward_ctrl = -1e-4 * np.square(a).sum()
        done = not (np.isfinite(ob).all() and (np.abs(ob[1]) <= 0.2))
        return reward_dist + reward_ctrl, {'done' : done}

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        reward, info = self.reward(self.sim.data.qpos, a)
        done = False
        return ob, reward, done, info

    def viewer_setup(self):
        v = self.viewer
        v.cam.trackbodyid = 0
        v.cam.distance = self.model.stat.extent

    def reset_model(self):
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        if self.random_start:
            qpos[0] += self.np_random.uniform(low=-0.5, high=0.5)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        if self.image:
            img = self.render(mode='rgb_array')
            return (cv2.resize(img, (self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR) / 255).flatten()
        else:
            return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()


class Cartpole(GymWrapper):

    environment_name = 'Cartpole'
    entry_point = "parasol.gym.mujoco.cartpole:GymCartpole"
    max_episode_steps = 100
    reward_threshold = -3.75  # ignore

    def __init__(self, **kwargs):
        config = {
            'image': kwargs.pop('image', False),
            'sliding_window': kwargs.pop('sliding_window', 0),
            'image_dim': kwargs.pop('image_dim', 32),
            'random_start': kwargs.pop('random_start', False),
        }
        super(Cartpole, self).__init__(config)

    def torque_matrix(self):
        return 2e-4 * np.eye(self.get_action_dim())

    def make_summary(self, observations, name):
        if self.image:
            observations = T.reshape(observations, [-1] + self.image_size())
            T.core.summary.image(name, observations)

    def is_image(self):
        return self.image

    def image_size(self):
        if self.image:
            return [self.image_dim, self.image_dim, 3]
        return None
