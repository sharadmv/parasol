from deepx import T
import cv2
import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
import scipy.misc

from ..gym_wrapper import GymWrapper

__all__ = ['Reacher']

class GymReacher(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        utils.EzPickle.__init__(self)
        if self.image:
            self.prev_obs = None #np.zeros([64, 64, 3])
        assets_dir = os.path.join(os.path.dirname(__file__), "assets", "reacher.xml")
        mujoco_env.MujocoEnv.__init__(self, assets_dir, 2)

    def reward(self, x, a):
        if self.easy_cost:
            reward_dist = - np.square(x).sum()
            reward_ctrl = - np.square(a).sum()
            dist = np.linalg.norm(x)
        else:
            reward_dist = - np.linalg.norm(x)
            reward_ctrl = - np.square(a).sum()
            dist = -reward_dist
        return reward_dist + reward_ctrl, {'distance' : dist}

    def step(self, a):
        vec = self.get_body_com("fingertip")-self.get_body_com("target")
        reward, info = self.reward(vec, a)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, info

    def get_start(self):
        if self.random_start:
            start = np.random.uniform(low=-np.pi, high=np.pi, size=self.model.nq)
        else:
            start = np.zeros(self.model.nq)
        return start

    def get_goal(self):
        if self.random_target:
            goal = np.array([self.np_random.uniform(-0.2, 0.2), self.np_random.uniform(-0.2, 0.2)])
        else:
            goal = self.default_goal
        return goal

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -90.0
        self.viewer.cam.distance = 0.6

    def reset_model(self):
        if self.random_start:
            qpos = self.np_random.uniform(low=-np.pi, high=np.pi, size=self.model.nq) + self.init_qpos
        else:
            qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        self.goal = self.get_goal()
        qpos[-2:] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-.005, high=.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        if self.image:
            self.prev_obs = None #np.zeros([64, 64, 3])
        obs = self._get_obs()
        return obs

    def _get_obs(self):
        if self.image:
            img = self.render(mode='rgb_array')
            return (cv2.resize(img, (self.image_dim, self.image_dim), interpolation=cv2.INTER_LINEAR) / 255).flatten()
        else:
            theta = self.sim.data.qpos.flat[:2]
            obs = np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target")
            ])[:-1] + np.concatenate([np.zeros(4), np.random.normal(size=2, scale=0.01), np.zeros(4)])
            return obs

class Reacher(GymWrapper):

    environment_name = 'Reacher'
    entry_point = "parasol.gym.mujoco.reacher:GymReacher"
    max_episode_steps = 50
    reward_threshold = -3.75

    def __init__(self, **kwargs):
        config = {
            'image': kwargs.pop('image', False),
            'sliding_window': kwargs.pop('sliding_window', 0),
            'random_target': kwargs.pop('random_target', False),
            'random_start': kwargs.pop('random_start', False),
            'default_goal': kwargs.pop('default_goal', [-0.1, -0.1]),
            'image_dim': kwargs.pop('image_dim', 64),
            'easy_cost': kwargs.pop('easy_cost', False),
        }
        super(Reacher, self).__init__(config)

    def torque_matrix(self):
        return 2 * np.eye(self.get_action_dim())

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
    
    def cost_fn(self, s, a):
        return np.linalg.norm(s[:,-3:], axis=-1) + np.sum(np.square(a), axis=-1)

