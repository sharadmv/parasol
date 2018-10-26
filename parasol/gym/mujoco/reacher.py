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
        self.image = kwargs.pop('image', False)
        self.__dict__.update(kwargs)
        utils.EzPickle.__init__(self)
        if self.image:
            self.prev_obs = None #np.zeros([64, 64, 3])
        assets_dir = os.path.join(os.path.dirname(__file__), "assets", "reacher.xml")
        mujoco_env.MujocoEnv.__init__(self, assets_dir, 2)

    def reward(self, x, a):
        reward_dist = - np.sum(np.square(x))
        reward_ctrl = - np.square(a).sum() * 0.05
        return reward_dist + reward_ctrl, {'Distance' : np.sqrt(-reward_dist)}

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
        qpos = self.get_start() + self.init_qpos
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
            return (cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_LINEAR) / 255).flatten()
        else:
            theta = self.sim.data.qpos.flat[:2]
            return np.concatenate([
                np.cos(theta),
                np.sin(theta),
                self.sim.data.qpos.flat[2:],
                self.sim.data.qvel.flat[:2],
                self.get_body_com("fingertip") - self.get_body_com("target")
            ])

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
            'image_size': kwargs.pop('image_size', 64),
        }
        super(Reacher, self).__init__(config)
