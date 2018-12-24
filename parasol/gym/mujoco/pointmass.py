from deepx import T
import math
import numpy as np
import os
from gym import utils
from gym.envs.mujoco import mujoco_env
import scipy.misc

from ..gym_wrapper import GymWrapper

__all__ = ['Pointmass']

class GymPointmass(mujoco_env.MujocoEnv, utils.EzPickle):

    def __init__(self, *args, **kwargs):
        self.__dict__.update(kwargs)
        utils.EzPickle.__init__(self)
        assets_dir = os.path.join(os.path.dirname(__file__), "assets", "pointmass.xml")
        self.reset_model()
        mujoco_env.MujocoEnv.__init__(self, assets_dir, 2)

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = 0
        self.viewer.cam.elevation = -90.0
        self.viewer.cam.distance = 3

    def reward(self, action):
        pos, target = self.position, self.goal
        dist = np.square(pos - target).sum()
        return -(dist + 1e-3 * np.square(action).sum()), {
            "distance" : np.sqrt(dist)
        }

    def step(self, a):
        reward, info = self.reward(a)
        done = False
        self.position = self.dynamics(self.position, a)
        return self._get_obs(), reward, done, info

    def dynamics(self, state, action):
        if self.image:
            state += action
            for i in range(len(state)):
                while not -2.8 <= state[i] <= 2.8:
                    if state[i] < -2.8:
                        state[i] = -5.6 - state[i]
                    if state[i] > 2.8:
                        state[i] = 5.6 - state[i]
            return state
        else:
            return state + action

    def get_start(self):
        if self.random_start:
            start = np.random.uniform(low=-1,high=1, size=2)
        else:
            start = np.zeros(2)
        return start

    def get_goal(self):
        if self.random_target:
            goal = np.random.uniform(low=-1,high=1, size=2)
        else:
            goal = np.zeros(2)
        return goal

    def render(self, *args, **kwargs):
        qpos = self.sim.get_state().qpos
        qpos[:2] = self.position
        qpos[2:] = self.goal
        self.set_state(qpos, self.sim.get_state().qvel)
        return super(GymPointmass, self).render(*args, **kwargs)

    def reset_model(self):
        self.position = self.get_start()
        self.goal = self.get_goal()
        return self._get_obs()

    def _get_obs(self):
        if self.image:
            scale = self.image_dim // 6
            img = np.zeros((self.image_dim, self.image_dim, 2))

            for i, pos in enumerate([self.position, self.goal]):
                x, y = pos * scale
                x, y = x + (self.image_dim // 2), y + (self.image_dim // 2)
                ind, val = bilinear(x, y)
                if img[..., i][tuple(ind)].shape != (2, 2):
                    continue
                img[..., i][tuple(ind)] = val.T[::-1]
            return img.flatten()
        else:
            return np.concatenate([
                self.position,
                self.goal
            ])

def bilinear(x, y):
    rx, ry = math.modf(x)[0], math.modf(y)[0]
    ix, iy = int(x), int(y)
    ind = [slice(ix-1, ix+1), slice(iy-1, iy+1)]
    val = np.zeros((2, 2))
    val[1, 1] = rx * ry
    val[0, 1] = (1 - rx) * ry
    val[1, 0] = rx * (1 - ry)
    val[0, 0] = (1 - rx) * (1 - ry)
    return ind, val

class Pointmass(GymWrapper):

    environment_name = 'Pointmass'
    entry_point = "parasol.gym.mujoco.pointmass:GymPointmass"
    max_episode_steps = 50
    reward_threshold = -100

    def __init__(self, **kwargs):
        config = {
            'sliding_window': kwargs.pop('sliding_window', 0),
            'image': kwargs.pop('image', False),
            'random_target': kwargs.pop('random_target', True),
            'random_start': kwargs.pop('random_start', True),
            'default_goal': kwargs.pop('default_goal', [-0.1, -0.1]),
            'image_dim': kwargs.pop('image_dim', 32),
        }
        super(Pointmass, self).__init__(config)

    def is_image(self):
        return self.image

    def make_summary(self, observations, name):
        if self.image:
            observations = T.reshape(observations, [-1] + self.image_size())
            T.core.summary.image(name+"-point", observations[..., 0:1])
            T.core.summary.image(name+"-goal", observations[..., 1:2])

    def image_size(self):
        return [self.image_dim, self.image_dim, 2]

