import tqdm
import numpy as np
import six
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

@six.add_metaclass(ABCMeta)
class ParasolEnvironment(object):

    def __init__(self, sliding_window=0):
        self.recording = False
        self.sliding_window = sliding_window
        self.prev_obs = None

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self, mode='human'):
        pass

    @contextmanager
    def video(self, video_path):
        self.recording = True
        self.start_recording(video_path)
        yield
        self.recording = False
        self.stop_recording()

    @abstractmethod
    def start_recording(self, video_path):
        pass

    @abstractmethod
    def grab_frame(self):
        pass

    @abstractmethod
    def stop_recording(self):
        pass

    @abstractmethod
    def config(self):
        pass

    @abstractmethod
    def state_dim(self):
        pass

    @abstractmethod
    def action_dim(self):
        pass

    def get_state_dim(self):
        return self.state_dim() * (1 + self.sliding_window)

    def get_action_dim(self):
        return self.action_dim()

    def is_recording(self):
        return self.recording

    def is_image(self):
        return False

    def image_size(self):
        return None

    @abstractmethod
    def make_summary(self, observations, name):
        pass

    @abstractmethod
    def _observe(self):
        pass

    def observe(self):
        if self.sliding_window == 0:
            return self._observe()
        curr_obs = self._observe()
        if self.prev_obs is None:
            self.prev_obs = [curr_obs] * self.sliding_window
        obs = [curr_obs] + self.prev_obs
        self.prev_obs = obs[:-1]
        return np.concatenate(obs, 0)

    def rollout(self, num_steps, policy = None, render = False, show_progress = False, init_std=1):
        if policy is None:
            policy = lambda x, t: np.random.normal(size=self.get_action_dim(), scale=init_std)
        states, actions, costs = (
            np.empty([num_steps] + [self.get_state_dim()]),
            np.empty([num_steps] + [self.get_action_dim()]),
            np.empty([num_steps])
        )
        infos = []
        current_state = self.reset()
        times = tqdm.trange(num_steps, desc='Rollout') if show_progress else range(num_steps)
        for t in times:
            states[t] = current_state
            if render:
                self.render(mode='human')
            if self.is_recording():
                self.render(mode='rgb_array')
                self.grab_frame()
            actions[t] = policy(states[t], t)
            current_state, costs[t], done, info = self.step(actions[t])
            infos.append(info)
        return states, actions, costs, infos

    def rollouts(self, num_rollouts, num_steps, show_progress=False, **kwargs):
        states, actions, costs = (
            np.empty([num_rollouts, num_steps] + [self.get_state_dim()]),
            np.empty([num_rollouts, num_steps] + [self.get_action_dim()]),
            np.empty([num_rollouts, num_steps])
        )
        infos = [None] * num_rollouts
        rollouts = tqdm.trange(num_rollouts, desc='Rollouts') if show_progress else range(num_rollouts)
        for i in rollouts:
            states[i], actions[i], costs[i], infos[i] = self.rollout(num_steps, **kwargs)
        return states, actions, costs, infos

    def get_config(self):
        config = self.config().copy()
        config['environment_name'] = self.environment_name
        return config
