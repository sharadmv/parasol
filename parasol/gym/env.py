import numpy as np
import six
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager

@six.add_metaclass(ABCMeta)
class ParasolEnvironment(object):

    def __init__(self):
        self.recording = None

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def render(self):
        pass

    @contextmanager
    def video(self, video_path):
        self.recording = video_path
        self.start_recording()
        yield
        self.recording = None
        self.stop_recording()

    @abstractmethod
    def start_recording(self):
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
    def get_state_dim(self):
        pass

    @abstractmethod
    def get_action_dim(self):
        pass

    def is_recording(self):
        return self.recording is not None

    def rollout(self, num_steps, policy = None, render = False):
        if policy is None:
            policy = lambda x, t: np.random.random(size=self.get_action_dim())
        states, actions, costs = (
            np.empty([num_steps] + self.get_state_dim()),
            np.empty([num_steps] + self.get_action_dim()),
            np.empty([num_steps])
        )
        infos = []
        current_state = self.reset()
        for t in range(num_steps):
            states[t] = current_state
            if render:
                self.render()
            if self.is_recording():
                self.grab_frame()
            actions[t] = policy(states[t], t)
            current_state, costs[t], done, info = self.step(actions[t])
            infos.append(info)
        return states, actions, costs, infos

    def get_config(self):
        config = self.config().copy()
        config['environment_name'] = self.environment_name
        return config
