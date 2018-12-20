import csv
import collections
import contextlib
import tqdm
import numpy as np
import six
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
import tensorflow as tf

gfile = tf.gfile

@six.add_metaclass(ABCMeta)
class ParasolEnvironment(object):

    def __init__(self, sliding_window=0):
        self.recording = False
        self.episode_number = 1
        self.currently_logging = False
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

    def rollout(self, num_steps, policy = None, render = False, show_progress =
                False, init_std=1, noise=None):
        if policy is None:
            policy = lambda x, t: np.random.normal(size=self.get_action_dim(), scale=init_std)
        states, actions, costs = (
            np.empty([num_steps] + [self.get_state_dim()]),
            np.empty([num_steps] + [self.get_action_dim()]),
            np.empty([num_steps])
        )
        infos = collections.defaultdict(list)
        current_state = self.reset()
        times = tqdm.trange(num_steps, desc='Rollout') if show_progress else range(num_steps)
        for t in times:
            states[t] = current_state
            if render:
                self.render(mode='human')
            if self.is_recording():
                self.render(mode='rgb_array')
                self.grab_frame()
            n = None
            if noise is not None:
                n = noise[t]
                actions[t] = policy(states[t], t, noise=n)
            current_state, costs[t], done, info = self.step(actions[t])
            for k, v in info.items():
                infos[k].append(v)
        if self.currently_logging:
            log_entry = collections.OrderedDict()
            log_entry['episode_number'] = self.episode_number
            log_entry['mean_cost'] = costs.mean()
            log_entry['total_cost'] = costs.sum()
            log_entry['final_cost'] = costs[-1]
            for k, v in infos.items():
                v = np.array(v)
                log_entry['mean_%s' % k] = v.mean()
                log_entry['total_%s' % k] = v.sum()
                log_entry['final_%s' % k] = v[-1]
            self.log_entry(log_entry)
            self.episode_number += 1
        return states, actions, costs, infos

    def rollouts(self, num_rollouts, num_steps, show_progress=False,
                 noise=None,
                 callback=lambda x: None,
                 **kwargs):
        states, actions, costs = (
            np.empty([num_rollouts, num_steps] + [self.get_state_dim()]),
            np.empty([num_rollouts, num_steps] + [self.get_action_dim()]),
            np.empty([num_rollouts, num_steps])
        )
        infos = [None] * num_rollouts
        rollouts = tqdm.trange(num_rollouts, desc='Rollouts') if show_progress else range(num_rollouts)
        for i in rollouts:
            with contextlib.ExitStack() as stack:
                context = callback(i)
                if context is not None:
                    stack.enter_context(callback(i))
                n = None
                if noise is not None:
                    n = noise()
                states[i], actions[i], costs[i], infos[i] = \
                        self.rollout(num_steps, noise=n,**kwargs)
        return states, actions, costs, infos

    def get_config(self):
        config = self.config().copy()
        config['environment_name'] = self.environment_name
        return config

    @contextmanager
    def logging(self, log_file, **kwargs):
        self.start_logging(log_file, **kwargs)
        yield
        self.stop_logging()

    def log_entry(self, entry):
        self.log_entries.append(entry)

    def start_logging(self, log_file, verbose=False):
        self.log_file = log_file
        self.log_entries = []
        self.currently_logging = True
        self.verbose_logging = verbose

    def stop_logging(self):
        if len(self.log_entries) > 0:
            with open(self.log_file, 'a+') as fp:
                log_writer = csv.writer(fp)
                if (self.episode_number - len(self.log_entries) - 1) == 0:
                    log_writer.writerow(self.log_entries[0].keys())
                for entry in self.log_entries:
                    log_writer.writerow(entry.values())
            if self.verbose_logging:
                for k in self.log_entries[0].keys():
                    if k == 'episode_number': continue
                    print("Average %s: %s" % (k, np.array([l[k] for l in
                                                           self.log_entries]).mean()))
        self.log_entries = None
        self.log_file = None
        self.currently_logging = False
