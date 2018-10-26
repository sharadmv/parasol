from abc import abstractmethod

import pygame
import cv2
import math
import matplotlib.animation as manimation
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
sns.set(style='white')

from scipy.misc import imresize

from ..utils import ImageEncoder
from ..env import ParasolEnvironment


__all__ = ['SimpleCar']

class SimpleCar(ParasolEnvironment):

    environment_name = 'SimpleCar'

    def __init__(self, random_start=False, random_target=False,
                 image=False, image_size=32, sliding_window=0):
        super(SimpleCar, self).__init__()
        self.random_start = random_start
        self.random_target = random_target
        self.image = image
        self.image_size = image_size
        self.sliding_window = sliding_window
        self.agent_configuration = None
        self.target_position = None
        self.curr_state = {}
        self._rendered = False
        self.reset()

    def config(self):
        return {
            'random_start': self.random_start,
            'random_target': self.random_target,
            'image': self.image,
            'image_size': self.image_size,
            'sliding_window': self.sliding_window
        }

    def step(self, action):
        action = np.array(action)
        cost = self.cost(self.curr_state, action)
        self.curr_state = self.dynamics(self.curr_state, action)
        agent, target = self.curr_state['agent'], self.curr_state['target']
        dist = np.linalg.norm(agent[:2] - target)
        return self.observe(), cost, False, {'dist': dist}

    def get_state_dim(self):
        if self.image:
            w = self.sliding_window + 1
            return [w * self.image_size ** 2]
        return [9]

    def get_action_dim(self):
        return [2]

    def observe(self):
        if self.image:
            self.render()
            frame = (255 - pygame.surfarray.pixels3d(self.screen).max(axis=-1))
            size = self.image_size
            obs = cv2.resize(frame, (size, size), cv2.INTER_LINEAR)
            return (obs / 255.).flatten()

        agent, target = self.curr_state['agent'], self.curr_state['target']
        x, y, theta, v, phi = agent
        return np.array([x, y, np.cos(theta), np.sin(theta), v,
                         np.cos(phi), np.sin(phi), target[0], target[1]])

    def dynamics(self, state, action):
        agent, target = state['agent'], state['target']
        x, y, theta, v, phi = agent
        dx, dy = v * np.cos(theta), v * np.sin(theta)
        # model car length as 1.0
        dtheta = v * np.tan(phi)
        agent = np.array([x + 0.1 * dx, y + 0.1 * dy, theta + 0.1 * dtheta,
                          v + 0.1 * action[0], phi + 0.1 * action[1]])
        #TODO: not properly modeling car's collisions with walls
        agent[0] = np.clip(agent[0], -2.3, 2.3)
        agent[1] = np.clip(agent[1], -2.3, 2.3)
        agent[3] = np.clip(agent[3], -1.0, 1.0)
        agent[4] = np.clip(agent[4], -np.pi/4, np.pi/4)
        return { 'agent': agent, 'target': target }

    def cost(self, state, action):
        agent, target = state['agent'], state['target']
        return (np.square(agent[:2] - target).sum() + 0.5 * 1e-3 * np.square(action).sum())

    def random_point(self):
        return np.random.uniform(-2.0, 2.0, 2)

    def reset(self):
        self._obs = None
        if self.random_start:
            x, y = self.random_point()
            theta = np.random.uniform(-np.pi, np.pi)
        else:
            x, y = np.array([1.5, -1.5]) + 0.1 * np.random.randn(2)
            size = self.image_size
            theta = np.pi / 2.0 + 0.1 * np.random.randn(1)
        self.curr_state['agent'] = np.array([x, y, theta, 0.0, 0.0])
        if self.random_target:
            self.curr_state['target'] = self.random_point()
        else:
            self.curr_state['target'] = np.array([-2.0, 2.0])
        return self.observe()

    def render(self):
        agent, target = self.curr_state['agent'], self.curr_state['target']
        if not self._rendered:
            pygame.init()
            self.screen = pygame.display.set_mode((500,500))
            self.screen.fill((255, 255, 255))

        x, y, theta, v, phi = agent
        w, h = 1.2, 0.6
        angle0 = np.array([np.cos(theta + np.pi / 6), np.sin(theta + np.pi / 6)])
        angle1 = np.array([np.cos(theta), np.sin(theta)])
        angle2 = np.array([np.cos(theta + np.pi / 2), np.sin(theta + np.pi / 2)])
        angle3 = np.array([np.cos(theta + np.pi / 4), np.sin(theta + np.pi / 4)])
        angle4 = np.array([np.cos(theta - np.pi / 4), np.sin(theta - np.pi / 4)])
        ll = np.array([x, y]) - (h/2) * np.sqrt(5) * angle0
        lr = ll + w * 5 / 6 * angle1
        ur = lr + h * angle2
        ul = ll + h * angle2
        ll_w1 = ll + w * 5 / 120 * angle1
        ll_w2 = ll_w1 - w * 5 / 60 * angle2
        ll_w3 = ll_w2 + w / 4 * angle1
        ll_w4 = ll_w3 + w * 5 / 60 * angle2
        lr_w1 = ll_w4 + w / 4 * angle1
        lr_w2 = lr_w1 - w * 5 / 60 * angle2
        lr_w3 = lr_w2 + w / 4 * angle1
        lr_w4 = lr_w3 + w * 5 / 60 * angle2
        hr = lr + w / 12 * np.sqrt(5) * angle3
        hl = ur + w / 12 * np.sqrt(5) * angle4
        ul_w1 = ul + w * 5 / 120 * angle1
        ul_w2 = ul_w1 + w * 5 / 60 * angle2
        ul_w3 = ul_w2 + w / 4 * angle1
        ul_w4 = ul_w3 - w * 5 / 60 * angle2
        ur_w1 = ul_w4 + w / 4 * angle1
        ur_w2 = ur_w1 + w * 5 / 60 * angle2
        ur_w3 = ur_w2 + w / 4 * angle1
        ur_w4 = ur_w3 - w * 5 / 60 * angle2
        fw_1 = (0.75 * lr_w1 + 0.25 * lr_w4) + w * 5 / 60 * angle2
        fw_2 = fw_1 + 2 * h / 3 * angle2
        fw_4 = (0.25 * lr_w1 + 0.75 * lr_w4) + w * 5 / 30 * angle2
        fw_3 = fw_4 + h / 3 * angle2
        verts = np.array([
            ll, ll_w1, ll_w2, ll_w3, ll_w4,
            lr_w1, lr_w2, lr_w3, lr_w4, lr,
            hr, hl,
            ur, ur_w4, ur_w3, ur_w2, ur_w1,
            ul_w4, ul_w3, ul_w2, ul_w1, ul,
        ])
        scale = 0.75
        w2_verts = np.array([fw_1, fw_2, fw_3, fw_4])
        verts = (verts * scale * 125 + 250).astype(np.int32)
        w2_verts = (w2_verts * scale * 125 + 250).astype(np.int32)
        target_position = (target * scale * 125 + 250).astype(np.int32)
        pygame.draw.polygon(
            self.screen,
            [0, 0, 0],
            verts
        )
        pygame.draw.polygon(
            self.screen,
            [255] * 3,
            w2_verts
        )
        pygame.draw.circle(
            self.screen,
            [0, 0, 0],
            target_position,
            10
        )
        pygame.display.update()

    def start_recording(self):
        frame_shape = (500, 500, 3)
        self.image_encoder = ImageEncoder(self.recording, frame_shape, 30)

    def grab_frame(self):
        frame = pygame.surfarray.pixels3d(self.screen)
        self.image_encoder.capture_frame(frame)

    def stop_recording(self):
        self.image_encoder.close()
