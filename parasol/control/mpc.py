import tqdm
import numpy as np
from .common import Controller
import parasol.util as util
import scipy.stats as st

class MPC(Controller):

    control_type = 'mpc'

    def __init__(self, model, env, horizon,
                 diag_cost=False,
                 action_min=-1.0, action_max=1.0, sample_std=1.0):
        self.model = model
        self.horizon = horizon
        self.ds, self.da = self.model.ds, self.model.da
        self.do, self.du = self.model.do, self.model.du
        self.horizon = horizon
        self.diag_cost = diag_cost
        self.action_min, self.action_max = action_min, action_max
        self.env = env

    def initialize(self):
        pass

    def act(self, obs, t, noise=None):
        state, _ = self.model.encode(obs, np.zeros(self.model.du))
        horizon = min(self.horizon, self.model.horizon - t)
        action = self.cem_opt(state, horizon, iters=10)
        if noise is not None:
            action += noise * 0.01
        return action

    def cem_opt(self, state, horizon, iters=1):
        mu = np.zeros((horizon,self.da))
        sigma = 0.2 * np.ones((horizon, self.da))
        for i in range(iters):
            states, actions = self.sim_actions_forward(state, horizon, mu, sigma)
            costs = self.eval_traj_costs(states, actions)
            best_candidates = actions[np.argsort(costs)[:15]]
            mu, sigma = np.mean(best_candidates, axis=0), np.std(best_candidates, axis=0)
            sigma = np.clip(sigma, 0, 2)
        return mu[0]

    def sim_actions_forward(self, state, horizon, mu, sigma):
        num_traj = 1024
        states = [np.tile(state, [num_traj, 1])]
        X = st.truncnorm(-2, 2, loc = mu, scale = sigma)
        actions = X.rvs(size = (num_traj, horizon, self.da))
        curr_states = states[0]
        for t in range(horizon):
            curr_states, _ = self.model.forward(curr_states, actions[:, t], 0)
            if t < horizon - 1:
                states.append(curr_states)
        return np.array(states).transpose([1, 0, 2]), actions

    def eval_traj_costs(self, states, actions):
        costs = np.zeros(states.shape[0])
        for t in range(states.shape[1]):
            costs += self.env.cost_fn(states[:, t], actions[:, t])
        return costs

    def train(self, rollouts, train_step, out_dir=None):
        pass
