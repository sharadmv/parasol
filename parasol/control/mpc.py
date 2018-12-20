import tqdm
import numpy as np
from .common import Controller
import parasol.util as util

class MPC(Controller):

    control_type = 'mpc'

    def __init__(self, model, horizon,
                 num_rollouts=100,
                 diag_cost=False,
                 action_min=-1.0, action_max=1.0):
        self.model = model
        self.horizon = horizon
        self.ds, self.da = self.model.ds, self.model.da
        self.do, self.du = self.model.do, self.model.du
        self.horizon = horizon
        self.diag_cost = diag_cost
        self.num_rollouts = num_rollouts
        self.action_min, self.action_max = action_min, action_max

        self.C = np.zeros([self.ds + self.da, self.ds + self.da])
        self.c = np.zeros([self.ds + self.da])

    def initialize(self):
        pass

    def act(self, obs, t, noise=None):
        state, _ = self.model.encode(obs, np.zeros(self.model.du))
        states, actions = self.sim_actions_forward(state, t)
        costs = self.eval_traj_costs(states, actions)
        best_simulated_path = np.argmin(costs)
        best_action = actions[best_simulated_path, 0]
        return best_action

    def sim_actions_forward(self, state, start_time):
        states = [np.tile(state, [512, 1])]
        env_horizon = self.model.horizon
        horizon = min(self.horizon, env_horizon - start_time - 1)
        actions = np.random.uniform(self.action_min, self.action_max, size=(512, horizon + 1, self.da))
        curr_states = states[0]
        for t in range(horizon):
            curr_states, _ = self.model.forward(curr_states, actions[:, t], start_time + t)
            states.append(curr_states)
        return np.array(states).transpose([1, 0, 2]), actions

    def eval_traj_costs(self, states, actions):
        costs = np.zeros(states.shape[0])
        for i in range(states.shape[0]):
            cost, traj = 0.0, np.concatenate([states[i], actions[i]], axis=-1)
            for t in range(states.shape[1]):
                sa = traj[t]
                cost += 0.5 * sa.dot(self.C).dot(sa) + self.c.dot(sa)
            costs[i] = cost
        return costs

    def train(self, rollouts, train_step, out_dir=None):
        observations, controls, costs, _ = rollouts

        N, T = observations.shape[:2]
        ds, da = self.ds, self.da
        dsa = ds + da
        self.C, self.c = np.zeros((ds+da, ds+da)), np.zeros((ds+da))
        states, actions = np.zeros((N, T, ds)), np.zeros((N, T, da))

        for t in tqdm.trange(T, desc='fit'):
            for i in range(0, N, 100):
                states[i:i+100, t], actions[i:i+100, t] = self.model.encode(observations[i:i + 100, t], controls[i:i + 100, t])
        S, A = states.reshape((N*T, ds)), actions.reshape((N*T, da))
        SA = np.concatenate([S, A], axis=-1)
        if self.diag_cost:
            dq, quad = dsa * 2, 0.5 * np.square(SA).reshape((N*T, dsa))
        else:
            dq = dsa ** 2 + dsa
            quad = 0.5 * np.einsum('na,nb->nab', SA, SA)
            quad = quad.reshape((N*T, dsa ** 2))
        Q, _, _ = util.linear_fit(
                np.concatenate([
                    quad, SA, costs.reshape((N*T, 1))
                ], axis=-1),
                slice(dq), slice(dq, dq + 1),
        )
        if self.diag_cost:
            self.C = np.diag(Q[0, :dsa])
            self.c = Q[0, dsa:]
        else:
            self.C = Q[0, :dsa ** 2].reshape((dsa, dsa))
            self.C = (self.C + self.C.T) / 2.0
            self.c = Q[0, dsa ** 2:]
