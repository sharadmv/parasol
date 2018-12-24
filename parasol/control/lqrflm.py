import tqdm
import pickle
import tensorflow as tf
import numpy as np
import scipy as sp
from gps.algorithm.dynamics.dynamics_prior_gmm import DynamicsPriorGMM

import parasol.util as util
from .common import Controller

gfile = tf.gfile

class LQRFLM(Controller):

    control_type = 'lqrflm'

    def __init__(self, model, env, horizon, kl_step=2.0, init_std=1.0, diag_cost=False,
                 linearize_policy = False,
                 pd_cost = False,
                 prior_type='gmm'):
        self.model = model
        self.horizon = horizon
        self.env = env
        self.ds, self.da = self.model.ds, self.model.da
        self.do, self.du = self.model.do, self.model.du
        self.init_std = np.ones(self.da) * init_std
        self.kl_step = kl_step * self.horizon
        self.horizon = horizon
        self.diag_cost = diag_cost
        self.prior_type = prior_type
        self.pd_cost = pd_cost
        self.linearize_policy = linearize_policy
        self.initialize()

    def initialize(self):
        self.initialize_policy()
        self.eta, self.step_mult = 1.0, 1.0

    def initialize_policy(self, initial_policy=None):
        T, ds, da = self.horizon, self.ds, self.da
        if initial_policy is None:
            self.policy_params = (
                np.zeros([T, da, ds]),
                np.zeros([T, da]),
                np.tile(np.diag(self.init_std ** 2), [T, 1, 1]),
                np.tile(np.diag(self.init_std), [T, 1, 1]),
                np.tile(np.diag(1.0 / (self.init_std ** 2)), [T, 1, 1])
            )
        else:
            K, k, S = initial_policy
            self.policy_params = (
                K,
                k,
                S,
                np.linalg.cholesky(S),
                np.linalg.inv(S)
            )

    def act(self, obs, t, noise=None):
        K, k, S, cS, iS = self.policy_params
        state, _ = self.model.encode(obs, np.zeros(self.da))
        if noise is None:
            noise = np.random.randn(self.da)
        noise = np.einsum('ab,b->a', cS[t], noise)
        action = np.einsum('ab,b->a', K[t], state) + k[t] + noise
        return action

    def train(self, rollouts, train_step, out_dir=None):
        self.fit_dynamics(rollouts, train_step)
        if train_step > 0:
            self.actual_impr = self.prev_cost_estimate - self.estimate_cost()
            self.step_adjust()
        self.prev_cost_estimate = self.estimate_cost()
        self.policy_params = self.tr_update()
        self.predicted_impr = self.prev_cost_estimate - self.estimate_cost()
        with gfile.GFile(out_dir / 'policy' / '{}.pkl'.format(train_step), 'wb') as fp:
            pickle.dump(self.policy_params, fp)

    def fit_dynamics(self, rollouts, train_step):
        observations, controls, costs, _ = rollouts
        N = observations.shape[0]
        T, ds, da = self.horizon, self.ds, self.da
        dsa = ds + da

        states, actions = np.zeros((N, T, ds)), np.zeros((N, T, da))

        for t in tqdm.trange(T, desc='Encoding'):
            for idx, chunk in util.chunk(observations[:, t], controls[:, t],
                                         chunk_size=100):
                states[idx, t], actions[idx, t] = self.model.encode(chunk[0], chunk[1])
            if t == 0:
                self.mu_s0 = np.mean(states[:, t], axis=0)
                self.S_s0 = np.diag(np.maximum(np.var(states[:, t], axis=0),
                                               1e-6))
        if self.prior_type == 'gmm':
            gmm = DynamicsPriorGMM({
                'max_samples': N, 'max_clusters': 20, 'min_samples_per_cluster': 40,
            })
            gmm.update(states, actions)

        self.D, self.d = np.zeros((T, ds, ds+da)), np.zeros((T, ds))
        self.S_D = np.zeros((T, ds, ds))
        for t in tqdm.trange(T, desc='Fitting dynamics'):
            if t < T - 1:
                SAS_ = np.concatenate(
                        [states[:, t], actions[:, t], states[:, t+1]], axis=-1,
                )
                if self.prior_type == 'mdl':
                    raise NotImplementedError
                elif self.prior_type == 'gmm':
                    prior = gmm.eval(ds, da, SAS_)
                else:
                    prior = None
                self.D[t], self.d[t], self.S_D[t] = util.linear_fit(
                        SAS_, slice(ds+da), slice(ds+da, ds+da+ds), prior=prior,
                )
                self.S_D[t] = 0.5 * (self.S_D[t] + self.S_D[t].T)
        self.C = np.zeros([self.ds + self.da, self.ds + self.da])
        self.c = np.zeros([self.ds + self.da])
        if self.pd_cost:
            self.C[:self.ds, :self.ds], self.c[:self.ds] = \
            util.quadratic_regression_pd(states, costs -
                                         np.einsum('nta,ab,ntb->nt', actions,
                                                   self.env.torque_matrix(),
                                                   actions))
        else:
            self.C[:self.ds, :self.ds], self.c[:self.ds] = \
            util.quadratic_regression(states, costs -
                                         0.5 * np.einsum('nta,ab,ntb->nt', actions,
                                                   self.env.torque_matrix(),
                                                   actions))
        self.C[self.ds:, self.ds:] = self.env.torque_matrix()
        self.c[self.ds:] = np.zeros(self.da)

        print(self.C)
        print(self.c)
        self.C = np.tile((self.C + self.C.T) / 2.0, [T, 1, 1])
        self.c = np.tile(self.c, [T, 1])

    def estimate_cost(self):
        mu, sigma = self.forward(self.policy_params)
        C, c = self.C, self.c
        predicted_cost = 0.0
        for t in range(self.horizon):
            predicted_cost += 0.5 * np.sum(sigma[t] * C[t]) + \
                    0.5 * mu[t].T.dot(C[t]).dot(mu[t]) + mu[t].T.dot(c[t])
        return predicted_cost

    def step_adjust(self):
        new_mult = self.predicted_impr / \
                (2.0 * max(1e-4, self.predicted_impr - self.actual_impr))
        new_mult = max(0.1, min(5.0, new_mult))
        new_step = max(min(new_mult * self.step_mult, 10.0), 0.1)
        print('Adjusting step size: {} -> {}'.format(self.step_mult, new_step))
        self.step_mult = new_step

    def forward(self, policy_params):
        K, k, S, _, _ = policy_params
        T, ds, da = self.horizon, self.ds, self.da
        idx_s = slice(self.ds)
        D, d, S_D = self.D, self.d, self.S_D

        sigma, mu = np.zeros((T, ds+da, ds+da)), np.zeros((T, ds+da))
        sigma[0, idx_s, idx_s] = self.S_s0
        mu[0, idx_s] = self.mu_s0

        for t in range(T):
            sigma[t] = np.vstack([
                np.hstack([
                    sigma[t, idx_s, idx_s],
                    sigma[t, idx_s, idx_s].dot(K[t].T),
                ]),
                np.hstack([
                    K[t].dot(sigma[t, idx_s, idx_s]),
                    K[t].dot(sigma[t, idx_s, idx_s]).dot(K[t].T) + S[t],
                ]),
            ])
            mu[t] = np.hstack([
                mu[t, idx_s],
                K[t].dot(mu[t, idx_s]) + k[t],
            ])
            if t < T - 1:
                sigma[t+1, idx_s, idx_s] = \
                        D[t].dot(sigma[t]).dot(D[t].T) + S_D[t]
                mu[t+1, idx_s] = D[t].dot(mu[t]) + d[t]
        return mu, sigma

    def tr_update(self, max_iter=50):
        eta = self.eta
        min_eta, max_eta = 1e-8, 1e16
        kl_step = self.kl_step * self.step_mult

        for itr in range(max_iter):
            print('Iteration {}, bracket: [{}] [{}] [{}]'.format(
                itr, min_eta, eta, max_eta,
            ))

            old_eta = eta
            new_policy_params, eta = self.backward(eta)

            new_mu, new_sigma = self.forward(new_policy_params)
            kl_div = self.traj_kl(new_policy_params,
                                  self.policy_params)
            con = kl_div - kl_step
            if abs(con) < 0.1 * kl_step:
                print('KL: {} / {}, converged iteration {}'.format(
                      kl_div, kl_step, itr,
                ))
                break

            if con < 0:
                max_eta = eta
                geom = np.sqrt(min_eta * max_eta)
                new_eta = max(geom, 0.1 * max_eta)
                print('KL: {} / {}, eta too big, new eta: {}'.format(
                      kl_div, kl_step, new_eta,
                ))
            else:
                min_eta = eta
                geom = np.sqrt(min_eta * max_eta)
                new_eta = min(geom, 10.0 * min_eta)
                print('KL: {} / {}, eta too small, new eta: {}'.format(
                      kl_div, kl_step, new_eta,
                ))
            eta = new_eta
            if abs(eta - old_eta) < 1e-6:
                print('KL: {} / {}, eta {}, stopping iteration {}'.format(
                        kl_div, kl_step, eta, itr,
                ))
                break
        return new_policy_params

    def backward(self, eta):
        T, ds, da = self.horizon, self.ds, self.da
        idx_s = slice(ds)
        idx_a = slice(ds, ds+da)
        D, d = self.D, self.d
        K, k, S, cS, iS = [a.copy() for a in self.policy_params]

        del_, eta0 = 1e-4, eta
        fail = True
        while fail:
            fail = False

            Vtt = np.zeros((T, ds, ds))
            Vt = np.zeros((T, ds))
            Qtt = np.zeros((T, ds+da, ds+da))
            Qt = np.zeros((T, ds+da))
            C, c = self.compute_costs(eta)

            for t in range(T - 1, -1, -1):
                Qtt[t], Qt[t] = C[t], c[t]
                if t < T - 1:
                    Qtt[t] += D[t].T.dot(Vtt[t+1]).dot(D[t])
                    Qt[t] += D[t].T.dot(Vt[t+1] + Vtt[t+1].dot(d[t]))
                Qtt[t] = 0.5 * (Qtt[t] + Qtt[t].T)

                inv_term = Qtt[t, idx_a, idx_a]
                k_term = Qt[t, idx_a]
                K_term = Qtt[t, idx_a, idx_s]
                try:
                    U = sp.linalg.cholesky(inv_term)
                    L = U.T
                except np.linalg.LinAlgError as e:
                    print('LinAlgError: {}'.format(e))
                    fail = True
                    break

                iS[t] = inv_term
                S[t] = sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, np.eye(da), lower=True)
                )
                cS[t] = sp.linalg.cholesky(S[t])
                k[t] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, k_term, lower=True)
                )
                K[t] = -sp.linalg.solve_triangular(
                    U, sp.linalg.solve_triangular(L, K_term, lower=True)
                )
                Vtt[t] = Qtt[t, idx_s, idx_s] + Qtt[t, idx_s, idx_a].dot(K[t])
                Vt[t] = Qt[t, idx_s] + Qtt[t, idx_s, idx_a].dot(k[t])
                Vtt[t] = 0.5 * (Vtt[t] + Vtt[t].T)


            if fail:
                old_eta = eta
                eta = eta0 + del_
                print('Increasing eta: {} -> {}'.format(old_eta, eta))
                del_ *= 2
                if eta >= 1e16:
                    raise ValueError(
                            'Failed to find PD solution even for very '
                            'large eta (check that dynamics and cost are '
                            'reasonably well conditioned)!'
                    )
        return (K, k, S, cS, iS), eta

    def compute_costs(self, eta):
        C, c = self.C, self.c
        C, c = C / eta, c / eta
        K, k, _, _, iS = self.policy_params

        for t in range(self.horizon - 1, -1, -1):
            C[t] += np.vstack([
                np.hstack([
                    K[t].T.dot(iS[t]).dot(K[t]),
                    -K[t].T.dot(iS[t]),
                ]),
                np.hstack([-iS[t].dot(K[t]), iS[t]]),
            ])
            c[t] += np.hstack([
                K[t].T.dot(iS[t]).dot(k[t]), -iS[t].dot(k[t])
            ])
        return C, c

    def traj_kl(self, new_policy_params, old_policy_params):
        mu, sigma = self.forward(new_policy_params)
        T, ds, da = self.horizon, self.ds, self.da

        K, k, S, cS, iS = old_policy_params
        K_, k_, S_, cS_, iS_ = new_policy_params
        kl_div = np.zeros(T)

        for t in range(T):
            logdet_prev = 2 * sum(np.log(np.diag(cS[t])))
            logdet_new = 2 * sum(np.log(np.diag(cS_[t])))

            K_diff, k_diff = K[t] - K_[t], k[t] - k_[t]
            mu_, sigma_ = mu[t, :ds], sigma[t, :ds, :ds]

            kl_div[t] = max(
                    0,
                    0.5 * (logdet_prev - logdet_new - da +
                           np.sum(np.diag(iS[t].dot(S_[t]))) +
                           k_diff.T.dot(iS[t]).dot(k_diff) +
                           mu_.T.dot(K_diff.T).dot(iS[t]).dot(K_diff).dot(mu_) +
                           np.sum(np.diag(K_diff.T.dot(iS[t]).dot(K_diff).dot(sigma_))) +
                           2 * k_diff.T.dot(iS[t]).dot(K_diff).dot(mu_)),
            )
        return np.sum(kl_div)
