import numpy as np
from deepx import T, stats

from .common import Dynamics

__all__ = ['LDS']

class LDS(Dynamics):

    def __init__(self, state_dim, action_dim, horizon=50, time_varying=False):
        self.ds, self.da = state_dim, action_dim
        self.horizon = horizon
        self.time_varying = time_varying
        self.cache = {}
        self.initialize_objective()

    def initialize_objective(self):
        H, ds, da = self.horizon, self.ds, self.da
        if self.time_varying:
            self.A = T.variable(T.random_normal([H - 1, ds, ds + da]))
            self.Q_log_diag = T.variable(T.random_normal([H - 1, ds]))
            self.Q = T.matrix_diag(T.exp(self.Q_log_diag))
        else:
            self.A = T.variable(T.random_normal([ds, ds + da]))
            self.Q_log_diag = T.variable(T.random_normal([ds]))
            self.Q = T.matrix_diag(T.exp(self.Q_log_diag))

    def forward(self, q_Xt, q_At):
        Xt, At = q_Xt.expected_value(), q_At.expected_value()
        batch_size = T.shape(Xt)[0]
        XAt = T.concatenate([Xt, At], -1)
        A, Q = self.get_dynamics()
        p_Xt1 = stats.Gaussian([
            T.tile(Q[None], [batch_size, 1, 1, 1]),
            T.einsum('nhs,hxs->nhx', XAt, A)
        ])
        return p_Xt1

    def get_dynamics(self):
        if self.time_varying:
            return self.A, self.Q
        else:
            return (
                T.tile(self.A[None], [self.horizon - 1, 1, 1]),
                T.tile(self.Q[None], [self.horizon - 1, 1, 1])
            )


    def kl_gradients(self, q_X, q_A, _):
        params = [self.A, self.Q_log_diag]
        return list(zip(params, T.grad(self.kl_divergence(q_X, q_A, _), params)))

    def get_statistics(self, q_Xt, q_At, q_Xt1):
        Xt1_Xt1T, Xt1 = stats.Gaussian.unpack(q_Xt1.expected_sufficient_statistics())

        Xt_XtT, Xt = stats.Gaussian.unpack(q_Xt.expected_sufficient_statistics())
        At_AtT, At = stats.Gaussian.unpack(q_At.expected_sufficient_statistics())

        XtAt = T.concatenate([Xt, At], -1)
        XtAt_XtAtT = T.concatenate([
            T.concatenate([Xt_XtT, T.outer(Xt, At)], -1),
            T.concatenate([T.outer(At, Xt), At_AtT], -1),
        ], -2)
        return (XtAt_XtAtT, XtAt), (Xt1_Xt1T, Xt1)

    def kl_divergence(self, q_X, q_A, _):
        # q_Xt - [N, H, ds]
        # q_At - [N, H, da]
        if (q_X, q_A) not in self.cache:
            q_Xt = stats.Gaussian([
                q_X.get_parameters('regular')[0][:, :-1],
                q_X.get_parameters('regular')[1][:, :-1],
            ])
            q_At = stats.Gaussian([
                q_A.get_parameters('regular')[0][:, :-1],
                q_A.get_parameters('regular')[1][:, :-1],
            ])
            p_Xt1 = self.forward(q_Xt, q_At)
            q_Xt1 = stats.Gaussian([
                q_X.get_parameters('regular')[0][:, 1:],
                q_X.get_parameters('regular')[1][:, 1:],
            ])
            self.cache[(q_X, q_A)] = T.mean(stats.kl_divergence(q_Xt1, p_Xt1), axis=-1)
        return self.cache[(q_X, q_A)]
