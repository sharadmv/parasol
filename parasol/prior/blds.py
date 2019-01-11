import numpy as np
from deepx import T, stats

from .lds import LDS

__all__ = ['BayesianLDS']

class BayesianLDS(LDS):

    def initialize_objective(self):
        H, ds, da = self.horizon, self.ds, self.da
        if self.time_varying:
            A = T.concatenate([H - 1, T.eye(ds), T.zeros([ds, da])], -1)
            self.A_prior = stats.MNIW([
                T.eye(ds, batch_shape=[H - 1]),
                A,
                1e1 * T.eye(ds + da, batch_shape=[H - 1]),
                T.to_float(ds + da + 1) * T.ones([H - 1])
            ], parameter_type='regular')
            self.A_variational = stats.MNIW(list(map(T.variable, stats.MNIW.regular_to_natural([
                T.eye(ds, batch_shape=[H - 1]),
                A + 1e-2 * T.random_normal([H - 1, ds, ds + da]),
                1e1 * T.eye(ds + da, batch_shape=[H - 1]),
                T.to_float(ds + da + 1) * T.ones([H - 1])
            ]))), parameter_type='natural')
        else:
            A = T.concatenate([T.eye(ds), T.zeros([ds, da])], -1)
            self.A_prior = stats.MNIW([
                (1) * T.eye(ds),
                A,
                T.eye(ds + da),
                T.to_float(ds + da + 1)
            ], parameter_type='regular')
            self.A_variational = stats.MNIW(list(map(T.variable, stats.MNIW.regular_to_natural([
                1 * T.eye(ds),
                A + 1e-2 * T.random_normal([ds, ds + da]),
                T.eye(ds + da),
                T.to_float(ds + da)
            ]))), parameter_type='natural')

    def get_dynamics(self):
        if self.time_varying:
            Q, A = self.A_variational.expected_value()
            return (
                A,
                Q,
            )
            return A, Q
        else:
            Q, A = self.A_variational.expected_value()
            return (
                T.tile(A[None], [self.horizon - 1, 1, 1]),
                T.tile(Q[None], [self.horizon - 1, 1, 1]),
            )

    def sufficient_statistics(self):
        if self.time_varying:
            return self.A_variational.expected_sufficient_statistics()
        else:
            stats = self.A_variational.expected_sufficient_statistics()
            return [
                T.tile(stats[0][None], [self.horizon - 1, 1, 1]),
                T.tile(stats[1][None], [self.horizon - 1, 1, 1]),
                T.tile(stats[2][None], [self.horizon - 1, 1, 1]),
                T.tile(stats[3][None], [self.horizon - 1]),
            ]

    def has_natural_gradients(self):
        return True

    def get_parameters(self):
        return self.A_variational.get_parameters('natural')

    def kl_gradients(self, q_X, q_A, _, num_data):
        if self.smooth:
            ds = self.ds
            ess = q_X.expected_sufficient_statistics()
            yyT = ess[..., :-1, ds:2*ds, ds:2*ds]
            xxT = ess[..., :-1, :ds, :ds]
            yxT = ess[..., :-1, ds:2*ds, :ds]
            aaT, a = stats.Gaussian.unpack(q_A.expected_sufficient_statistics())
            aaT, a = aaT[:, :-1], a[:, :-1]
            x = ess[..., :-1, -1, :ds]
            y = ess[..., :-1, -1, ds:2*ds]
            xaT = T.outer(x, a)
            yaT = T.outer(y, a)
            xaxaT = T.concatenate([
                T.concatenate([xxT, xaT], -1),
                T.concatenate([T.matrix_transpose(xaT), aaT], -1),
            ], -2)
            batch_size = T.shape(ess)[0]
            num_batches = T.to_float(num_data) / T.to_float(batch_size)
            ess = [
                yyT,
                T.concatenate([yxT, yaT], -1),
                xaxaT,
                T.ones([batch_size, self.horizon - 1])
            ]
        else:
            q_Xt = q_X.__class__([
                q_X.get_parameters('regular')[0][:, :-1],
                q_X.get_parameters('regular')[1][:, :-1],
            ])
            q_At = q_A.__class__([
                q_A.get_parameters('regular')[0][:, :-1],
                q_A.get_parameters('regular')[1][:, :-1],
            ])
            q_Xt1 = q_X.__class__([
                q_X.get_parameters('regular')[0][:, 1:],
                q_X.get_parameters('regular')[1][:, 1:],
            ])
            (XtAt_XtAtT, XtAt), (Xt1_Xt1T, Xt1) = self.get_statistics(q_Xt, q_At, q_Xt1)
            batch_size = T.shape(XtAt)[0]
            num_batches = T.to_float(num_data) / T.to_float(batch_size)
            ess = [
                Xt1_Xt1T,
                T.einsum('nha,nhb->nhba', XtAt, Xt1),
                XtAt_XtAtT,
                T.ones([batch_size, self.horizon - 1])
            ]
        if self.time_varying:
            ess = [
                T.sum(ess[0], [0]),
                T.sum(ess[1], [0]),
                T.sum(ess[2], [0]),
                T.sum(ess[3], [0]),
            ]
        else:
            ess = [
                T.sum(ess[0], [0, 1]),
                T.sum(ess[1], [0, 1]),
                T.sum(ess[2], [0, 1]),
                T.sum(ess[3], [0, 1]),
            ]
        return [-(a + num_batches * b - c) / T.to_float(num_data) for a, b, c in zip(
            self.A_prior.get_parameters('natural'),
            ess,
            self.A_variational.get_parameters('natural'),
        )]

    def kl_divergence(self, q_X, q_A, num_data):
        if (q_X, q_A) not in self.cache:
            if self.smooth:
                state_prior = stats.GaussianScaleDiag([
                    T.ones(self.ds),
                    T.zeros(self.ds)
                ])
                self.p_X = stats.LDS(
                    (self.sufficient_statistics(), state_prior, None, q_A.expected_value(), self.horizon),
                'internal')
                local_kl = stats.kl_divergence(q_X, self.p_X)
                if self.time_varying:
                    global_kl = T.sum(stats.kl_divergence(self.A_variational, self.A_prior))
                else:
                    global_kl = stats.kl_divergence(self.A_variational, self.A_prior)
                prior_kl = T.mean(local_kl, axis=0) + global_kl / T.to_float(num_data)
                A, Q = self.get_dynamics()
                model_stdev = T.sqrt(T.matrix_diag_part(Q))
                self.cache[(q_X, q_A)] = prior_kl, {
                    'local-kl': local_kl,
                    'global-kl': global_kl,
                    'model-stdev': model_stdev,
                }
            else:
                q_Xt = q_X.__class__([
                    q_X.get_parameters('regular')[0][:, :-1],
                    q_X.get_parameters('regular')[1][:, :-1],
                ])
                q_At = q_A.__class__([
                    q_A.get_parameters('regular')[0][:, :-1],
                    q_A.get_parameters('regular')[1][:, :-1],
                ])
                p_Xt1 = self.forward(q_Xt, q_At)
                q_Xt1 = q_X.__class__([
                    q_X.get_parameters('regular')[0][:, 1:],
                    q_X.get_parameters('regular')[1][:, 1:],
                ])
                num_data = T.to_float(num_data)
                rmse = T.sqrt(T.sum(T.square(q_Xt1.get_parameters('regular')[1] - p_Xt1.get_parameters('regular')[1]), axis=-1))
                A, Q = self.get_dynamics()
                model_stdev = T.sqrt(T.matrix_diag_part(Q))
                encoding_stdev = T.sqrt(T.matrix_diag_part(q_Xt1.get_parameters('regular')[0]))
                local_kl = T.sum(stats.kl_divergence(q_Xt1, p_Xt1), axis=1)
                if self.time_varying:
                    global_kl = T.sum(stats.kl_divergence(self.A_variational, self.A_prior))
                else:
                    global_kl = stats.kl_divergence(self.A_variational, self.A_prior)
                self.cache[(q_X, q_A)] = (
                    T.mean(local_kl, axis=0) + global_kl / T.to_float(num_data),
                    {'rmse': rmse, 'encoder-stdev': encoding_stdev, 'model-stdev': model_stdev,
                    'local-kl': local_kl, 'global-kl': global_kl}
                )
        return self.cache[(q_X, q_A)]
