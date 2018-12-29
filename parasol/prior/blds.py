import numpy as np
from deepx import T, stats

from .lds import LDS

__all__ = ['BayesianLDS']

class BayesianLDS(LDS):

    def initialize_objective(self):
        H, ds, da = self.horizon, self.ds, self.da
        if self.time_varying:
            self.A_prior = stats.MNIW([
                T.eye(ds, batch_shape=[H - 1]),
                T.zeros([H - 1, ds, ds + da]),
                1e2 * T.eye(ds + da, batch_shape=[H - 1]),
                T.to_float(ds + da + 1) * T.ones([H - 1])
            ], parameter_type='regular')
            self.A_variational = stats.MNIW(list(map(T.variable, stats.MNIW.regular_to_natural([
                T.eye(ds, batch_shape=[H - 1]),
                T.random_normal([H - 1, ds, ds + da]),
                1e2 * T.eye(ds + da, batch_shape=[H - 1]),
                T.to_float(ds + da + 1) * T.ones([H - 1])
            ]))), parameter_type='natural')
        else:
            self.A_prior = stats.MNIW([
                (1) * T.eye(ds),
                T.zeros([ds, ds + da]),
                1e2 * T.eye(ds + da),
                T.to_float(ds + da + 1)
            ], parameter_type='regular')
            self.A_variational = stats.MNIW(list(map(T.variable, stats.MNIW.regular_to_natural([
                1 * T.eye(ds),
                T.random_normal([ds, ds + da]),
                1e2 * T.eye(ds + da),
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
        return self.A_variational.expected_sufficient_statistics()

    def has_natural_gradients(self):
        return True

    def get_parameters(self):
        return self.A_variational.get_parameters('natural')

    def kl_and_grads(self, q_X, q_A, num_data):
        kl, info = self.kl_divergence(q_X, q_A, num_data)
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
        (XtAt_XtAtT, XtAt), (Xt1_Xt1T, Xt1) = self.get_statistics(q_Xt, q_At, q_Xt1)
        batch_size = T.shape(XtAt)[0]
        num_batches = T.to_float(num_data) / T.to_float(batch_size)
        if self.time_varying:
            return kl, [(c, -(a + num_batches * b - c)) for a, b, c in zip(
                self.A_prior.get_parameters('natural'),
                [
                    T.sum(Xt1_Xt1T, [0]),
                    T.einsum('nha,nhb->hba', XtAt, Xt1),
                    T.sum(XtAt_XtAtT, [0]),
                    T.sum(T.ones([batch_size]), [0])
                ],
                self.A_variational.get_parameters('natural'),
            )], info
        else:
            return kl, [(c, -(a + num_batches * b - c)) for a, b, c in zip(
                self.A_prior.get_parameters('natural'),
                [
                    T.sum(Xt1_Xt1T, [0, 1]),
                    T.einsum('nha,nhb->ba', XtAt, Xt1),
                    T.sum(XtAt_XtAtT, [0, 1]),
                    T.sum(T.ones([batch_size, self.horizon - 1]), [0, 1])
                ],
                self.A_variational.get_parameters('natural'),
            )], info

    def kl_divergence(self, q_X, q_A, num_data):
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
            num_data = T.to_float(num_data)
            rmse = T.sqrt(T.sum(T.square(q_Xt1.get_parameters('regular')[1] - p_Xt1.get_parameters('regular')[1]), axis=-1))
            model_stdev = T.sqrt(T.core.matrix_diag_part(p_Xt1.get_parameters('regular')[0]))
            encoding_stdev = T.sqrt(T.core.matrix_diag_part(q_Xt1.get_parameters('regular')[0]))
            local_kl = T.mean(T.sum(stats.kl_divergence(q_Xt1, p_Xt1), axis=1), axis=0)
            if self.time_varying:
                global_kl = T.sum(stats.kl_divergence(self.A_variational, self.A_prior))
            else:
                global_kl = stats.kl_divergence(self.A_variational, self.A_prior)
            self.cache[(q_X, q_A)] = (
                local_kl + global_kl / T.to_float(num_data),
                {'rmse': rmse, 'encoder-stdev': encoding_stdev, 'model-stdev': model_stdev,
                 'local-kl': local_kl, 'global-kl': global_kl}
            )
        return self.cache[(q_X, q_A)]
