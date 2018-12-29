import numpy as np
from deepx import T, stats

from .common import Prior

__all__ = ['Normal']

class Normal(Prior):

    def get_parameters(self):
        return []

    def kl_and_grads(self, q_X, q_A, num_data):
        params = self.get_parameters()
        kl, info = self.kl_divergence(q_X, q_A, num_data)
        return kl, list(zip(params, T.grad(kl, params))), info

    def kl_divergence(self, q_X, q_A, num_data):
        mu_shape = T.shape(q_X.get_parameters('regular')[1])
        p_X = stats.Gaussian([
            T.eye(self.ds, batch_shape=mu_shape[:-1]),
            T.zeros(mu_shape)
        ])
        encoder_stdev = T.sqrt(T.core.matrix_diag_part(q_X.get_parameters('regular')[0]))
        return T.mean(T.sum(stats.kl_divergence(q_X, p_X), -1), 0), {'encoder-stdev': encoder_stdev}

    def has_dynamics(self):
        return False
