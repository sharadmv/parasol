import numpy as np
from deepx import T, stats

from .common import Prior

__all__ = ['Normal']

class Normal(Prior):

    def get_parameters(self):
        return []

    def kl_and_grads(self, q_X, q_A, num_data):
        params = self.get_parameters()
        kl = self.kl_divergence(q_X, q_A, num_data)
        return kl, list(zip(params, T.grad(T.mean(kl, axis=0), params)))

    def kl_divergence(self, q_X, q_A, num_data):
        mu_shape = T.shape(q_X.get_parameters('regular')[1])
        p_X = stats.Gaussian([
            T.eye(self.ds, batch_shape=mu_shape[:-1]),
            T.zeros(mu_shape)
        ])
        return T.sum(stats.kl_divergence(q_X, p_X), -1)

    def has_dynamics(self):
        return False
