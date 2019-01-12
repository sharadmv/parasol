import numpy as np
from deepx import T, stats

from .common import Prior

__all__ = ['Normal']

class Normal(Prior):

    def get_parameters(self):
        return []

    def kl_gradients(self, q_X, q_A, kl, num_data):
        return []

    def kl_divergence(self, q_X, q_A, num_data):
        mu_shape = T.shape(q_X.get_parameters('regular')[1])
        p_X = stats.GaussianScaleDiag([
            T.ones(mu_shape),
            T.zeros(mu_shape)
        ])
        return T.mean(T.sum(stats.kl_divergence(q_X, p_X), -1), 0), {}

    def has_dynamics(self):
        return False
