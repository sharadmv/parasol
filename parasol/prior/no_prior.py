from deepx import T

from .common import Prior

__all__ = ['NoPrior']

class NoPrior(Prior):

    def get_parameters(self):
        return []

    def kl_and_grads(self, q_X, q_A, num_data):
        kl, info = self.kl_divergence(q_X, q_A, num_data)
        return kl, [], info

    def kl_divergence(self, q_X, q_A, num_data):
        batch_size = T.shape(q_X.expected_value())[0]
        return T.zeros(batch_size), {}

    def has_dynamics(self):
        return False
