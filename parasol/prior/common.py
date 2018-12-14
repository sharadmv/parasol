import six
from abc import ABCMeta, abstractmethod

from deepx import T

@six.add_metaclass(ABCMeta)
class Prior(object):

    def __init__(self, ds, da, horizon):
        self.ds, self.da = ds, da
        self.horizon = horizon

    @abstractmethod
    def get_parameters(self):
        pass

    def kl_and_grads(self, q_X, q_A, num_data):
        params = self.get_parameters()
        kl = self.kl_divergence(q_X, q_A, num_data)
        return kl, list(zip(params, T.grad(T.mean(kl, axis=0), params)))

    @abstractmethod
    def kl_divergence(self, q_X, q_A, num_data):
        pass

    @abstractmethod
    def has_dynamics(self):
        pass

class Dynamics(Prior):

    @abstractmethod
    def forward(self, q_Xt, q_At):
        pass

    def kl_and_grads(self, q_X, q_A, num_data):
        params = self.get_parameters()
        kl = self.kl_divergence(q_X, q_A, num_data)
        return kl, list(zip(params, T.grad(T.mean(kl, axis=0), params)))

    def has_dynamics(self):
        return True

    @abstractmethod
    def get_dynamics(self):
        pass
