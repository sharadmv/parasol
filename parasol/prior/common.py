import six
from abc import ABCMeta, abstractmethod

from deepx import T

@six.add_metaclass(ABCMeta)
class Prior(object):

    def __init__(self, ds, da, horizon):
        self.ds, self.da = ds, da
        self.horizon = horizon

    def encode(self, q_X, q_A):
        return q_X, q_A

    @abstractmethod
    def get_parameters(self):
        pass

    def posterior_kl_grads(self, q_X, q_A, num_data):
        q_X, q_A = self.encode(q_X, q_A)
        kl, info = self.kl_divergence(q_X, q_A, num_data)
        grads = self.kl_gradients(q_X, q_A, kl, num_data)
        return (q_X, q_A), kl, list(zip(self.get_parameters(), grads)), info

    def has_natural_gradients(self):
        return False

    @abstractmethod
    def kl_gradients(self, q_X, q_A, kl, num_data):
        pass

    @abstractmethod
    def kl_divergence(self, q_X, q_A, num_data):
        pass

    @abstractmethod
    def has_dynamics(self):
        pass

    def __getstate__(self):
        return {
            'ds': self.ds,
            'da': self.da,
            'horizon': self.horizon,
        }

    def __setstate__(self, state):
        self.__init__(state['ds'], state['da'], state['horizon'])

class Dynamics(Prior):

    @abstractmethod
    def forward(self, q_Xt, q_At):
        pass

    def kl_and_grads(self, q_X, q_A, num_data):
        params = self.get_parameters()
        kl, info = self.kl_divergence(q_X, q_A, num_data)
        return kl, list(zip(params, T.grad(kl, params))), info

    def has_dynamics(self):
        return True

    @abstractmethod
    def next_state(self, state, action, t):
        pass

    @abstractmethod
    def get_dynamics(self):
        pass
