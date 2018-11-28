import six
from abc import ABCMeta, abstractmethod

@six.add_metaclass(ABCMeta)
class Dynamics(object):

    @abstractmethod
    def forward(self, q_Xt, q_At):
        pass

    @abstractmethod
    def kl_gradients(self, q_Xt, q_At, q_Xt1):
        pass

    @abstractmethod
    def kl_divergence(self, q_Xt, q_At, q_Xt1):
        pass
