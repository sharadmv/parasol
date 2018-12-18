import six
from abc import abstractmethod, ABCMeta

@six.add_metaclass(ABCMeta)
class Model(object):

    def __init__(self, do, du, horizon):
        self.do = do
        self.du = du
        self.horizon = horizon

    @abstractmethod
    def train(self, rollouts):
        pass

    @abstractmethod
    def encode(self, y, a):
        pass

    @abstractmethod
    def decode(self, x):
        pass

    @abstractmethod
    def get_dynamics(self):
        pass

    @property
    @abstractmethod
    def has_dynamics(self):
        pass

    @abstractmethod
    def forward(self, state, action, t):
        pass

    def make_summaries(self, env):
        pass

    def __getstate__(self):
        return {
            'do': self.do,
            'du': self.du,
            'horizon': self.horizon,
        }
