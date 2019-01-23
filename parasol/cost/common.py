import six
from abc import ABCMeta, abstractmethod

@six.add_metaclass(ABCMeta)
class CostFunction(object):

    def __init__(self, ds, da, *args, **kwargs):
        self.ds, self.da = ds, da

    @abstractmethod
    def get_parameters(self):
        pass

    @abstractmethod
    def log_likelihood(self, states, costs):
        pass

    @abstractmethod
    def evaluate(self, states):
        pass

    def __getstate__(self):
        return {
            'ds': self.ds,
            'da': self.da,
        }

    def __setstate__(self, state):
        self.__init__(state['ds'], state['da'])

    @abstractmethod
    def is_cost_function(self):
        pass
