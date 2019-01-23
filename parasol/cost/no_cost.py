from deepx import T

from .common import CostFunction

class NoCost(CostFunction):

    def get_parameters(self):
        return []

    def log_likelihood(self, states, costs):
        return T.zeros_like(costs)

    def evaluate(self, states):
        raise Exception("Cannot evaluate NoCost function")

    def is_cost_function(self):
        return False
