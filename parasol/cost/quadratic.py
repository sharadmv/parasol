from deepx import T, stats

from .common import CostFunction

class Quadratic(CostFunction):

    def __init__(self, *args, **kwargs):
        self.pd = kwargs.pop('pd', False)
        self.cost_stdev = kwargs.pop('cost_stdev', 1.0)
        super(Quadratic, self).__init__(*args, **kwargs)
        self.initialize_objective()

    def initialize_objective(self):
        self.C, self.c = (
            T.variable(T.random_normal([self.ds, self.ds])),
            T.variable(T.random_normal([self.ds])),
        )

    def get_parameters(self):
        return [self.C, self.c]

    def log_likelihood(self, states, costs):
        mean = T.einsum('nia,ab,nib->ni', states, self.C, states)
        stdev = T.ones_like(mean) * self.cost_stdev
        return stats.GaussianScaleDiag([
            stdev, mean
        ]).log_likelihood(costs)
