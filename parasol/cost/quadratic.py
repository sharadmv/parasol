from deepx import T, stats

from .common import CostFunction

class Quadratic(CostFunction):

    def __init__(self, *args, **kwargs):
        self.pd = kwargs.pop('pd', False)
        self.cost_stdev = kwargs.pop('cost_stdev', 1.0)
        self.learn_stdev = kwargs.pop('learn_stdev', False)
        super(Quadratic, self).__init__(*args, **kwargs)
        self.initialize_objective()

    def initialize_objective(self):
        self.C, self.c = (
            T.variable(T.random_normal([self.ds, self.ds])),
            T.variable(T.random_normal([self.ds])),
        )
        if self.learn_stdev:
            self.stdev = T.variable(T.to_float(self.cost_stdev))
        else:
            self.stdev = T.to_float(self.cost_stdev)

    def get_parameters(self):
        if self.learn_stdev:
            return [self.C, self.c, self.stdev]
        return [self.C, self.c]

    def log_likelihood(self, states, costs):
        mean = (
            0.5 * T.einsum('nia,ab,nib->ni', states, self.C, states)
            + T.einsum('nia,a->ni', states, self.c)
        )
        stdev = T.ones_like(mean) * self.stdev
        return stats.GaussianScaleDiag([
            stdev, mean
        ]).log_likelihood(costs)
