from .common import Model

class NoModel(Model):

    def __init__(self, do, du, horizon):
        self.do = self.ds = do
        self.du = self.da = du
        self.horizon = horizon

    def train(self, rollouts, **kwargs):
        pass

    def encode(self, y, a):
        return y, a

    def filter(self, y, a, t):
        return y[t], a[t]

    def decode(self, x):
        return x

    def get_dynamics(self):
        raise Exception()

    def has_dynamics(self):
        return False

    def forward(self, state, action, t):
        raise Exception()

    def make_summaries(self, env):
        pass

    def __getstate__(self):
        return {
            'do': self.do,
            'du': self.du,
            'horizon': self.horizon,
        }
