import pickle
import numpy as np
from deepx import T, stats, nn

from parasol import util
from .common import Dynamics

__all__ = ['NNDS']

class NNDS(Dynamics):

    def __init__(self, ds, da, horizon, network=None):
        super(NNDS, self).__init__(ds, da, horizon)
        assert network is not None, 'Must specify network'
        self.architecture = pickle.dumps(network)
        self.network = network
        self.cache = {}

    def forward(self, q_Xt, q_At):
        Xt, At = q_Xt.sample()[0], q_At.sample()[0]
        return util.map_network(self.network)(T.concatenate([Xt, At], -1))

    def get_dynamics(self):
        return self.network.get_parameters()

    def __getstate__(self):
        state = super(NNDS, self).__getstate__()
        state['architecture'] = self.architecture
        state['weights'] = T.get_current_session().run(self.get_parameters())
        return state

    def __setstate__(self, state):
        network = pickle.loads(state.pop('architecture'))
        weights = state.pop('weights')
        self.__init__(state['ds'], state['da'], network=network)
        T.get_current_session().run([T.core.assign(a, b) for a, b in zip(self.get_parameters(), weights)])

    def get_parameters(self):
        return self.network.get_parameters()

    def kl_divergence(self, q_X, q_A, _):
        # q_Xt - [N, H, ds]
        # q_At - [N, H, da]
        if (q_X, q_A) not in self.cache:
            q_Xt = q_X.__class__([
                q_X.get_parameters('regular')[0][:, :-1],
                q_X.get_parameters('regular')[1][:, :-1],
            ])
            q_At = q_A.__class__([
                q_A.get_parameters('regular')[0][:, :-1],
                q_A.get_parameters('regular')[1][:, :-1],
            ])
            p_Xt1 = self.forward(q_Xt, q_At)
            q_Xt1 = q_X.__class__([
                q_X.get_parameters('regular')[0][:, 1:],
                q_X.get_parameters('regular')[1][:, 1:],
            ])
            rmse = T.sqrt(T.sum(T.square(q_Xt1.get_parameters('regular')[1] - p_Xt1.get_parameters('regular')[1]), axis=-1))
            model_stdev = T.sqrt(p_Xt1.get_parameters('regular')[0])
            encoding_stdev = T.sqrt(q_Xt1.get_parameters('regular')[0])
            self.cache[(q_X, q_A)] = T.sum(stats.kl_divergence(q_Xt1, p_Xt1), axis=-1), {'rmse': rmse, 'encoding-stdev': encoding_stdev, 'model-stdev': model_stdev}
        return self.cache[(q_X, q_A)]

    def next_state(self, state, action, t):
        state_action = T.concatenate([state, action], -1)
        sigma, delta_mu = self.network(state_action).get_parameters('regular')
        return stats.Gaussian([
            sigma,
            delta_mu + state,
        ])

    def kl_gradients(self, q_X, q_A, kl, num_data):
        return T.grad(kl, self.get_parameters())

    def is_filtering_prior(self):
        return False
