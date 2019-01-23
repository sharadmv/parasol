import pickle

from deepx import T, nn, stats

from parasol import util
from .common import CostFunction


class NNCost(CostFunction):

    def __init__(self, *args, **kwargs):
        super(NNCost, self).__init__(*args, **kwargs)
        network = kwargs.pop('network', None)
        assert network is not None, 'Must specify network'
        self.architecture = pickle.dumps(network)
        self.network = network

    def __getstate__(self):
        state = super(NNCost, self).__getstate__()
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

    def log_likelihood(self, states, costs):
        return T.sum(util.map_network(self.network)(states).log_likelihood(costs[..., None]), axis=-1)

    def evaluate(self, states):
        return util.map_network(self.network)(states).get_parameters('regular')[1]

    def is_cost_function(self):
        return True
