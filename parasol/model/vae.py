import pickle
import tensorflow as tf
import numpy as np
import tqdm
from deepx import T

from parasol import util
from parasol.prior import Normal, NNDS, LDS, BayesianLDS, NoPrior

from .common import Model

gfile = tf.gfile

PRIOR_MAP = {
    'nnds': NNDS,
    'normal': Normal,
    'lds': LDS,
    'blds': BayesianLDS,
    'none': NoPrior,
}

class VAE(Model):

    def __init__(self, do, du, horizon,
                 ds, da,
                 state_encoder, state_decoder,
                 action_encoder, action_decoder,
                 prior):
        super(VAE, self).__init__(do, du, horizon)
        self.ds, self.da = ds, da
        self.state_encoder = state_encoder
        self.state_decoder = state_decoder
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder
        self.prior_params = prior
        if self.prior_params is None:
            self.prior_params = {'prior_type': 'none'}
        self.initialize()

    def initialize(self):
        self.graph = T.core.Graph()
        with self.graph.as_default():
            prior_params = self.prior_params.copy()
            prior_type = prior_params.pop('prior_type')
            self.prior = PRIOR_MAP[prior_type](self.ds, self.da, self.horizon, **prior_params)

            self.O = T.placeholder(T.floatx(), [None, None, self.do])
            self.U = T.placeholder(T.floatx(), [None, None, self.du])
            self.num_data = T.scalar()
            self.beta = T.placeholder(T.floatx(), [])
            self.learning_rate = T.placeholder(T.floatx(), [])

            batch_size = T.shape(self.O)[0]

            q_S = self.q_S = util.map_network(self.state_encoder)(self.O)
            q_O = self.q_O = util.map_network(self.state_decoder)(q_S.sample()[0])
            q_A = self.q_A = util.map_network(self.action_encoder)(self.U)
            q_U = self.q_U = util.map_network(self.action_decoder)(q_A.sample()[0])

            prior_kl, kl_grads = self.prior.kl_and_grads(q_S, q_A, self.num_data)

            log_likelihood = T.sum(q_O.log_likelihood(self.O), axis=1)

            elbo = T.mean(log_likelihood - prior_kl)
            train_elbo = T.mean(log_likelihood - self.beta * prior_kl)
            T.core.summary.scalar("log-likelihood", T.mean(log_likelihood))
            T.core.summary.scalar("prior-kl", T.mean(prior_kl))
            T.core.summary.scalar("beta", self.beta)
            T.core.summary.scalar("elbo", elbo)
            T.core.summary.scalar("beta-elbo", train_elbo)
            neural_params = (
                self.state_encoder.get_parameters()
                + self.state_decoder.get_parameters()
                + self.action_encoder.get_parameters()
                + self.action_decoder.get_parameters()
            )
            self.neural_op = T.core.train.AdamOptimizer(self.learning_rate).minimize(-train_elbo,
                                                                                    var_list=neural_params)
            if len(kl_grads) > 0:
                if self.prior.has_natural_gradients():
                    opt = T.core.train.GradientDescentOptimizer
                else:
                    opt = T.core.train.AdamOptimizer
                self.dynamics_op = opt(self.learning_rate).apply_gradients([
                    (b, a) for a, b in kl_grads
                ])
            else:
                self.dynamics_op = T.core.no_op()
            self.train_op = T.core.group(self.neural_op, self.dynamics_op)
        self.session = T.interactive_session(graph=self.graph)

    def make_summaries(self, env):
        if env.is_image():
            idx = T.random_uniform([], minval = 0, maxval = self.horizon - 1, dtype = T.int32)
            env.make_summary(self.q_O.get_parameters('regular')[0:1][:, idx], "reconstruction")
            env.make_summary(self.O[0:1][:, idx], "truth")
        self.summary = T.core.summary.merge_all()

    def train(self, rollouts,
              out_dir='out/',
              num_epochs=100,
              batch_size=50,
              learning_rate=1e-4,
              dump_every=None, summary_every=1000,
              beta_start=0.0, beta_rate=1e-4, beta_end=1.0):
        beta = beta_start
        O, U = rollouts[0], rollouts[1]
        N = O.shape[0]
        writer = T.core.summary.FileWriter(out_dir / "tb")
        global_iter = 0
        for epoch in tqdm.trange(num_epochs, desc='Experiment'):
            if dump_every is not None and epoch % dump_every == 0:
                self.dump_weights(epoch, out_dir)
            permutation = np.random.permutation(N)

            for i in tqdm.trange(0, N, batch_size, desc="Epoch %u" % (epoch + 1)):
                batch_idx = permutation[slice(i, i + batch_size)]
                if global_iter % summary_every == 0:
                    summary, _ = self.session.run([self.summary, self.train_op], {
                        self.O: O[batch_idx],
                        self.U: U[batch_idx],
                        self.beta: beta,
                        self.num_data: N,
                        self.learning_rate: learning_rate,
                    })
                    writer.add_summary(summary, global_iter)
                    writer.flush()
                else:
                    self.session.run(self.train_op, {
                        self.O: O[batch_idx],
                        self.U: U[batch_idx],
                        self.beta: beta,
                        self.num_data: N,
                        self.learning_rate: learning_rate,
                    })
                global_iter += 1
                beta = min(beta_end, beta + beta_rate)
        self.dump_weights("final", out_dir)

    def __getstate__(self):
        state = super(VAE, self).__getstate__()
        state['ds'] = self.ds
        state['da'] = self.da
        state['state_encoder'] = self.state_encoder
        state['state_decoder'] = self.state_decoder
        state['action_encoder'] = self.action_encoder
        state['action_decoder'] = self.action_decoder
        state['prior_params'] = self.prior_params

        state['weights'] = self.get_weights()
        return state

    def __setstate__(self, state):
        weights = state.pop('weights')
        self.__dict__.update(state)
        self.initialize()
        self.set_weights(weights)

    def get_weights(self):
        return self.session.run((self.state_encoder.get_parameters(), self.state_decoder.get_parameters(), self.action_encoder.get_parameters(), self.action_decoder.get_parameters(), self.prior.get_parameters()))

    def set_weights(self, weights):
        self.session.run([T.core.assign(a, b) for a, b in zip(self.state_encoder.get_parameters(), weights[0])])
        self.session.run([T.core.assign(a, b) for a, b in zip(self.state_decoder.get_parameters(), weights[1])])
        self.session.run([T.core.assign(a, b) for a, b in zip(self.action_encoder.get_parameters(), weights[2])])
        self.session.run([T.core.assign(a, b) for a, b in zip(self.action_decoder.get_parameters(), weights[3])])
        self.session.run([T.core.assign(a, b) for a, b in zip(self.prior.get_parameters(), weights[4])])

    def dump_weights(self, epoch, out_dir):
        with gfile.GFile(out_dir / "weights" / ("model-%s.pkl" % epoch), 'wb') as fp:
            pickle.dump(self, fp)

    def decode(self):
        pass

    def encode(self):
        pass

    def get_dynamics(self):
        pass

    def has_dynamics(self):
        pass
