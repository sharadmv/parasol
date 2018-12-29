import copy
import pickle
import tensorflow as tf
import numpy as np
import tqdm
from deepx import T, stats

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
                 prior, smooth=False):
        super(VAE, self).__init__(do, du, horizon)
        self.ds, self.da = ds, da
        self.state_encoder = copy.deepcopy(state_encoder)
        self.state_decoder = copy.deepcopy(state_decoder)
        self.action_encoder = copy.deepcopy(action_encoder)
        self.action_decoder = copy.deepcopy(action_decoder)
        self.prior_params = prior
        if self.prior_params is None:
            self.prior_params = {'prior_type': 'none'}
        self.smooth = smooth
        self.initialize()

    def initialize(self):
        self.graph = T.core.Graph()
        with self.graph.as_default():
            prior_params = self.prior_params.copy()
            prior_type = prior_params.pop('prior_type')
            self.prior = PRIOR_MAP[prior_type](self.ds, self.da, self.horizon, **prior_params)

            self.O = T.placeholder(T.floatx(), [None, None, self.do])
            self.U = T.placeholder(T.floatx(), [None, None, self.du])
            self.S = T.placeholder(T.floatx(), [None, None, self.ds])
            self.A = T.placeholder(T.floatx(), [None, None, self.da])

            self.t = T.placeholder(T.int32, [])
            self.state, self.action = T.placeholder(T.floatx(), [None, self.ds]), T.placeholder(T.floatx(), [None, self.da])
            if self.prior.has_dynamics():
                self.next_state = self.prior.next_state(self.state, self.action, self.t)

            self.num_data = T.scalar()
            self.beta = T.placeholder(T.floatx(), [])
            self.learning_rate = T.placeholder(T.floatx(), [])
            self.model_learning_rate = T.placeholder(T.floatx(), [])

            batch_size = T.shape(self.O)[0]

            q_S = self.q_S = util.map_network(self.state_encoder)(self.O)
            q_O = self.q_O = util.map_network(self.state_decoder)(q_S.sample()[0])
            q_A = self.q_A = util.map_network(self.action_encoder)(self.U)
            q_U = self.q_U = util.map_network(self.action_decoder)(q_A.sample()[0])

            self.q_S_sample = self.q_S.sample()[0]
            self.q_A_sample = self.q_A.sample()[0]
            self.q_O_sample = self.q_O.sample()[0]
            self.q_U_sample = self.q_U.sample()[0]

            self.q_O_ = util.map_network(self.state_decoder)(self.S)
            self.q_U_ = util.map_network(self.action_decoder)(self.A)
            self.q_O__sample = self.q_O_.sample()[0]
            self.q_U__sample = self.q_U_.sample()[0]

            prior_kl, kl_grads, info = self.prior.kl_and_grads(q_S, q_A, self.num_data)

            log_likelihood = T.sum(q_O.log_likelihood(self.O), axis=1)

            elbo = T.mean(log_likelihood - prior_kl)
            train_elbo = T.mean(log_likelihood - self.beta * prior_kl)
            T.core.summary.scalar("log-likelihood", T.mean(log_likelihood))
            T.core.summary.scalar("prior-kl", T.mean(prior_kl))
            T.core.summary.scalar("beta", self.beta)
            T.core.summary.scalar("elbo", elbo)
            T.core.summary.scalar("beta-elbo", train_elbo)
            for k, v in info.items():
                T.core.summary.scalar(k, T.mean(v))
            self.summary = T.core.summary.merge_all()
            neural_params = (
                self.state_encoder.get_parameters()
                + self.state_decoder.get_parameters()
                + self.action_encoder.get_parameters()
                + self.action_decoder.get_parameters()
            )
            neural_params = (
                self.state_encoder.get_parameters()
                + self.state_decoder.get_parameters()
                + self.action_encoder.get_parameters()
                + self.action_decoder.get_parameters()
            )
            if len(neural_params) > 0:
                self.neural_op = T.core.train.AdamOptimizer(self.learning_rate).minimize(-train_elbo,
                                                                                        var_list=neural_params)
            else:
                self.neural_op = T.core.no_op()
            if len(kl_grads) > 0:
                if self.prior.has_natural_gradients():
                    opt = lambda x: T.core.train.MomentumOptimizer(x, 0.5)
                else:
                    opt = T.core.train.AdamOptimizer
                self.dynamics_op = opt(self.model_learning_rate).apply_gradients([
                    (b, a) for a, b in kl_grads
                ])
            else:
                self.dynamics_op = T.core.no_op()
            self.train_op = T.core.group(self.neural_op, self.dynamics_op)
        self.session = T.interactive_session(graph=self.graph, allow_soft_placement=True, log_device_placement=False)

    def make_summaries(self, env):
        with self.graph.as_default():
            if env.is_image():
                idx = T.random_uniform([], minval = 0, maxval = self.horizon - 1, dtype = T.int32)
                env.make_summary(self.q_O.get_parameters('regular')[0:1][:, idx], "reconstruction")
                env.make_summary(self.O[0:1][:, idx], "truth")
            self.summary = T.core.summary.merge_all()

    def train(self, rollouts,
              out_dir=None,
              num_epochs=100,
              batch_size=50,
              learning_rate=1e-4,
              model_learning_rate=None,
              dump_every=None, summary_every=1000,
              beta_increase=0,
              beta_start=0.0, beta_rate=1e-4, beta_end=1.0):
        beta = beta_start
        if model_learning_rate is None:
            model_learning_rate = learning_rate
        O, U = rollouts[0], rollouts[1]
        N = O.shape[0]
        if out_dir is None:
            writer = None
        else:
            writer = T.core.summary.FileWriter(out_dir / "tb", graph=self.graph)
        global_iter = 0
        for epoch in tqdm.trange(num_epochs, desc='Training'):
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
                        self.model_learning_rate: model_learning_rate,
                    })
                    if writer is not None:
                        writer.add_summary(summary, global_iter)
                        writer.flush()
                else:
                    self.session.run(self.train_op, {
                        self.O: O[batch_idx],
                        self.U: U[batch_idx],
                        self.beta: beta,
                        self.num_data: N,
                        self.learning_rate: learning_rate,
                        self.model_learning_rate: model_learning_rate,
                    })
                global_iter += 1
                if epoch >= beta_increase:
                    beta = min(beta_end, beta + beta_rate)
        if writer is not None:
            writer.flush()
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
        if out_dir is not None:
            with gfile.GFile(out_dir / "weights" / ("model-%s.pkl" % epoch), 'wb') as fp:
                pickle.dump(self, fp)

    def encode(self, o, u, sample=False):
        leading_dim = o.shape[:-1]
        o = o[(None,) * (2 - len(leading_dim))]
        u = u[(None,) * (2 - len(leading_dim))]
        if sample:
            s, a = self.session.run([self.q_S_sample, self.q_A_sample], {
                self.O: o,
                self.U: u,
            })
        else:
            s, a = self.session.run([self.q_S.expected_value(), self.q_A.expected_value()], {
                self.O: o,
                self.U: u,
            })
        return s.reshape(leading_dim + (-1,)), a.reshape(leading_dim + (-1,))

    def decode(self, s, sample=False):
        leading_dim = s.shape[:-1]
        s = s[(None,) * (2 - len(leading_dim))]
        if sample:
            o = self.session.run(self.q_O__sample, {
                self.S: s,
            })
        else:
            o = self.session.run(self.q_O_.expected_value(), {
                self.S: s,
            })
        return o.reshape(leading_dim + (-1,))

    def get_dynamics(self):
        return self.session.run(self.prior.get_dynamics())

    def has_dynamics(self):
        return self.prior.has_dynamics()

    def forward(self, state, action, t):
        assert self.has_dynamics()
        sigma, mu = self.session.run(self.next_state.get_parameters('regular'), {
            self.state: state,
            self.action: action,
            self.t: t
        })
        return mu, sigma
