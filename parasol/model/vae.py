import copy
import pickle
import tensorflow as tf
import numpy as np
import tqdm
from deepx import T, stats

from parasol import util
import parasol.prior as prior
import parasol.cost as cost

from .common import Model

gfile = tf.gfile

PRIOR_MAP = {
    'nnds': prior.NNDS,
    'normal': prior.Normal,
    'lds': prior.LDS,
    'blds': prior.BayesianLDS,
    'none': prior.NoPrior,
}

COST_MAP = {
    'nn': cost.NNCost,
    'quadratic': cost.Quadratic,
    'none': cost.NoCost,
}

class VAE(Model):

    def __init__(self, do, du, horizon,
                 ds, da,
                 state_encoder, state_decoder,
                 action_encoder, action_decoder,
                 prior, cost=None):
        super(VAE, self).__init__(do, du, horizon)
        self.ds, self.da = ds, da
        self.state_encoder = copy.deepcopy(state_encoder)
        self.state_decoder = copy.deepcopy(state_decoder)
        self.action_encoder = copy.deepcopy(action_encoder)
        self.action_decoder = copy.deepcopy(action_decoder)
        self.prior_params = prior
        if self.prior_params is None:
            self.prior_params = {'prior_type': 'none'}
        self.cost_params = cost
        if self.cost_params is None:
            self.cost_params = {'cost_type': 'none'}

    def initialize(self):
        self.graph = T.core.Graph()
        with self.graph.as_default():
            prior_params = self.prior_params.copy()
            prior_type = prior_params.pop('prior_type')
            self.prior = PRIOR_MAP[prior_type](self.ds, self.da, self.horizon, **prior_params)

            cost_params = self.cost_params.copy()
            cost_type = cost_params.pop('cost_type')
            self.cost = COST_MAP[cost_type](self.ds, self.da, **cost_params)

            self.O = T.placeholder(T.floatx(), [None, None, self.do])
            self.U = T.placeholder(T.floatx(), [None, None, self.du])
            self.C = T.placeholder(T.floatx(), [None, None])
            self.S = T.placeholder(T.floatx(), [None, None, self.ds])
            self.A = T.placeholder(T.floatx(), [None, None, self.da])

            self.t = T.placeholder(T.int32, [])
            self.state, self.action = T.placeholder(T.floatx(), [None, self.ds]), T.placeholder(T.floatx(), [None, self.da])
            if self.prior.has_dynamics():
                self.next_state = self.prior.next_state(self.state, self.action, self.t)
                self.prior_dynamics = self.prior.get_dynamics()

            self.num_data = T.scalar()
            self.beta = T.placeholder(T.floatx(), [])
            self.learning_rate = T.placeholder(T.floatx(), [])
            self.model_learning_rate = T.placeholder(T.floatx(), [])

            self.S_potentials = util.map_network(self.state_encoder)(self.O)
            self.A_potentials = util.map_network(self.action_encoder)(self.U)

            if self.prior.has_dynamics():
                self.prior_dynamics_stats = self.prior.sufficient_statistics()

            if self.prior.is_dynamics_prior():
                self.data_strength = T.placeholder(T.floatx(), [])
                self.max_iter = T.placeholder(T.int32, [])
                posterior_dynamics, (encodings, actions) = \
                        self.prior.posterior_dynamics(self.S_potentials, self.A_potentials,
                                                      data_strength=self.data_strength,
                                                      max_iter=self.max_iter)
                self.posterior_dynamics_ = posterior_dynamics, (encodings.expected_value(), actions.expected_value())

            if self.prior.is_filtering_prior():
                self.dynamics_stats = (
                    T.placeholder(T.floatx(), [None, self.ds, self.ds]),
                    T.placeholder(T.floatx(), [None, self.ds, self.ds + self.da]),
                    T.placeholder(T.floatx(), [None, self.ds + self.da, self.ds + self.da]),
                    T.placeholder(T.floatx(), [None]),
                )
                S_natparam = self.S_potentials.get_parameters('natural')
                num_steps = T.shape(S_natparam)[1]

                self.padded_S = stats.Gaussian(T.core.pad(
                    self.S_potentials.get_parameters('natural'),
                    [[0, 0], [0, self.horizon - num_steps], [0, 0], [0, 0]]
                ), 'natural')
                self.padded_A = stats.GaussianScaleDiag([
                    T.core.pad(self.A_potentials.get_parameters('regular')[0],
                            [[0, 0], [0, self.horizon - num_steps], [0, 0]]),
                    T.core.pad(self.A_potentials.get_parameters('regular')[1],
                            [[0, 0], [0, self.horizon - num_steps], [0, 0]])
                ], 'regular')
                self.q_S_padded, self.q_A_padded = self.prior.encode(
                    self.padded_S, self.padded_A,
                    dynamics_stats=self.dynamics_stats
                )
                self.q_S_filter = self.q_S_padded.filter(max_steps=num_steps)
                self.q_A_filter = self.q_A_padded.__class__(
                    self.q_A_padded.get_parameters('natural')[:, :num_steps]
                , 'natural')
                self.e_q_S_filter = self.q_S_filter.expected_value()
                self.e_q_A_filter = self.q_A_filter.expected_value()

            (self.q_S, self.q_A), self.prior_kl, self.kl_grads, self.info = self.prior.posterior_kl_grads(
                self.S_potentials, self.A_potentials, self.num_data
            )

            self.q_S_sample = self.q_S.sample()[0]
            self.q_A_sample = self.q_A.sample()[0]

            self.q_O = util.map_network(self.state_decoder)(self.q_S_sample)
            self.q_U = util.map_network(self.action_decoder)(self.q_A_sample)
            self.q_O_sample = self.q_O.sample()[0]
            self.q_U_sample = self.q_U.sample()[0]

            self.q_O_ = util.map_network(self.state_decoder)(self.S)
            self.q_U_ = util.map_network(self.action_decoder)(self.A)
            self.q_O__sample = self.q_O_.sample()[0]
            self.q_U__sample = self.q_U_.sample()[0]

            self.cost_likelihood = self.cost.log_likelihood(self.q_S_sample, self.C)
            self.log_likelihood = T.sum(self.q_O.log_likelihood(self.O), axis=1)

            self.elbo = T.mean(self.log_likelihood + self.cost_likelihood - self.prior_kl)
            train_elbo = T.mean(self.log_likelihood + self.beta * (self.cost_likelihood - self.prior_kl))
            T.core.summary.scalar("encoder-stdev", T.mean(self.S_potentials.get_parameters('regular')[0]))
            T.core.summary.scalar("log-likelihood", T.mean(self.log_likelihood))
            T.core.summary.scalar("cost-likelihood", T.mean(self.cost_likelihood))
            T.core.summary.scalar("prior-kl", T.mean(self.prior_kl))
            T.core.summary.scalar("beta", self.beta)
            T.core.summary.scalar("elbo", self.elbo)
            T.core.summary.scalar("beta-elbo", train_elbo)
            for k, v in self.info.items():
                T.core.summary.scalar(k, T.mean(v))
            self.summary = T.core.summary.merge_all()
            neural_params = (
                self.state_encoder.get_parameters()
                + self.state_decoder.get_parameters()
                + self.action_encoder.get_parameters()
                + self.action_decoder.get_parameters()
            )
            cost_params = self.cost.get_parameters()
            if len(neural_params) > 0:
                optimizer = T.core.train.AdamOptimizer(self.learning_rate)
                gradients, variables = zip(*optimizer.compute_gradients(-train_elbo, var_list=neural_params))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.neural_op = optimizer.apply_gradients(zip(gradients, variables))
            else:
                self.neural_op = T.core.no_op()
            if len(cost_params) > 0:
                self.cost_op = T.core.train.AdamOptimizer(self.learning_rate).minimize(-self.elbo, var_list=cost_params)
            else:
                self.cost_op = T.core.no_op()
            if len(self.kl_grads) > 0:
                if self.prior.is_dynamics_prior():
                    # opt = lambda x: T.core.train.MomentumOptimizer(x, 0.5)
                    opt = lambda x: T.core.train.GradientDescentOptimizer(x)
                else:
                    opt = T.core.train.AdamOptimizer
                self.dynamics_op = opt(self.model_learning_rate).apply_gradients([
                    (b, a) for a, b in self.kl_grads
                ])
            else:
                self.dynamics_op = T.core.no_op()
            self.train_op = T.core.group(self.neural_op, self.dynamics_op, self.cost_op)
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
        O, U, C = rollouts[0], rollouts[1], rollouts[2]
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
                        self.C: C[batch_idx],
                        self.beta: beta,
                        self.num_data: N,
                        self.learning_rate: learning_rate,
                        self.model_learning_rate: model_learning_rate,
                    })
                    if writer is not None:
                        writer.add_summary(summary, global_iter)
                else:
                    self.session.run(self.train_op, {
                        self.O: O[batch_idx],
                        self.U: U[batch_idx],
                        self.C: C[batch_idx],
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
        state['cost_params'] = self.cost_params
        state['weights'] = self.get_weights()
        return state

    def __setstate__(self, state):
        weights = state.pop('weights')
        self.__dict__.update(state)
        self.initialize()
        self.set_weights(weights)

    def get_weights(self):
        return self.session.run((
            self.state_encoder.get_parameters(),
            self.state_decoder.get_parameters(),
            self.action_encoder.get_parameters(),
            self.action_decoder.get_parameters(),
            self.prior.get_parameters(),
            self.cost.get_parameters(),
        ))

    def set_weights(self, weights):
        self.session.run([T.core.assign(a, b) for a, b in zip(self.state_encoder.get_parameters(), weights[0])])
        self.session.run([T.core.assign(a, b) for a, b in zip(self.state_decoder.get_parameters(), weights[1])])
        self.session.run([T.core.assign(a, b) for a, b in zip(self.action_encoder.get_parameters(), weights[2])])
        self.session.run([T.core.assign(a, b) for a, b in zip(self.action_decoder.get_parameters(), weights[3])])
        self.session.run([T.core.assign(a, b) for a, b in zip(self.prior.get_parameters(), weights[4])])
        if len(weights) > 5:
            self.session.run([T.core.assign(a, b) for a, b in zip(self.cost.get_parameters(), weights[5])])

    def dump_weights(self, epoch, out_dir):
        if out_dir is not None:
            with gfile.GFile(out_dir / "weights" / ("model-%s.pkl" % epoch), 'wb') as fp:
                pickle.dump(self, fp)

    def filter(self, o, u, t, sample=False, dynamics=None):
        leading_dim = o.shape[:-2]
        if len(leading_dim) == 0:
            o, u = o[None], u[None]
        if sample:
            if self.prior.is_filtering_prior():
                raise NotImplementedError
            else:
                s, a = self.session.run([self.q_S_sample, self.q_A_sample], {
                    self.O: o[..., t:t + 1, :],
                    self.U: u[..., t:t + 1, :],
                })
        else:
            if self.prior.is_filtering_prior():
                if dynamics is None:
                    s, a = self.session.run([self.e_q_S_filter, self.e_q_A_filter], {
                        self.O: o[..., :t + 1, :],
                        self.U: u[..., :t + 1, :],
                        self.dynamics_stats: self.session.run(self.prior_dynamics_stats)
                    })
                else:
                    s, a = self.session.run([self.e_q_S_filter, self.e_q_A_filter], {
                        self.O: o[..., :t + 1, :],
                        self.U: u[..., :t + 1, :],
                        self.dynamics_stats: dynamics
                        # self.dynamics_stats: self.session.run(self.prior_dynamics_stats)
                    })
            else:
                s, a = self.session.run([self.q_S.expected_value(), self.q_A.expected_value()], {
                    self.O: o[..., t:t + 1, :],
                    self.U: u[..., t:t + 1, :],
                })
        s, a = s[..., -1, :], a[..., -1, :]
        if len(leading_dim) == 0:
            s, a = s[0], a[0]
        return s, a

    def encode(self, o, u, sample=False):
        leading_dim = o.shape[:-2]
        if len(leading_dim) == 0:
            o, u = o[None], u[None]
        if sample:
            def encode_chunk(_, o, u):
                return self.session.run([self.q_S_sample, self.q_A_sample], {
                    self.O: o,
                    self.U: u
                })
        else:
            def encode_chunk(_, o, u):
                return self.session.run([self.q_S.expected_value(), self.q_A.expected_value()], {
                    self.O: o,
                    self.U: u
                })
        s, a = util.chunk_map(encode_chunk, o, u, show_progress="Encoding")
        if len(leading_dim) == 0:
            s, a = s[0], a[0]
        return s, a

    def decode(self, s, sample=False):
        raise NotImplementedError
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
        return self.session.run(self.prior_dynamics)

    def get_prior_parameters(self):
        return self.session.run(self.prior.get_parameters())

    def posterior_dynamics(self, rollouts, data_strength=1.0, max_iter=100):
        assert self.prior.is_dynamics_prior()
        return self.session.run(self.posterior_dynamics_, {
            self.O: rollouts[0],
            self.U: rollouts[1],
            self.data_strength: data_strength,
            self.max_iter: max_iter,
        })

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
