from path import Path
import pickle
import random
import tqdm
import numpy as np
import deepx
from deepx import T, stats
import parasol.gym as gym
import parasol.util as util
from tensorflow import gfile

from .common import Experiment
from parasol.prior import Normal, NNDS, LDS, BayesianLDS, NoPrior

PRIOR_MAP = {
    'nnds': NNDS,
    'normal': Normal,
    'lds': LDS,
    'blds': BayesianLDS,
    'none': NoPrior,
}

class TrainVAE(Experiment):

    experiment_type = "train_vae"

    def __init__(self,
                 experiment_name,
                 env,
                 state_encoder, state_decoder,
                 action_encoder, action_decoder,
                 do,
                 ds,
                 du,
                 da,
                 seed=0,
                 num_rollouts=100,
                 horizon=50,
                 policy_variance=1.,
                 num_epochs=1000,
                 learning_rate=1e-4,
                 batch_size=20,
                 prior=None,
                 dump_every=None,
                 summary_every=1000,
                 beta_start=0.0, beta_rate=1e-4,
                 beta_end=1.0, **kwargs):
        super(TrainVAE, self).__init__(experiment_name, **kwargs)
        self.env_params = env
        self.seed = seed
        self.state_encoder = state_encoder
        self.state_decoder = state_decoder
        self.action_encoder = action_encoder
        self.action_decoder = action_decoder
        self.architecture = (state_encoder, state_decoder, action_encoder, action_decoder)

        self.do, self.ds = do, ds
        self.du, self.da = du, da

        self.policy_variance = policy_variance
        self.num_rollouts = num_rollouts
        self.num_epochs = num_epochs
        self.dump_every = dump_every
        self.summary_every = summary_every
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.prior_params = prior

        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_rate = beta_rate
        self.beta_end = beta_end

        self.env = gym.from_config(self.env_params)

        self.graph = T.core.Graph()
        with self.graph.as_default():
            self.initialize_objective()

    def initialize_objective(self):
        self.O = T.placeholder(T.floatx(), [None, None, self.do])
        self.U = T.placeholder(T.floatx(), [None, None, self.du])
        self.num_data = T.scalar()
        self.beta = T.placeholder(T.floatx(), [])
        batch_size = T.shape(self.O)[0]

        q_S = util.map_network(self.state_encoder)(self.O)
        q_O = util.map_network(self.state_decoder)(q_S.sample()[0])
        q_A = util.map_network(self.action_encoder)(self.U)
        q_U = util.map_network(self.action_decoder)(q_A.sample()[0])

        if self.prior_params is None:
            self.prior_params = {'prior_type': 'none'}
        prior_params = self.prior_params.copy()
        prior_type = prior_params.pop('prior_type')
        self.prior = PRIOR_MAP[prior_type](self.ds, self.da, self.horizon, **prior_params)
        prior_kl, kl_grads = self.prior.kl_and_grads(q_S, q_A, self.num_data)

        log_likelihood = T.sum(q_O.log_likelihood(self.O), axis=1)

        elbo = T.mean(log_likelihood - prior_kl)
        train_elbo = T.mean(log_likelihood - self.beta * prior_kl)
        T.core.summary.scalar("log-likelihood", T.mean(log_likelihood))
        T.core.summary.scalar("prior-kl", T.mean(prior_kl))
        T.core.summary.scalar("beta", self.beta)
        T.core.summary.scalar("elbo", elbo)
        T.core.summary.scalar("beta_elbo", train_elbo)
        if self.env.is_image():
            image_size = self.env.image_size()
            idx = T.random_uniform([], minval = 0, maxval = self.horizon - 1, dtype = T.int32)
            reconstruction = T.reshape(q_O.get_parameters('regular')[0:1][:, idx], [-1] + image_size)
            truth = T.reshape(self.O[0:1][:, idx], [-1] + image_size)
            T.core.summary.image('reconstruction', reconstruction)
            T.core.summary.image('original', truth)
        neural_params = (
            self.state_encoder.get_parameters()
            + self.state_decoder.get_parameters()
            + self.action_encoder.get_parameters()
            + self.action_decoder.get_parameters()
        )
        self.neural_op = T.core.train.AdamOptimizer(self.learning_rate).minimize(-train_elbo,
                                                                                var_list=neural_params)
        if len(kl_grads) > 0:
            self.dynamics_op = T.core.train.AdamOptimizer(self.learning_rate).apply_gradients([
                (b, a) for a, b in kl_grads
            ])
        else:
            self.dynamics_op = T.core.no_op()
        self.train_op = T.core.group(self.neural_op, self.dynamics_op)
        self.summary = T.core.summary.merge_all()

    def to_dict(self):
        return {
            "seed": self.seed,
            "out_dir": self.out_dir,
            "environment": self.env_params,
            "experiment_name": self.experiment_name,
            "experiment_type": self.experiment_type,
            "architecture": {
                "state_encoder": self.state_encoder,
                "state_decoder": self.state_decoder,
                "action_encoder": self.action_encoder,
                "action_decoder": self.action_decoder,
            },
            "prior": self.prior_params.copy() if self.prior_params is not None else None,
            "data": {
                "num_rollouts": self.num_rollouts,
                "horizon": self.horizon,
                "policy_variance": self.policy_variance,
            },
            "train": {
                "num_epochs": self.num_epochs,
                "dump_every": self.dump_every,
                "summary_every": self.summary_every,
                "learning_rate": self.learning_rate,
                "beta_start": self.beta_start,
                "beta_rate": self.beta_rate,
                "beta_end": self.beta_end,
                "batch_size": self.batch_size,
            },
            "do": self.do,
            "ds": self.ds,
            "du": self.du,
            "da": self.da,
        }

    @classmethod
    def from_dict(cls, params):
        return TrainVAE(
            params['experiment_name'],
            params['environment'],
            params['architecture']['state_encoder'],
            params['architecture']['state_decoder'],
            params['architecture']['action_encoder'],
            params['architecture']['action_decoder'],
            params['do'],
            params['ds'],
            params['du'],
            params['da'],
            seed=params['seed'],
            prior=params['prior'],
            num_epochs=params['train']['num_epochs'],
            learning_rate=params['train']['learning_rate'],
            beta_start=params['train']['beta_start'],
            beta_rate=params['train']['beta_rate'],
            beta_end=params['train']['beta_end'],
            batch_size=params['train']['batch_size'],
            summary_every=params['train']['summary_every'],
            dump_every=params['train']['dump_every'],
            num_rollouts=params['data']['num_rollouts'],
            out_dir=params['out_dir'],
            horizon=params['data']['horizon'],
            policy_variance=params['data']['policy_variance'],
        )

    def dump_weights(self, sess, epoch, out_dir):
        with gfile.GFile(out_dir / "weights" / ("network-%s.pkl" % epoch), 'wb') as fp:
            weights = sess.run((self.state_encoder.get_parameters(), self.state_decoder.get_parameters(), self.action_encoder.get_parameters(), self.action_decoder.get_parameters()))
            pickle.dump((self.architecture, weights), fp)
        with gfile.GFile(out_dir / "weights" / ("prior-%s.pkl" % epoch), 'wb') as fp:
            pickle.dump(self.prior, fp)

    def run_experiment(self, out_dir):
        out_dir = Path(out_dir)

        T.core.set_random_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        env = self.env

        def policy(x, _):
            return np.random.multivariate_normal(mean=np.zeros(env.get_action_dim()), cov=np.eye(env.get_action_dim()) * self.policy_variance)
        rollouts = env.rollouts(self.num_rollouts, self.horizon, policy=policy, show_progress=True)
        O, U = rollouts[0], rollouts[1]

        beta = self.beta_start
        N = O.shape[0]
        writer = T.core.summary.FileWriter(out_dir / "tb")
        global_iter = 0
        with T.session(graph=self.graph) as sess:
            for epoch in tqdm.trange(self.num_epochs, desc='Experiment'):
                if self.dump_every is not None and epoch % self.dump_every == 0:
                    self.dump_weights(sess, epoch, out_dir)
                permutation = np.random.permutation(N)

                for i in tqdm.trange(0, N, self.batch_size, desc="Epoch %u" % (epoch + 1)):
                    batch_idx = slice(i, i + self.batch_size)
                    if global_iter % self.summary_every == 0:
                        summary, _ = sess.run([self.summary, self.train_op], {
                            self.O: O[batch_idx],
                            self.U: U[batch_idx],
                            self.beta: beta,
                            self.num_data: N,
                        })
                        writer.add_summary(summary, global_iter)
                        writer.flush()
                    else:
                        sess.run(self.train_op, {
                            self.O: O[batch_idx],
                            self.U: U[batch_idx],
                            self.beta: beta,
                            self.num_data: N,
                        })
                    global_iter += 1
                    beta = min(self.beta_end, beta + self.beta_rate)
            self.dump_weights(sess, "final", out_dir)
