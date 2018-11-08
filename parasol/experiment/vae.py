import random
import tqdm
import numpy as np
from deepx import T, stats
import json
import parasol.gym as gym
from .common import Experiment

class TrainVAE(Experiment):

    experiment_type = "train_vae"

    def __init__(self,
                 experiment_name,
                 env, encoder, decoder,
                 observation_dimension,
                 latent_dimension,
                 seed=0,
                 num_rollouts=100,
                 horizon = 50,
                 policy_variance = 1.,
                 num_epochs=100,
                 learning_rate=1e-4,
                 batch_size=20,
                 beta_start=0.0, beta_rate=1e-4,
                 beta_end=1.0, **kwargs):
        super(TrainVAE, self).__init__(experiment_name, **kwargs)
        self.env = env
        self.seed = seed
        self.encoder = encoder
        self.decoder = decoder
        self.observation_dimension = observation_dimension
        self.latent_dimension = latent_dimension
        self.policy_variance = policy_variance
        self.num_rollouts = num_rollouts
        self.num_epochs = num_epochs
        self.horizon = horizon
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.beta_start = beta_start
        self.beta_rate = beta_rate
        self.beta_end = beta_end
        self.initialize_objective()

    def initialize_objective(self):
        self.X = T.placeholder(T.floatx(), [None, self.observation_dimension])
        self.beta = T.placeholder(T.floatx(), [])
        batch_size = T.shape(self.X)[0]

        p_Z = stats.Gaussian([
            T.eye(self.latent_dimension, batch_shape=[batch_size]),
            T.zeros([batch_size, self.latent_dimension])]
        )
        q_Z = self.encoder(self.X)
        q_X = self.decoder(q_Z.sample()[0])
        log_likelihood = T.mean(q_X.log_likelihood(self.X), axis=0)
        prior_kl = T.mean(stats.kl_divergence(q_Z, p_Z), axis=0)

        elbo = log_likelihood - prior_kl
        train_elbo = log_likelihood - self.beta * prior_kl
        T.core.summary.scalar("log-likelihood", log_likelihood)
        T.core.summary.scalar("prior-kl", prior_kl)
        T.core.summary.scalar("beta", self.beta)
        T.core.summary.scalar("elbo", elbo)
        T.core.summary.scalar("beta_elbo", train_elbo)
        self.train_op = T.core.train.AdamOptimizer(self.learning_rate).minimize(-train_elbo)
        self.summary = T.core.summary.merge_all()

    def to_dict(self):
        return {
            "seed": self.seed,
            "environment": self.env,
            "experiment_name": self.experiment_name,
            "experiment_type": self.experiment_type,
            "architecture": {
                "state_encoder": self.encoder,
                "state_decoder": self.decoder,
            },
            "data": {
                "num_rollouts": self.num_rollouts,
                "horizon": self.horizon,
                "policy_variance": self.policy_variance,
            },
            "train": {
                "num_epochs": self.num_epochs,
                "learning_rate": self.learning_rate,
                "beta_start": self.beta_start,
                "beta_rate": self.beta_rate,
                "beta_end": self.beta_end,
                "batch_size": self.batch_size,
            },
            "observation_dimension": self.observation_dimension,
            "latent_dimension": self.latent_dimension,
        }

    def from_dict(self, params):
        return TrainVAE(
            params['experiment_name'],
            params['experiment_type'],
            params['environment'],
            params['architecture']['state_encoder'],
            params['architecture']['state_decoder'],
            params['observation_dimension'],
            params['latent_dimension'],
            seed=params['seed'],
            num_epochs=params['train']['num_epochs'],
            learning_rate=params['train']['learning_rate'],
            beta_start=params['train']['beta_start'],
            beta_rate=params['train']['beta_rate'],
            beta_end=params['train']['beta_end'],
            batch_size=params['train']['batch_size'],
            num_rollouts=params['data']['num_rollouts'],
            horizon=params['data']['horizon'],
            policy_variance=params['data']['policy_variance'],
        )

    def to_json(self):
        params = self.to_dict()
        params['architecture']['state_encoder'] = str(self.encoder)
        params['architecture']['state_decoder'] = str(self.decoder)
        return json.dumps(params)

    def run_experiment(self, out_dir):
        T.core.set_random_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)

        env = gym.from_config(self.env)

        def policy(x, _):
            return np.random.multivariate_normal(mean=np.zeros(env.get_action_dim()), cov=np.eye(env.get_action_dim()) * self.policy_variance)
        S = []
        for i in tqdm.trange(self.num_rollouts, desc="Data"):
            S.append(env.rollout(self.horizon, policy=policy)[0])
        S = S.reshape([-1, env.get_state_dim()])

        beta = self.beta_start
        N = S.shape[0]
        writer = T.core.summary.FileWriter(out_dir / "tb")
        global_iter = 0
        with T.session() as sess:
            for epoch in tqdm.trange(self.num_epochs, desc='Experiment'):
                permutation = np.random.permutation(N)

                for i in tqdm.trange(0, N, self.batch_size, desc="Epoch %u" % (epoch + 1)):
                    batch_idx = slice(i, i + self.batch_size)
                    if global_iter % 1000 == 0:
                        summary, _ = sess.run([self.summary, self.train_op], {
                            self.X: S[batch_idx],
                            self.beta: beta
                        })
                        writer.add_summary(summary, global_iter)
                        writer.flush()
                    else:
                        sess.run(self.train_op, {
                            self.X: S[batch_idx],
                            self.beta: beta
                        })
                    global_iter += 1
                    beta = min(self.beta_end, beta + self.beta_rate)
