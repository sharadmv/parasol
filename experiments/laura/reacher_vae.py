from deepx import nn
from parasol.experiment import TrainVAE
import parasol.gym as gym

env_params = {
    "environment_name": "Reacher",
    "random_start": False,
    "random_target": False,
    # "image": True,
    # "image_dim": 32,
}
env = gym.from_config(env_params)
do = env.get_state_dim()
ds = do
du = da = env.get_action_dim()
horizon = 50

experiment = TrainVAE(
    "nnds",
    env_params,
    dict(
        do=do, du=du, ds=ds, da=da, horizon=horizon,
        # state_encoder=(nn.Reshape(do, [32, 32, 2])
                       # >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                       # >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                       # >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                       # >> nn.Flatten() >> nn.Relu(200) >> nn.Gaussian(ds)),
        # state_decoder=(nn.Relu(ds, 1024) >> nn.Reshape([16, 16, 4])
                       # >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                       # >> nn.Deconvolution([2, 2, 32]) >> nn.Relu()
                       # >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                       # >> nn.Convolution([2, 2, 2]) >> nn.Flatten() >> nn.Bernoulli()),
        state_encoder=nn.IdentityVariance(),
        state_decoder=nn.IdentityVariance(),
        action_encoder=nn.IdentityVariance(),
        action_decoder=nn.IdentityVariance(),
        # prior={'prior_type': 'none'},
        # prior={'prior_type': 'normal'},
        # prior={'prior_type': 'lds'},
        # prior={'prior_type': 'blds'},
        prior={'prior_type': 'nnds', 'network': nn.Relu(ds + da, 200) >> nn.Relu(200) >> nn.Gaussian(ds)},
    ),
    train=dict(
        num_epochs=4000,
        learning_rate=1e-3,
        batch_size=16,
        dump_every=1000,
        summary_every=50 / 2,
        beta_start=1.0, 
        beta_rate=0,
    ),
    data=dict(
        num_rollouts=100,
        init_std=.2,
    ),
    #out_dir='s3://parasol-experiments/vae/reacher-noimage',
    out_dir='out/reacher_mpc',
)
experiment.run(remote=False)