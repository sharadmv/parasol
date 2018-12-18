from deepx import nn
from parasol.util import json
from parasol.experiment import TrainVAE, from_json
import parasol.gym as gym

env_params = {
    "environment_name": "Pointmass",
    "random_start": True,
    "random_target": True,
    "image": True,
    "image_dim": 32,
}
env = gym.from_config(env_params)
do = env.get_state_dim()
ds = 4
du = da = env.get_action_dim()
horizon = 50

experiment = TrainVAE(
    "nnds",
    env_params,
    dict(
        do=do, du=du, ds=ds, da=da, horizon=horizon,
        state_encoder=(nn.Reshape(do, [32, 32, 2])
                       >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                       >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                       >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                       >> nn.Flatten() >> nn.Relu(200) >> nn.Gaussian(ds)),
        state_decoder=(nn.Relu(ds, 1024) >> nn.Reshape([16, 16, 4])
                       >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                       >> nn.Deconvolution([2, 2, 32]) >> nn.Relu()
                       >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                       >> nn.Convolution([2, 2, 2]) >> nn.Flatten() >> nn.Bernoulli()),
        action_encoder=nn.IdentityVariance(),
        action_decoder=nn.IdentityVariance(),
        # prior={'prior_type': 'none'},
        # prior={'prior_type': 'normal'},
        # prior={'prior_type': 'lds'},
        # prior={'prior_type': 'blds'},
        prior={'prior_type': 'nnds', 'network': nn.Relu(ds + da, 200) >> nn.Relu(200) >> nn.Gaussian(ds)},
    ),
    train=dict(
        num_epochs=1000,
        learning_rate=1e-4,
        batch_size=2,
        dump_every=50,
        summary_every=50 / 2,
    ),
    data=dict(
        num_rollouts=100,
        policy_variance=1
    ),
    out_dir='temp2/test',
)
experiment.run(remote=False, instance_type='m5.4xlarge')
