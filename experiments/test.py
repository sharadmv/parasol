from deepx import nn
from parasol.util import json
from parasol.experiment import TrainVAE, from_json
import parasol.gym as gym

env_params = {
    "environment_name": "SimpleCar",
    "random_start": True,
    "random_target": True,
    "image": True,
    "image_dim": 32,
}
env = gym.from_config(env_params)
do = env.get_state_dim()
ds = 9
du = da = env.get_action_dim()

experiment = TrainVAE(
    "test-bigger-nnds",
    env_params,
    (nn.Reshape(do, [32, 32, 1])
    >> nn.Convolution([3, 3, 32]) >> nn.Relu()
    >> nn.Convolution([3, 3, 32]) >> nn.Relu()
    >> nn.Convolution([3, 3, 32]) >> nn.Relu()
    >> nn.Flatten() >> nn.Relu(200) >> nn.Gaussian(ds)),
    (nn.Relu(ds, 1024) >> nn.Reshape([16, 16, 4])
    >> nn.Convolution([3, 3, 32]) >> nn.Relu()
    >> nn.Deconvolution([2, 2, 32]) >> nn.Relu()
    >> nn.Convolution([3, 3, 32]) >> nn.Relu()
    >> nn.Convolution([2, 2, 1]) >> nn.Flatten() >> nn.Bernoulli()),
    nn.IdentityVariance(),
    nn.IdentityVariance(),
    do, ds,
    du, da,
    # prior={'prior_type': 'lds'},
    prior={'prior_type': 'nnds', 'network': nn.Relu(ds + da, 200) >> nn.Relu(200) >> nn.Gaussian(ds)},
    # prior=None,
    num_epochs=1000,
    num_rollouts=100,
    learning_rate=1e-4,
    out_dir='s3://parasol-experiments/test/',
    batch_size=2,
    summary_every=10
)
print(experiment.run(remote=True))
