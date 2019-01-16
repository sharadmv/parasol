from path import Path

from deepx import nn
from parasol.experiment import run, sweep
import parasol.gym as gym

env_params = {
    "environment_name": "Pendulum",
    "image": True,
}
env = gym.from_config(env_params)
do = env.get_state_dim()
ds = 3
du = da = env.get_action_dim()
horizon = 200

experiment = dict(
    experiment_name='pendulum-image-blds-smooth',
    experiment_type='train_vae',
    env=env_params,
    model=dict(
        do=do, du=du, ds=ds, da=da, horizon=horizon,
        state_encoder=(nn.Reshape(do, [32, 32, 3])
                    >> nn.Convolution([5, 5, 32], strides=(1, 1)) >> nn.Relu()
                    >> nn.Convolution([3, 3, 8], strides=(2, 2))
                    >> nn.Flatten() >> nn.Relu(256) >> nn.Gaussian(ds)),
        state_decoder=(nn.Relu(ds, 512) >> nn.Reshape([8, 8, 8])
                    >> nn.Deconvolution([3, 3, 32])
                    >> nn.Deconvolution([5, 5, 3])
                    >> nn.Flatten() >> nn.Bernoulli()),
        action_encoder=nn.IdentityVariance(variance=1e-4),
        action_decoder=nn.IdentityVariance(variance=1e-4),
        prior=sweep([
            dict(prior_type='blds', smooth=True),
            dict(prior_type='blds', smooth=False),
        ], ['blds-smooth', 'blds']),
    ),
    train=dict(
        num_epochs=1000,
        learning_rate=1e-3,
        model_learning_rate=1e-5 / 4,
        beta_start=1e-4,
        beta_end=10.0,
        beta_rate=5e-5,
        beta_increase=0,
        batch_size=2,
        dump_every=100,
        summary_every=50,
    ),
    data=dict(
        load_data=Path('s3://parasol-experiments/data/pendulum-200.pkl'),
        num_rollouts=100,
        init_std=2.0,
        smooth_noise=False,
    ),
    dump_data=True,
    seed=0,
    out_dir='s3://parasol-experiments/vae/pendulum-image',
)
run(experiment, remote=True, gpu=False, num_threads=1, instance_type='m5.4xlarge')
