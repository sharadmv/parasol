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
    experiment_name='pendulum-image',
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
        prior=dict(prior_type='lds', smooth=True),
    ),
    train=dict(
        num_epochs=1000,
        learning_rate=1e-3,
        model_learning_rate=1e-3,
        beta_start=1e-4,
        beta_end=10.0,
        beta_rate=5e-5,
        beta_increase=0,
        batch_size=1,
        dump_every=100,
        summary_every=10,
    ),
    data=dict(
        # load_data=Path('out/vae/pendulum/rollouts.pkl'),
        num_rollouts=100,
        init_std=2.0,
        smooth_noise=True,
    ),
    dump_data=True,
    seed=0,
    out_dir='out/vae/pendulum',
)
run(experiment, remote=False, gpu=True, num_threads=1, instance_type='g3.4xlarge')
