from deepx import nn
from parasol.experiment import run, sweep
import parasol.gym as gym

env_params = {
    "environment_name": "Reacher",
    "random_start": True,
    "random_target": True,
    "image": True,
    "image_dim": 64,
}
env = gym.from_config(env_params)
do = env.get_state_dim()
ds = 10
du = da = env.get_action_dim()
horizon = 50

experiment = dict(
    experiment_name='reacher-image',
    experiment_type='train_vae',
    environment=env_params,
    model=dict(
        do=do, du=du, ds=ds, da=da, horizon=horizon,
        state_encoder=(nn.Reshape(do, [64, 64, 3])
                    >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                    >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                    >> nn.Convolution([3, 3, 16]) >> nn.Relu()
                    >> nn.Flatten() >> nn.Relu(200) >> nn.Gaussian(ds)),
        state_decoder=(nn.Relu(ds, 1024) >> nn.Reshape([16, 16, 4])
                    >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                    >> nn.Deconvolution([2, 2, 32]) >> nn.Relu()
                    >> nn.Convolution([3, 3, 32]) >> nn.Relu()
                    >> nn.Deconvolution([2, 2, 32]) >> nn.Relu()
                    >> nn.Convolution([2, 2, 3]) >> nn.Flatten() >> nn.Bernoulli()),
        action_encoder=nn.IdentityVariance(),
        action_decoder=nn.IdentityVariance(),
        prior={
            'prior_type': sweep(['normal', 'none', 'lds', 'blds']),
        }
    ),
    train=dict(
        num_epochs=2000,
        learning_rate=2e-4,
        batch_size=2,
        dump_every=100,
        summary_every=200,
    ),
    data=dict(
        num_rollouts=100,
        init_std=0.5,
    ),
    seed=0,
    out_dir='s3://parasol-experiments/vae/reacher-image-gpu',
)
run(experiment, remote=True, gpu=True, num_threads=4, instance_type='m5.4xlarge')
