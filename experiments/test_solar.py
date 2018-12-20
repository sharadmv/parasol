import numpy as np
np.set_printoptions(suppress=True, precision=4)
import parasol.gym as gym
from parasol.experiment import Solar

env_params = {
    "environment_name": "SimpleCar",
    "random_start": True,
    "random_target": False,
}
env = gym.from_config(env_params)
experiment = Solar(
    'test-solar',
    env_params,
    control={
        'control_type': 'lqrflm',
        'horizon': 100,
        'init_std': 10,
        'kl_step': 10,
    },
    model='s3://parasol-experiments/car-noimage/blds/weights/model-final.pkl',
    horizon=100,
    seed=1,
    num_videos=0,
    out_dir='out/',
    model_train={
        'num_epochs': 0
    }
)
experiment.run()
