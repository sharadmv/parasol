import numpy as np
np.set_printoptions(suppress=True, precision=4)
from parasol.util import json
import parasol.gym as gym
from parasol.experiment import Solar

env_params = {
    "environment_name": "Pointmass",
    "random_start": True,
    "random_target": True,
}
env = gym.from_config(env_params)
experiment = Solar(
    'test-solar',
    env_params,
    {
        'control_type': 'mpc',
        'horizon': 100,
        'action_min': -0.05,
        'action_max': 0.05,
    },
    's3://parasol-experiments/pm-noimage/blds/weights/model-final.pkl',
    seed=1,
    out_dir='out/',
)
experiment.run()
