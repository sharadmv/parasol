import numpy as np
np.set_printoptions(suppress=True, precision=4, linewidth=120)
from parasol.experiment import Solar

experiment = Solar(
    'reacher-lqrflm',
    {
        "environment_name": "Reacher",
        "random_start": False,
        "random_target": False,
    },
    # control={
        # 'control_type': 'mpc',
        # 'horizon': 3,
        # 'action_min': -10,
        # 'action_max': 10,
    # },
    control={
        'control_type': 'lqrflm',
        'horizon': 50,
        'init_std': 0.5,
        'kl_step': 1,
    },
    # model='s3://parasol-experiments/vae/reacher-noimage/blds/weights/model-final.pkl',
    model=None,
    horizon=50,
    seed=0,
    rollouts_per_iter=100,
    num_videos=1,
    out_dir='out/',
    model_train={
        'num_epochs': 0
    }
)
experiment.run(remote=False)
