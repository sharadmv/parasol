import numpy as np
np.set_printoptions(suppress=True, precision=4, linewidth=120)
from parasol.experiment import run, sweep

experiment = dict(
    experiment_type='solar',
    experiment_name='reacher-lqrflm',
    env={
        "environment_name": "Reacher",
        "random_start": False,
        "random_target": False,
    },
    control={
        'control_type': 'lqrflm',
        'prior_type': 'gmm',
        'horizon': 50,
        'init_std': sweep([0.3, 0.5]),
        'kl_step': sweep([0.1, 1])
    },
    # model='s3://parasol-experiments/vae/reacher-noimage/blds/weights/model-final.pkl',
    model=None,
    horizon=50,
    seed=0,
    rollouts_per_iter=sweep([20, 50]),
    num_iters=10,
    num_videos=5,
    out_dir='s3://parasol-experiments/solar/sweep-test',
    model_train={
        'num_epochs': 0
    }
)
run(experiment, remote=True)
