from parasol.experiment import Solar

experiment = Solar(
    'pointmass-noimage-nnds',
    {
        "environment_name": "Pointmass",
        "random_start": True,
        "random_target": True,
    },
    control={
        'control_type': 'mpc',
        'horizon': 1,
        'action_min': -1,
        'action_max': 1,
    },
    model='s3://parasol-experiments/pm-noimage/blds/weights/model-final.pkl',
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
