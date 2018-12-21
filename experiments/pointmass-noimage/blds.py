from parasol.experiment import Solar

experiment = Solar(
    'pointmass-noimage',
    {
        "environment_name": "Pointmass",
        "random_start": True,
        "random_target": True,
    },
    control={
        'control_type': 'lqrflm',
        'horizon': 50,
        'kl_step': 1,
        'init_std': 0.5,
    },
    # model='s3://parasol-experiments/pm-noimage/blds/weights/model-final.pkl',
    horizon=50,
    seed=0,
    rollouts_per_iter=100,
    num_videos=1,
    model=None,
    out_dir='out/',
    model_train={
        'num_epochs': 0
    }
)
experiment.run(remote=False)
