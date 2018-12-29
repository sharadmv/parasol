from parasol.experiment import run, sweep

experiment = dict(
    experiment_type='solar',
    experiment_name='reacher-mpc',
    env={
        "environment_name": "Reacher",
    },
    control={
        'control_type': 'mpc',
        'horizon': 20,
    },
    # model='s3://parasol-experiments/vae/reacher-image/reacher-image_model{prior{prior_type}}-blds/weights/model-1700.pkl',
    model='out/reacher_mpc/nnds/weights/model-final.pkl',
    horizon=50,
    seed=0,
    rollouts_per_iter=2,
    num_iters=20,
    num_videos=2,
    out_dir='out/solar/reacher_mpc',
    model_train={
        'num_epochs': 100
    }
)
run(experiment, remote=False)