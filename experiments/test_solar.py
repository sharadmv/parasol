from parasol.experiment import run, sweep

experiment = dict(
    experiment_type='solar',
    experiment_name='pendulum-lqrflm',
    env={
        "environment_name": "Pendulum",
    },
    control={
        'control_type': 'lqrflm',
        'prior_type': 'gmm',
        'horizon': 100,
        'init_std': 5,
        'kl_step': 0.1,
    },
    # model='s3://parasol-experiments/vae/reacher-image/reacher-image_model{prior{prior_type}}-blds/weights/model-1700.pkl',
    model=None,
    horizon=100,
    seed=0,
    rollouts_per_iter=100,
    num_iters=20,
    num_videos=2,
    out_dir='out/solar/pendulum',
    model_train={
        'num_epochs': 0
    }
)
run(experiment, remote=False, num_threads=1, instance_type='m5.4xlarge')
