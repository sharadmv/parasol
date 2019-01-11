from parasol.experiment import run, sweep

experiment = dict(
    experiment_type='solar',
    experiment_name='pendulum-lqrflm',
    env={
        'environment_name': 'Pendulum',
    },
    control={
        'control_type': 'lqrflm',
        'prior_type': 'gmm',
        'horizon': 200,
        'init_std': 2.,
        'kl_step': 0.1,
    },
    model=None,
    horizon=200,
    seed=0,
    rollouts_per_iter=20,
    num_iters=20,
    buffer_size=100,
    smooth_noise=True,
    num_videos=2,
    out_dir='out/solar/pendulum',
    model_train={
        'num_epochs': 0
    }
)
run(experiment, remote=False, num_threads=1, instance_type='m5.4xlarge')
