from parasol.experiment import run

experiment = dict(
    experiment_type='solar',
    experiment_name='lqrflm',
    env={
        'environment_name': 'Reacher',
        'image': True,
        'default_goal': [0.1, 0.1],
        'random_start': False,
        'random_target': False,
        'pd_cost': True,
        'easy_cost': False,
    },
    control=dict(
        control_type='lqrflm',
        data_strength=50,
        prior_type='model',
        horizon=50,
        init_std=0.5,
        kl_step=10,
    ),
    model='data/vae/reacher-image/weights/model-final.pkl',
    horizon=50,
    seed=0,
    rollouts_per_iter=40,
    num_iters=10,
    buffer_size=None,
    smooth_noise=False,
    num_videos=2,
    out_dir='data/solar/reacher-image/',
    model_train={
        'num_epochs': 0
    }
)
run(
    experiment,
    remote=False,
    num_threads=1,
    instance_type='m5.4xlarge'
)
