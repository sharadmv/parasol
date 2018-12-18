from parasol.util import json
import parasol.gym as gym
from parasol.experiment import Solar

env_params = {
    "environment_name": "SimpleCar",
    "random_start": True,
    "random_target": True,
    "image": True,
    "image_dim": 32,
}
env = gym.from_config(env_params)
do = env.get_state_dim()
ds = 9
du = da = env.get_action_dim()

experiment = Solar(
    'test-solar',
    env_params,
    {}, 'temp2/test/nnds/weights/model-0.pkl',
    out_dir='out/',
)

print(json.dumps(experiment))
experiment.run()
