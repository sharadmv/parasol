from .mujoco import Reacher, Pointmass

ENVS = [Reacher, Pointmass]

ENV_MAP = { env.environment_name : env for env in ENVS }

def from_config(env_config):
    name = env_config.pop('environment_name')
    return make(name, **env_config)

def make(env_name, **kwargs):
    if env_name not in ENV_MAP:
        raise Exception("Environment %s does not exist" % env_name)
    return ENV_MAP[env_name](**kwargs)
