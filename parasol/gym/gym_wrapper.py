import gym
from gym.envs.registration import register
from .utils import VideoRecorder

from .env import ParasolEnvironment

class GymWrapper(ParasolEnvironment):

    version = 1

    def __init__(self, config):
        env_name = "Parasol%s-v%u" % (self.environment_name, self.version)
        register(
            id=env_name,
            entry_point=self.entry_point,
            max_episode_steps=self.max_episode_steps,
            reward_threshold=self.reward_threshold,
            kwargs=config
        )
        GymWrapper.version += 1
        self._config = config
        self.__dict__.update(config)
        self.gym_env = gym.make(env_name)
        self.state = None
        super(GymWrapper, self).__init__(config['sliding_window'])

    def reset(self):
        self.state = self.gym_env.reset()
        return self.observe()

    def step(self, action):
        self.state, reward, done, info = self.gym_env.step(action)
        return self.observe(), reward, done, info

    def _observe(self):
        return self.state

    def render(self, mode='human'):
        return self.gym_env.render(mode=mode)

    def state_dim(self):
        return self.gym_env.observation_space.shape[0]

    def action_dim(self):
        return self.gym_env.action_space.shape[0]

    def start_recording(self, video_path):
        self.video_recorder = VideoRecorder(
            env=self.gym_env,
            base_path=video_path[:-4],
            metadata={},
            enabled=True,
        )

    def grab_frame(self):
        self.video_recorder.capture_frame()

    def stop_recording(self):
        self.video_recorder.close()
        self.video_recorder = None

    def config(self):
        return self._config
