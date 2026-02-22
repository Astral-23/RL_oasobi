import gymnasium as gym
import numpy as np


class ClipRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    def reward(self, reward):
        return float(np.sign(reward))
    
class MaxPoolWrapper(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev = None
        
    def reset(self, **kwargs):
        self.prev = None
        return super().reset(**kwargs)

    def observation(self, obs):
        if self.prev is not None:
            out = np.maximum(self.prev, obs)
        else:
            out = obs
        self.prev = np.array(obs,copy=True)
        return out