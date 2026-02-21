import gymnasium as gym
import numpy as np


class ClipRewardWrapper(gym.RewardWrapper):
    def __init__(self, env):
        super().__init__(env)
    def reward(self, reward):
        return float(np.sign(reward))