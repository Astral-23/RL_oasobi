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
    
class FrameSkipMaxPoolWrapper(gym.Wrapper):
    """
    FrameSkip + MaxPool。返すのは最後のフレームではなく、最後の2フレのMaxPool
    envのframeskipと併用しないこと。
    """
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip
    
    def reset(self, **kwargs):
        return self.env.reset(**kwargs)
    
    def step(self, action):
        obs_buf = []
        total_reward = 0
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            obs_buf.append(np.array(obs, copy=True))
            obs_buf = obs_buf[-2:]
            total_reward += reward
            if terminated or truncated:
                break
        if len(obs_buf) == 1:
            next_obs = obs_buf[0]
        else:
            next_obs = np.maximum(obs_buf[0], obs_buf[1])
        return next_obs, total_reward, terminated, truncated, info
    
        
        
        
        