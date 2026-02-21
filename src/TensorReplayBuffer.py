import torch
import numpy as np

class TensorReplayBuffer:
    def __init__(self, memory_size, specs, device):
        """
        specs: { "フィールド名": {"shape": (形), "dtype": torch.dtype}, ... }
        """
        self.memory_size = memory_size
        self.specs = specs
        self.device = device
        self.ptr = 0
        self.size = 0

        self.buffers = {}
        for name, spec in specs.items():
            full_shape = (memory_size, *spec["shape"])
            self.buffers[name] = torch.zeros(
                full_shape, dtype=spec["dtype"], device=device
            )

    def add(self, **data):
        """キーワード引数でデータを受け取る (例: buffer.add(states=s, actions=a, ...))"""
        for name, value in data.items():
            if name in self.buffers:
                if not isinstance(value, torch.Tensor):
                    val = torch.from_numpy(np.array(value)).to(self.device)
                else:
                    val = value
                self.buffers[name][self.ptr] = val

        self.ptr = (self.ptr + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)

    def sample(self, batch_size):
        """辞書形式でミニバッチを返す"""
        if self.size == 0:
            raise RuntimeError("buffer is empty")
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return {name: self.buffers[name][indices] for name in self.buffers}

    def __len__(self):
        return self.size



class FrameStackReplayBuffer:
    """
    FrameStackするときに効率が良いbuffer
    
    note: statesとnext_statesで重複したものを保存している。改善の余地あり（1/2）。特に、状態がメモリ上支配的な可能性が高い。
    
    """
    def __init__(self, memory_size, frame_shape, window_size, device, frame_dtype=torch.uint8, action_dtype=torch.int64, reward_dtype=torch.float32):
        self.memory_size = memory_size
        self.window_size = window_size
        self.frame_shape = frame_shape
        self.device = device
        

        self.states = torch.zeros((memory_size, *frame_shape), dtype=frame_dtype, device=device)
        self.next_states = torch.zeros((memory_size, *frame_shape), dtype=frame_dtype, device=device)
        self.actions = torch.zeros((memory_size, ), dtype=action_dtype, device=device)
        self.rewards = torch.zeros((memory_size, ), dtype=reward_dtype, device=device)
        self.terminateds = torch.zeros((memory_size, ), dtype=torch.bool, device=device)
        self.episode_id = torch.full((memory_size,), -1, dtype=torch.int64, device=device)
        self.cur_episode = 0
            
        self.ptr = 0
        self.size = 0
        
    def add(self, states, action, reward, terminated, truncated, next_states):
        
        if not isinstance(states, torch.Tensor):
            states = torch.as_tensor(states, dtype=self.states.dtype, device=self.device)
        if not isinstance(next_states, torch.Tensor):
            next_states = torch.as_tensor(next_states, dtype=self.next_states.dtype, device=self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=self.actions.dtype, device=self.device)
        if not isinstance(reward, torch.Tensor):
            reward = torch.as_tensor(reward, dtype=self.rewards.dtype, device=self.device)
        if not isinstance(terminated, torch.Tensor):
            terminated = torch.as_tensor(terminated, dtype=self.terminateds.dtype, device=self.device) 
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.as_tensor(truncated, dtype=torch.bool, device=self.device)
            
        if states.shape == (self.window_size, *self.frame_shape):
            states = states[-1]
        if next_states.shape == (self.window_size, *self.frame_shape):
            next_states = next_states[-1]

            
        self.states[self.ptr] = states
        self.next_states[self.ptr] = next_states
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.terminateds[self.ptr] = terminated
        

        self.episode_id[self.ptr] = self.cur_episode    
            
        self.ptr = (self.ptr + 1) % self.memory_size
        self.size = min(self.size + 1, self.memory_size)
        
        if terminated or truncated:
            self.cur_episode += 1
            
            
    def sample(self, batch_size):
        if self.size == 0:
            raise RuntimeError("buffer is empty")
        
        if self.size < self.memory_size:
            r = torch.randint(0, self.size, (batch_size, ), device=self.device)
        else:
            ## (r-window_size, r] ヲのデータを抜き出すとして、rとして不適格なのは、[ptr-1, ptr]をまたぐケース。この場合、当該エピソードで4時間取れる保証がない（ただし、スタート状態は繰り返すので問題なし）
            ## ptr + window_size - 1 <= r で、長さ memory_size - (window_size - 1) だけ取れる。 mod meory_sizeで考えると簡単。
            r = torch.randint(self.ptr + self.window_size - 1, self.ptr + self.memory_size, (batch_size, ), device=self.device)
            r = r % self.memory_size
        
        idx = torch.empty((batch_size, self.window_size), dtype=torch.int64, device=self.device)  
        idx[:, -1] = r
        base_id = self.episode_id[r]
        

        for j in range(self.window_size-2, -1, -1):
            prev = idx[:, j+1]
            cand = (prev - 1) % self.memory_size
            ## episodeをまたぐ場合、開始状態を繰り返す
            valid = base_id == self.episode_id[cand]
            idx[:, j] = torch.where(valid, cand, prev)
            
        states = self.states[idx]
        next_states = torch.empty_like(states)
        next_states[:, :-1] = states[:, 1:]
        next_states[:, -1] = self.next_states[r]
        
        return {
            "states": states,
            "actions": self.actions[r].unsqueeze(1),
            "rewards": self.rewards[r].unsqueeze(1),
            "terminateds": self.terminateds[r].to(torch.float32).unsqueeze(1),
            "next_states": next_states,
        }
        
    def __len__(self):
        return self.size