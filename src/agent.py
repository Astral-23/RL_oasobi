import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import numpy as np
from TensorReplayBuffer import TensorReplayBuffer, FrameStackReplayBuffer
from model import AirRaidModel


class AirRaidAgent:
    def __init__(
        self,
        env,
        model,
        learning_rate,
        initial_epsilon,
        final_epsilon,
        decay_steps, 
        discount_factor,
        batch_size=32,
        window_size=4,
        memory_size=50000,
        device = "auto"
    ):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        self.env = env
        self.model = model
        self.lr = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = initial_epsilon
        self.initial_epsilon = initial_epsilon
        self.final_epsilon = final_epsilon
        self.decay_steps = decay_steps
        self.is_training = True
        self.batch_size = batch_size
        #self.specs = {
        #    "states": {"shape": (window_size, 250, 160), "dtype": torch.uint8},
        #    "action": {"shape": (1,), "dtype": torch.long},
        #    "reward": {"shape": (1,), "dtype": torch.float32},
        #    "terminated": {"shape": (1,), "dtype": torch.float32},
        #    "truncated": {"shape": (1,), "dtype": torch.float32},
        #    "next_states": {"shape": (window_size, 250, 160), "dtype": torch.uint8},
        #}
        self.memory_size = memory_size
        #self.buffer = TensorReplayBuffer(memory_size, self.specs, "cpu")
        self.buffer = FrameStackReplayBuffer(memory_size, (250, 160), window_size, "cpu")
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()

    def to(self, device):
        self.device = device
        self.model.to(self.device)

    def set_mode(self, training=True):
        self.is_training = training
        if training:
            self.model.train()
        else:
            self.model.eval()

    def get_action(self, obs):
        """
        is_training=True: イプシロン-greedy
        is_training=False: greedy
        """
        with torch.no_grad():
            if self.is_training and np.random.random() < self.epsilon:
                return self.env.action_space.sample()
            else:
                state = (
                    torch.tensor(obs, dtype=torch.float32).unsqueeze(0).to(self.device)
                )
                return torch.argmax(self.model(state / 255)).item()

    def update(self):
        if len(self.buffer) < self.batch_size:
            return

        batch = self.buffer.sample(self.batch_size)
        states = batch["states"].to(self.device).float() / 255
        next_states = batch["next_states"].to(self.device).float() / 255 
        actions = batch["actions"].to(self.device)
        rewards = batch["rewards"].to(self.device).squeeze(-1)
        terminateds = batch["terminateds"].to(self.device).squeeze(-1)

        current_q_values = self.model(states).gather(1, actions).squeeze(-1)

        with torch.no_grad():
            max_next_q_values = self.model(next_states).max(1)[0]
            targets = rewards + (
                self.discount_factor * max_next_q_values * (1 - terminateds)
            )

        loss = self.criterion(current_q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    
    def decay_epsilon(self):
        self.epsilon = max(self.final_epsilon, self.epsilon  + (self.final_epsilon - self.initial_epsilon)/ self.decay_steps)
        
    def add_experience(self, obs, action, reward, terminated, truncated, next_obs):
        self.buffer.add(
            states=obs,
            action=action,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            next_states=next_obs,
        )
        
    def save(self, file_path):
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon
        }
        torch.save(checkpoint, file_path)

    def load(self, file_path):
        checkpoint = torch.load(file_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']

    def evaluate(self, num_episodes=5):
        pre_mode = self.is_training
        self.set_mode(False)
        
        avg_reward = 0
        avg_length = 0        
        
        for _ in range(num_episodes):
            obs, info = self.env.reset()
            episode_over = False
            
            while not episode_over:
                action = self.get_action(obs)
                next_obs, reward, terminated, truncated, next_info = self.env.step(action)
                episode_over = terminated or truncated
                obs = next_obs
                avg_reward += reward
                avg_length += 1
                
        avg_reward /= num_episodes
        avg_length /= num_episodes
        self.set_mode(pre_mode)
        return avg_reward, avg_length