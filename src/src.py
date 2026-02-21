import torch
import torch.nn as nn
import torch.functional as F
import torch.optim as optim
import gymnasium as gym
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime
import ale_py
from TensorReplayBuffer import TensorReplayBuffer
from model import AirRaidModel
from agent import AirRaidAgent

torch.set_num_threads(10)

# コマンドライン引数の取得と保存先ディレクトリの設定
if len(sys.argv) < 2:
    print("実行時に保存先のフォルダ名を指定してください。 例: python script.py test_run_01")
    sys.exit(1)

save_dir = os.path.join("../records", sys.argv[1])
os.makedirs(save_dir, exist_ok=True)

# これを明示的に呼ぶことでALE環境がgymに登録されます
gym.register_envs(ale_py)

H = 250
W = 160
episode_num = 10000
time_stamps = 100 #ログの頻度にも影響するよ
video_time_stamps = time_stamps // 10

window_size = 4

env = gym.make("ALE/AirRaid-v5", render_mode=None)
#env = gym.wrappers.RecordVideo(
#    env,
#    video_folder=save_dir,
#    name_prefix="training",
#    episode_trigger=lambda x: x == 0 or (x + 1) % (episode_num // video_time_stamps) == 0,
#)

env = gym.wrappers.RecordEpisodeStatistics(env)
env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
env = gym.wrappers.FrameStackObservation(env, window_size)
model = AirRaidModel(window_size)
model.compile()
model.to(device="cuda")
agent = AirRaidAgent(env, model, learning_rate=0.001, initial_epsilon=1.0, final_epsilon=0.01, decay_steps = episode_num // 5, discount_factor=0.99, window_size=window_size, memory_size=200000)
rewards_history = []
lengths_history = []
eval_rewards_history = []
eval_episodes_history = []
eval_lengths_history = []

agent.set_mode(True)
for episode in tqdm(range(episode_num)):
    obs, info = env.reset()
    episode_over = False
    t = 0
    while not episode_over:
        action = agent.get_action(obs)

        next_obs, reward, terminated, truncated, next_info = env.step(action)
        episode_over = terminated or truncated

      
        if episode_over and "episode" in next_info:
            ep_info = next_info["episode"]


            ep_reward = float(ep_info["r"])
            ep_length = int(ep_info["l"])
            
            rewards_history.append(ep_reward)
            lengths_history.append(ep_length)

            if episode == 0 or (episode + 1) % (episode_num // time_stamps) == 0:
                eval_reward, eval_length = agent.evaluate()
                eval_rewards_history.append(eval_reward)
                eval_lengths_history.append(eval_length)
                eval_episodes_history.append(episode + 1)
                print(f"\n[Evaluation] Episode {episode+1}: Avg Reward: {eval_reward:.2f} Avg length: {eval_length:.2f}")
                

        agent.add_experience(obs, action, reward, terminated, truncated, next_obs)

        t = t + 1
        agent.update()
        obs = next_obs
        info = next_info
        
    agent.decay_epsilon()
        
        
        
        
# --- データの整合性チェック ---
has_eval = len(eval_rewards_history) > 0

# 1. time.txtに時刻を記録
current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
time_file_path = os.path.join(save_dir, "time.txt")
with open(time_file_path, "w") as f:
    f.write(current_time + "\n")

# 2. グラフの描画
file_path = os.path.join(save_dir, "training_performance.png")
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 12)) # 縦を少し長めに調整

# --- 上段: 報酬 (Reward) ---
ax1.plot(rewards_history, label="Training Reward (Noisy)", color="blue", alpha=0.3)
if has_eval:
    ax1.plot(eval_episodes_history, eval_rewards_history, 
             label="Evaluation Reward (Greedy)", color="red", marker='o', linewidth=2)
ax1.set_title("Reward Performance")
ax1.set_xlabel("Episode")
ax1.set_ylabel("Total Reward")
ax1.legend(loc="upper left")
ax1.grid(True, linestyle='--', alpha=0.5)

# --- 下段: エピソードの長さ (Length / Steps) ---
# 学習時のステップ（薄いオレンジ）
ax2.plot(lengths_history, label="Training Length (Noisy)", color="orange", alpha=0.3)
# 評価時のステップ（濃い緑や茶色で区別）
if has_eval:
    # eval_lengths_history は evaluate_agent が返した平均長さのリスト
    ax2.plot(eval_episodes_history, eval_lengths_history, 
             label="Evaluation Length (Greedy)", color="darkgreen", marker='s', linewidth=2)
ax2.set_title("Episode Duration (Survival Steps)")
ax2.set_xlabel("Episode")
ax2.set_ylabel("Steps")
ax2.legend(loc="upper left")
ax2.grid(True, linestyle='--', alpha=0.5)

plt.tight_layout()
plt.savefig(file_path)
plt.close()

print(f"--- 統計を保存しました ---")
print(f"最終評価スコア: {eval_rewards_history[-1] if has_eval else 'N/A'}")
print(f"最終生存ステップ: {eval_lengths_history[-1] if has_eval else 'N/A'}")

# モデルの保存
agent.save(os.path.join(save_dir, "model.pth"))