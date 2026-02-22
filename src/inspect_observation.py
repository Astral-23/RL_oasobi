import argparse
import os

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from wrapper import ClipRewardWrapper, MaxPoolWrapper


def build_env(window_size=4, image_size=120):
    gym.register_envs(ale_py)
    env = gym.make("ALE/AirRaid-v5", render_mode=None)
    env = ClipRewardWrapper(env)
    env = MaxPoolWrapper(env)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, (image_size, image_size))
    env = gym.wrappers.FrameStackObservation(env, window_size)
    return env


def check_and_get_latest(obs, window_size, image_size):
    arr = np.asarray(obs)
    assert arr.ndim == 3, f"expected (C,H,W), got {arr.shape}"
    assert arr.shape == (window_size, image_size, image_size), (
        f"expected {(window_size, image_size, image_size)}, got {arr.shape}"
    )
    assert np.issubdtype(arr.dtype, np.integer), f"unexpected dtype: {arr.dtype}"
    lo = int(arr.min())
    hi = int(arr.max())
    assert 0 <= lo <= hi <= 255, f"pixel range out of bounds: [{lo}, {hi}]"
    return arr[-1]


def save_gray(img, path):
    plt.imsave(path, img, cmap="gray", vmin=0, vmax=255)


def main():
    parser = argparse.ArgumentParser(
        description="Randomly act for N env steps and save the latest grayscale frame seen by the agent."
    )
    parser.add_argument("out_dir", nargs="?", default="../records/obs_debug")
    parser.add_argument("--steps", type=int, required=True, help="Number of env.step() calls to record.")
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=120)
    args = parser.parse_args()

    assert args.steps >= 1, "--steps must be >= 1"
    assert args.window_size >= 1, "--window-size must be >= 1"
    assert args.image_size >= 1, "--image-size must be >= 1"

    os.makedirs(args.out_dir, exist_ok=True)
    frames_dir = os.path.join(args.out_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    env = build_env(window_size=args.window_size, image_size=args.image_size)

    try:
        if hasattr(env.unwrapped, "get_action_meanings"):
            print("action meanings:", env.unwrapped.get_action_meanings())
        print("frameskip:", getattr(env.unwrapped, "_frameskip", getattr(env.unwrapped, "frameskip", None)))
    except Exception:
        pass

    obs, info = env.reset()
    latest = check_and_get_latest(obs, args.window_size, args.image_size)
    save_gray(latest, os.path.join(frames_dir, "step_00000.png"))

    for step in range(1, args.steps + 1):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        latest = check_and_get_latest(obs, args.window_size, args.image_size)
        save_gray(latest, os.path.join(frames_dir, f"step_{step:05d}.png"))

        if terminated or truncated:
            obs, info = env.reset()
            # Reset直後の観測も shape/dtype/range の確認だけ行う
            check_and_get_latest(obs, args.window_size, args.image_size)

    env.close()
    print(f"saved {args.steps + 1} frames to: {frames_dir}")
    print("sanity checks passed: shape, dtype, pixel range")


if __name__ == "__main__":
    main()
