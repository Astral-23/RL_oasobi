import argparse
import time

import ale_py
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

from wrapper import ClipRewardWrapper, MaxPoolWrapper, FrameSkipMaxPoolWrapper


def build_env(image_size=120, window_size=4, use_clip=True, use_maxpool=True):
    gym.register_envs(ale_py)
    env = gym.make("ALE/AirRaid-v5", render_mode=None, frameskip=1)
    if use_clip:
        env = ClipRewardWrapper(env)
    env = FrameSkipMaxPoolWrapper(env, skip=4)
    env = gym.wrappers.GrayscaleObservation(env, keep_dim=False)
    env = gym.wrappers.ResizeObservation(env, (image_size, image_size))
    env = gym.wrappers.FrameStackObservation(env, window_size)
    return env


def obs_to_array(obs):
    arr = np.asarray(obs)
    if arr.ndim == 2:
        arr = arr[None, ...]
    return arr


def tile_stack(arr):
    # arr: (C,H,W), display up to 4 frames in 2x2 grid
    c, h, w = arr.shape
    if c == 1:
        return arr[0]
    frames = [arr[min(i, c - 1)] for i in range(4)]
    top = np.concatenate([frames[0], frames[1]], axis=1)
    bottom = np.concatenate([frames[2], frames[3]], axis=1)
    return np.concatenate([top, bottom], axis=0)


def frame_for_display(obs, mode="latest"):
    arr = obs_to_array(obs)
    if mode == "tile":
        return tile_stack(arr)
    return arr[-1]


def main():
    parser = argparse.ArgumentParser(description="Play AirRaid while viewing wrapped (post-FrameStack) observations.")
    parser.add_argument("--image-size", type=int, default=120)
    parser.add_argument("--window-size", type=int, default=4)
    parser.add_argument("--step-hz", type=float, default=15.0, help="Approximate env.step() rate.")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--no-maxpool", action="store_true")
    parser.add_argument("--view", choices=["latest", "tile"], default="latest")
    args = parser.parse_args()

    env = build_env(
        image_size=args.image_size,
        window_size=args.window_size,
        use_clip=not args.no_clip,
        use_maxpool=not args.no_maxpool,
    )

    action_meanings = None
    if hasattr(env.unwrapped, "get_action_meanings"):
        try:
            action_meanings = env.unwrapped.get_action_meanings()
        except Exception:
            action_meanings = None
    print("frameskip:", getattr(env.unwrapped, "_frameskip", getattr(env.unwrapped, "frameskip", None)))
    if action_meanings is not None:
        print("action meanings:")
        for i, name in enumerate(action_meanings):
            print(f"  {i}: {name}")

    obs, info = env.reset()
    state = {
        "action": 0,
        "held_actions": [],
        "quit": False,
        "paused": False,
        "step_once": False,
        "view": args.view,
        "episode_reward": 0.0,
        "episode_len": 0,
        "episode_idx": 0,
    }

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_axis_off()
    img = ax.imshow(frame_for_display(obs, state["view"]), cmap="gray", vmin=0, vmax=255, interpolation="nearest")

    def update_title():
        ax.set_title(
            f"episode={state['episode_idx']} len={state['episode_len']} reward={state['episode_reward']:.1f} "
            f"action={state['action']} view={state['view']} {'PAUSED' if state['paused'] else ''}"
        )

    def recompute_action():
        state["action"] = state["held_actions"][-1] if state["held_actions"] else 0

    def push_action(action_id):
        if action_id in state["held_actions"]:
            state["held_actions"] = [a for a in state["held_actions"] if a != action_id]
        state["held_actions"].append(action_id)
        recompute_action()

    def release_action(action_id):
        if action_id in state["held_actions"]:
            state["held_actions"] = [a for a in state["held_actions"] if a != action_id]
            recompute_action()

    def on_key(event):
        key = (event.key or "").lower()
        if key is None:
            return
        if key in {str(i) for i in range(10)}:
            push_action(int(key))
            print(f"action (hold) -> {state['action']}")
            return
        if key == "q":
            state["quit"] = True
            plt.close(fig)
            return
        if key == "p":
            state["paused"] = not state["paused"]
            print("paused" if state["paused"] else "resumed")
            return
        if key == "n":
            state["step_once"] = True
            return
        if key == "r":
            state["episode_idx"] += 1
            state["episode_reward"] = 0.0
            state["episode_len"] = 0
            obs_reset, _ = env.reset()
            img.set_data(frame_for_display(obs_reset, state["view"]))
            update_title()
            fig.canvas.draw_idle()
            return
        if key == "m":
            state["view"] = "tile" if state["view"] == "latest" else "latest"
            print(f"view -> {state['view']}")
            return
        if key == " ":
            push_action(0)
            print("action (hold) -> 0")

    def on_key_release(event):
        key = (event.key or "").lower()
        if key in {str(i) for i in range(10)}:
            release_action(int(key))
            return
        if key == " ":
            release_action(0)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("key_release_event", on_key_release)
    print("keys: hold 0-9 action, hold space=noop, p pause, n step once, m toggle latest/tile, r reset, q quit")

    step_dt = 1.0 / max(args.step_hz, 1e-3)
    next_tick = time.perf_counter()

    try:
        while plt.fignum_exists(fig.number) and not state["quit"]:
            now = time.perf_counter()
            if state["paused"] and not state["step_once"]:
                plt.pause(0.01)
                continue
            if now < next_tick and not state["step_once"]:
                plt.pause(min(0.01, next_tick - now))
                continue

            obs, reward, terminated, truncated, info = env.step(state["action"])
            state["episode_reward"] += float(reward)
            state["episode_len"] += 1
            img.set_data(frame_for_display(obs, state["view"]))
            update_title()
            fig.canvas.draw_idle()
            plt.pause(0.001)

            if terminated or truncated:
                print(
                    f"episode done: idx={state['episode_idx']} len={state['episode_len']} reward={state['episode_reward']:.1f}"
                )
                state["episode_idx"] += 1
                state["episode_reward"] = 0.0
                state["episode_len"] = 0
                obs, info = env.reset()
                img.set_data(frame_for_display(obs, state["view"]))
                update_title()
                fig.canvas.draw_idle()

            state["step_once"] = False
            next_tick = time.perf_counter() + step_dt
    finally:
        env.close()
        plt.close("all")


if __name__ == "__main__":
    main()
