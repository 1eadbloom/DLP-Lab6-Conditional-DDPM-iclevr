"""
test_model.py  –  DLP Lab5 Evaluation Script (Tasks 1, 2, 3)

Command-line usage
------------------
# Task 1 – CartPole
python test_model.py --task 1 --model-path LAB5_StudentID_task1.pt

# Task 2 – Pong (vanilla DQN)
python test_model.py --task 2 --model-path LAB5_StudentID_task2.pt

# Task 3 – Pong (enhanced DQN), 20 fixed seeds 0-19
python test_model.py --task 3 \
    --model-path LAB5_StudentID_task3_2500000.pt \
    --env-steps  2500000

Colab usage (from a notebook cell)
-----------------------------------
    from test_model import evaluate, EvalArgs
    args = EvalArgs(task=3,
                    model_path="/content/drive/MyDrive/Lab5/task3/LAB5_ID_task3_2000000.pt",
                    env_steps=2_000_000)
    evaluate(args)
"""

import torch
import torch.nn as nn
import numpy as np
import random
import gymnasium as gym
import cv2
import imageio
import ale_py
import os
from collections import deque
import argparse

gym.register_envs(ale_py)


# ---------------------------------------------------------------------------
# Network  (must mirror dqn.py exactly)
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    def __init__(self, num_actions, input_dim=4, use_cnn=False):
        super(DQN, self).__init__()
        self.use_cnn = use_cnn
        if use_cnn:
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions),
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions),
            )

    def forward(self, x):
        if self.use_cnn:
            return self.network(x / 255.0)
        return self.network(x)


# ---------------------------------------------------------------------------
# Atari pre-processor
# ---------------------------------------------------------------------------

class AtariPreprocessor:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY) \
               if (len(obs.shape) == 3 and obs.shape[2] == 3) else obs
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        self.frames.append(self.preprocess(obs).copy())
        return np.stack(self.frames, axis=0)


# ---------------------------------------------------------------------------
# EvalArgs  –  dataclass-style object so the script works both from the
#              command line AND from a Colab cell (no argparse needed).
# ---------------------------------------------------------------------------

class EvalArgs:
    def __init__(self,
                 task:       int  = 1,
                 model_path: str  = "",
                 output_dir: str  = "./eval_videos",
                 episodes:   int  = 20,
                 seed:       int  = 313551076,
                 env_steps:  int  = 0,
                 save_video: bool = False):
        self.task       = task
        self.model_path = model_path
        self.output_dir = output_dir
        self.episodes   = episodes
        self.seed       = seed
        self.env_steps  = env_steps
        self.save_video = save_video


# ---------------------------------------------------------------------------
# Single-episode runner
# ---------------------------------------------------------------------------

def run_episode(env, model, preprocessor, device, seed,
                save_video=False, video_path=None):
    obs, _ = env.reset(seed=seed)
    state  = preprocessor.reset(obs) if preprocessor else np.array(obs, dtype=np.float32)

    done  = False
    total = 0.0
    frames = []

    while not done:
        if save_video:
            frames.append(env.render())

        t = torch.from_numpy(np.array(state)).float().unsqueeze(0).to(device)
        with torch.no_grad():
            action = model(t).argmax().item()

        next_obs, reward, terminated, truncated, _ = env.step(action)
        done  = terminated or truncated
        total += reward
        state = preprocessor.step(next_obs) if preprocessor \
                else np.array(next_obs, dtype=np.float32)

    if save_video and video_path and frames:
        os.makedirs(os.path.dirname(os.path.abspath(video_path)), exist_ok=True)
        with imageio.get_writer(video_path, fps=30) as writer:
            for f in frames:
                writer.append_data(f)
        print(f"  Video saved → {video_path}")

    return total


# ---------------------------------------------------------------------------
# Main evaluate function
# ---------------------------------------------------------------------------

def evaluate(args: EvalArgs):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Device] {device}")

    # ---- Environment & model type ------------------------------------------
    if args.task == 1:
        env_id    = "CartPole-v1"
        use_cnn   = False
        use_atari = False
    else:
        env_id    = "ALE/Pong-v5"
        use_cnn   = True
        use_atari = True

    env          = gym.make(env_id, render_mode="rgb_array")
    num_actions  = env.action_space.n
    preprocessor = AtariPreprocessor() if use_atari else None

    if use_cnn:
        model = DQN(num_actions, use_cnn=True).to(device)
    else:
        obs_dim = env.observation_space.shape[0]
        model   = DQN(num_actions, input_dim=obs_dim, use_cnn=False).to(device)

    model.load_state_dict(
        torch.load(args.model_path, map_location=device, weights_only=True))
    model.eval()
    print(f"[Model] Loaded from {args.model_path}")

    # ---- Seed list ---------------------------------------------------------
    # Task 3 uses the official grading protocol: seeds 0 … 19
    if args.task == 3:
        seeds            = list(range(20))
        env_steps_label  = args.env_steps
    else:
        seeds            = [args.seed + ep for ep in range(args.episodes)]
        env_steps_label  = None

    os.makedirs(args.output_dir, exist_ok=True)

    # ---- Run episodes ------------------------------------------------------
    rewards = []
    for ep, seed in enumerate(seeds):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        env.action_space.seed(seed)

        save_vid = args.save_video and ep == 0
        vid_path = os.path.join(args.output_dir, f"task{args.task}_ep{ep}.mp4") \
                   if save_vid else None

        r = run_episode(env, model, preprocessor, device, seed,
                        save_video=save_vid, video_path=vid_path)
        rewards.append(r)

        if env_steps_label is not None:
            print(f"Environment steps: {env_steps_label}, "
                  f"seed: {seed}, eval reward: {r:.0f}")
        else:
            print(f"Episode {ep:>2}: seed={seed}, total reward={r:.2f}")

    avg = float(np.mean(rewards))
    print(f"\nAverage reward: {avg:.2f}  ({len(rewards)} episodes)")

    # ---- Grading estimate --------------------------------------------------
    if args.task == 1:
        score = min(avg, 480) / 480 * 15
        print(f"[Task 1] Estimated grading score ≈ {score:.2f} / 15")
    elif args.task == 2:
        score = (min(avg, 19) + 21) / 40 * 20
        print(f"[Task 2] Estimated grading score ≈ {score:.2f} / 20")
    else:
        print("[Task 3] Score depends on sample efficiency (see grading table).")

    env.close()
    return avg


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLP Lab5 – Model Evaluation")
    parser.add_argument("--task",       type=int,  required=True, choices=[1, 2, 3])
    parser.add_argument("--model-path", type=str,  required=True)
    parser.add_argument("--output-dir", type=str,  default="./eval_videos")
    parser.add_argument("--episodes",   type=int,  default=20,
                        help="Number of eval episodes (tasks 1 & 2 only)")
    parser.add_argument("--seed",       type=int,  default=313551076,
                        help="Base random seed (tasks 1 & 2 only)")
    parser.add_argument("--env-steps",  type=int,  default=0,
                        help="Env steps at snapshot time (task 3, display only)")
    parser.add_argument("--save-video", action="store_true")
    raw = parser.parse_args()

    args = EvalArgs(
        task       = raw.task,
        model_path = raw.model_path,
        output_dir = raw.output_dir,
        episodes   = raw.episodes,
        seed       = raw.seed,
        env_steps  = raw.env_steps,
        save_video = raw.save_video,
    )
    evaluate(args)
