# Spring 2026, 535518 Deep Learning
# Lab5: Value-based RL
# Contributors: Kai-Siang Ma and Alison Wen
# Instructor: Ping-Chun Hsieh

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import gymnasium as gym
import cv2
import ale_py
import os
from collections import deque
import wandb
import argparse
import time

gym.register_envs(ale_py)

# ---------------------------------------------------------------------------
# Colab / Google Drive helper
# ---------------------------------------------------------------------------

def setup_drive(use_drive: bool, drive_dir: str) -> str:
    """
    Mount Google Drive (if in Colab) and return the save directory.
    Falls back to a local path when Drive is unavailable.
    """
    if not use_drive:
        return drive_dir  # treat as local path

    try:
        from google.colab import drive
        drive.mount("/content/drive", force_remount=False)
        full_path = os.path.join("/content/drive/MyDrive", drive_dir)
        os.makedirs(full_path, exist_ok=True)
        print(f"[Drive] Saving checkpoints to: {full_path}")
        return full_path
    except Exception as e:
        print(f"[Drive] Could not mount Drive ({e}). Using local path: {drive_dir}")
        os.makedirs(drive_dir, exist_ok=True)
        return drive_dir


# ---------------------------------------------------------------------------
# Weight init
# ---------------------------------------------------------------------------

def init_weights(m):
    if isinstance(m, (nn.Conv2d, nn.Linear)):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)


# ---------------------------------------------------------------------------
# DQN network
# ---------------------------------------------------------------------------

class DQN(nn.Module):
    """
    Deep Q-Network supporting:
      - CartPole  : flat vector input  (use_cnn=False)
      - Atari Pong: stacked frames     (use_cnn=True)
    """
    def __init__(self, num_actions, input_dim=4, use_cnn=False):
        super(DQN, self).__init__()
        self.use_cnn = use_cnn

        ########## YOUR CODE HERE (5~10 lines) ##########
        if use_cnn:
            # Standard Nature-DQN CNN  –  input: (B, 4, 84, 84)
            self.network = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4),   # → (B,32,20,20)
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),  # → (B,64, 9, 9)
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),  # → (B,64, 7, 7)
                nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 7 * 7, 512),
                nn.ReLU(),
                nn.Linear(512, num_actions),
            )
        else:
            # MLP for CartPole  –  input: (B, input_dim)
            self.network = nn.Sequential(
                nn.Linear(input_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 128),
                nn.ReLU(),
                nn.Linear(128, num_actions),
            )
        ########## END OF YOUR CODE ##########

    def forward(self, x):
        if self.use_cnn:
            return self.network(x / 255.0)   # normalise pixels to [0,1]
        return self.network(x)


# ---------------------------------------------------------------------------
# Atari pre-processor
# ---------------------------------------------------------------------------

class AtariPreprocessor:
    """Grayscale + resize + frame-stack for Atari observations."""

    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)

    def preprocess(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return cv2.resize(gray, (84, 84), interpolation=cv2.INTER_AREA)

    def reset(self, obs):
        frame = self.preprocess(obs)
        self.frames = deque([frame] * self.frame_stack, maxlen=self.frame_stack)
        return np.stack(self.frames, axis=0)

    def step(self, obs):
        self.frames.append(self.preprocess(obs))
        return np.stack(self.frames, axis=0)


# ---------------------------------------------------------------------------
# Prioritised Experience Replay  (Task 3)
# ---------------------------------------------------------------------------

class PrioritizedReplayBuffer:
    """
    Prioritized replay as in Schaul et al. (2016).
    https://arxiv.org/abs/1511.05952
    """

    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity   = capacity
        self.alpha      = alpha
        self.beta       = beta
        self.buffer     = []
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.pos        = 0

    # ---- add ---------------------------------------------------------------
    def add(self, transition, error):
        ########## YOUR CODE HERE (for Task 3) ##########
        priority = (abs(float(error)) + 1e-6) ** self.alpha
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.priorities[self.pos] = priority
        self.pos = (self.pos + 1) % self.capacity
        ########## END OF YOUR CODE (for Task 3) ##########

    # ---- sample ------------------------------------------------------------
    def sample(self, batch_size):
        ########## YOUR CODE HERE (for Task 3) ##########
        n     = len(self.buffer)
        probs = self.priorities[:n] / self.priorities[:n].sum()
        indices = np.random.choice(n, batch_size, replace=False, p=probs)

        # Importance-sampling weights  w_i = (1 / N·P(i))^β , normalised
        weights = (n * probs[indices]) ** (-self.beta)
        weights /= weights.max()

        batch = [self.buffer[i] for i in indices]
        states, actions, rewards, next_states, dones = zip(*batch)
        return (states, actions, rewards, next_states, dones,
                indices, weights.astype(np.float32))
        ########## END OF YOUR CODE (for Task 3) ##########

    # ---- update priorities -------------------------------------------------
    def update_priorities(self, indices, errors):
        ########## YOUR CODE HERE (for Task 3) ##########
        for idx, err in zip(indices, errors):
            self.priorities[idx] = (abs(float(err)) + 1e-6) ** self.alpha
        ########## END OF YOUR CODE (for Task 3) ##########

    def __len__(self):
        return len(self.buffer)


# ---------------------------------------------------------------------------
# DQN Agent
# ---------------------------------------------------------------------------

# Milestones (env steps) at which Task-3 snapshots must be saved
TASK3_MILESTONES = {600_000, 1_000_000, 1_500_000, 2_000_000, 2_500_000}


class DQNAgent:
    def __init__(self, env_name="CartPole-v1", args=None):
        self.env_name  = env_name
        self.use_atari = "ALE/" in env_name or "ale/" in env_name.lower()

        self.env      = gym.make(env_name, render_mode="rgb_array")
        self.test_env = gym.make(env_name, render_mode="rgb_array")
        self.num_actions = self.env.action_space.n

        self.preprocessor = AtariPreprocessor() if self.use_atari else None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Device] {self.device}")

        # ---- Networks --------------------------------------------------------
        if self.use_atari:
            self.q_net      = DQN(self.num_actions, use_cnn=True).to(self.device)
            self.target_net = DQN(self.num_actions, use_cnn=True).to(self.device)
        else:
            obs_dim = self.env.observation_space.shape[0]
            self.q_net      = DQN(self.num_actions, input_dim=obs_dim).to(self.device)
            self.target_net = DQN(self.num_actions, input_dim=obs_dim).to(self.device)

        self.q_net.apply(init_weights)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=args.lr)

        # ---- Hyper-parameters ------------------------------------------------
        self.batch_size              = args.batch_size
        self.gamma                   = args.discount_factor
        self.epsilon                 = args.epsilon_start
        self.epsilon_decay           = args.epsilon_decay
        self.epsilon_min             = args.epsilon_min
        self.env_count               = 0
        self.train_count             = 0
        self.best_reward             = -21 if self.use_atari else 0
        self.max_episode_steps       = args.max_episode_steps
        self.replay_start_size       = args.replay_start_size
        self.target_update_frequency = args.target_update_frequency
        self.train_per_step          = args.train_per_step
        self.save_dir                = args.save_dir
        self.student_id              = getattr(args, 'student_id', 'StudentID')
        os.makedirs(self.save_dir, exist_ok=True)

        # ---- Task-3 flags ----------------------------------------------------
        self.use_double  = getattr(args, 'use_double', False)
        self.use_per     = getattr(args, 'use_per',    False)
        self.multi_step  = getattr(args, 'multi_step', 1)
        self.n_step_buffer = deque(maxlen=self.multi_step)
        self._saved_milestones: set = set()   # track which steps are saved

        # ---- Replay memory ---------------------------------------------------
        if self.use_per:
            self.memory = PrioritizedReplayBuffer(
                args.memory_size,
                alpha=getattr(args, 'per_alpha', 0.6),
                beta =getattr(args, 'per_beta',  0.4),
            )
        else:
            self.memory = deque(maxlen=args.memory_size)

    # ======================================================================
    # Internal helpers
    # ======================================================================

    def _get_state(self, obs, reset=False):
        if self.use_atari:
            return self.preprocessor.reset(obs) if reset \
                   else self.preprocessor.step(obs)
        return np.array(obs, dtype=np.float32)

    def _to_tensor(self, state):
        return torch.from_numpy(np.array(state)).float().unsqueeze(0).to(self.device)

    def _compute_n_step_return(self, buf):
        R = 0.0
        for i, (_, _, r, _, d) in enumerate(buf):
            R += (self.gamma ** i) * r
            if d:
                break
        _, _, _, last_ns, last_done = buf[-1]
        return R, last_ns, last_done

    def _push(self, transition):
        if self.use_per:
            n    = len(self.memory)
            pmax = self.memory.priorities[:n].max() if n > 0 else 1.0
            self.memory.add(transition, pmax)
        else:
            self.memory.append(transition)

    def _maybe_save_milestone(self):
        """Save a task-3 snapshot when env_count crosses a milestone."""
        for ms in TASK3_MILESTONES:
            if self.env_count >= ms and ms not in self._saved_milestones:
                fname = f"LAB5_{self.student_id}_task3_{ms}.pt"
                path  = os.path.join(self.save_dir, fname)
                torch.save(self.q_net.state_dict(), path)
                self._saved_milestones.add(ms)
                print(f"[Milestone] Saved Task-3 snapshot → {path}")

    # ======================================================================
    # Action selection
    # ======================================================================

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.num_actions - 1)
        with torch.no_grad():
            return self.q_net(self._to_tensor(state)).argmax().item()

    # ======================================================================
    # Training loop
    # ======================================================================

    def run(self, episodes=1000):
        for ep in range(episodes):
            obs, _  = self.env.reset()
            state   = self._get_state(obs, reset=True)
            done    = False
            total_reward = 0
            step_count   = 0
            self.n_step_buffer.clear()

            while not done and step_count < self.max_episode_steps:
                action = self.select_action(state)
                next_obs, reward, terminated, truncated, _ = self.env.step(action)
                done        = terminated or truncated
                next_state  = self._get_state(next_obs)

                # Multi-step buffer
                self.n_step_buffer.append((state, action, reward, next_state, done))
                if len(self.n_step_buffer) == self.multi_step:
                    R, ns, d = self._compute_n_step_return(self.n_step_buffer)
                    s0, a0, *_ = self.n_step_buffer[0]
                    self._push((s0, a0, R, ns, d))

                for _ in range(self.train_per_step):
                    self.train()

                state        = next_state
                total_reward += reward
                self.env_count += 1
                step_count     += 1

                # ---- milestone snapshots (Task 3) ----------------------------
                self._maybe_save_milestone()

                if self.env_count % 1000 == 0:
                    print(f"[Collect] Ep:{ep} Step:{step_count} "
                          f"SC:{self.env_count} UC:{self.train_count} "
                          f"Eps:{self.epsilon:.4f}")
                    wandb.log({
                        "Episode": ep, "Step Count": step_count,
                        "Env Step Count": self.env_count,
                        "Update Count": self.train_count,
                        "Epsilon": self.epsilon,
                    })

            # Flush leftover n-step transitions
            while self.n_step_buffer:
                R, ns, d = self._compute_n_step_return(self.n_step_buffer)
                s0, a0, *_ = self.n_step_buffer[0]
                self._push((s0, a0, R, ns, d))
                self.n_step_buffer.popleft()

            print(f"[Ep {ep}] Reward:{total_reward} "
                  f"SC:{self.env_count} UC:{self.train_count} "
                  f"Eps:{self.epsilon:.4f}")
            wandb.log({
                "Episode": ep, "Total Reward": total_reward,
                "Env Step Count": self.env_count,
                "Update Count": self.train_count,
                "Epsilon": self.epsilon,
            })

            # Periodic checkpoint every 100 episodes
            if ep % 100 == 0:
                path = os.path.join(self.save_dir, f"model_ep{ep}.pt")
                torch.save(self.q_net.state_dict(), path)
                print(f"[Checkpoint] Saved → {path}")

            # Evaluation & best-model every 20 episodes
            if ep % 20 == 0:
                eval_reward = self.evaluate()
                if eval_reward > self.best_reward:
                    self.best_reward = eval_reward
                    best_path = os.path.join(self.save_dir, "best_model.pt")
                    torch.save(self.q_net.state_dict(), best_path)
                    print(f"[Best] New best ({eval_reward}) → {best_path}")
                print(f"[TrueEval] Ep:{ep} EvalReward:{eval_reward:.2f} "
                      f"SC:{self.env_count}")
                wandb.log({
                    "Env Step Count": self.env_count,
                    "Update Count": self.train_count,
                    "Eval Reward": eval_reward,
                })

    # ======================================================================
    # Evaluation
    # ======================================================================

    def evaluate(self):
        obs, _ = self.test_env.reset()
        state  = self._get_state(obs, reset=True)
        done   = False
        total  = 0
        while not done:
            with torch.no_grad():
                action = self.q_net(self._to_tensor(state)).argmax().item()
            next_obs, reward, terminated, truncated, _ = self.test_env.step(action)
            done  = terminated or truncated
            total += reward
            state  = self._get_state(next_obs)
        return total

    # ======================================================================
    # Training step
    # ======================================================================

    def train(self):
        if len(self.memory) < self.replay_start_size:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.train_count += 1

        # ---- Sample mini-batch -------------------------------------------
        ########## YOUR CODE HERE (<5 lines) ##########
        if self.use_per:
            (states, actions, rewards, next_states, dones,
             per_indices, is_weights) = self.memory.sample(self.batch_size)
            is_weights = torch.tensor(is_weights, dtype=torch.float32).to(self.device)
        else:
            batch = random.sample(self.memory, self.batch_size)
            states, actions, rewards, next_states, dones = zip(*batch)
            per_indices, is_weights = None, None
        ########## END OF YOUR CODE ##########

        # Convert to tensors
        states      = torch.from_numpy(np.array(states,      dtype=np.float32)).to(self.device)
        next_states = torch.from_numpy(np.array(next_states, dtype=np.float32)).to(self.device)
        actions     = torch.tensor(actions, dtype=torch.int64  ).to(self.device)
        rewards     = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        dones       = torch.tensor(dones,   dtype=torch.float32).to(self.device)

        q_values = self.q_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        ########## YOUR CODE HERE (~10 lines) ##########
        with torch.no_grad():
            if self.use_double:
                # Double DQN: online net selects action, target net evaluates
                best_a  = self.q_net(next_states).argmax(dim=1, keepdim=True)
                next_q  = self.target_net(next_states).gather(1, best_a).squeeze(1)
            else:
                next_q  = self.target_net(next_states).max(dim=1)[0]

            gamma_n = self.gamma ** self.multi_step
            targets = rewards + gamma_n * next_q * (1.0 - dones)

        td_errors = (targets - q_values).detach().cpu().numpy()

        if self.use_per and is_weights is not None:
            loss = (is_weights * nn.functional.smooth_l1_loss(
                        q_values, targets, reduction='none')).mean()
            self.memory.update_priorities(per_indices, np.abs(td_errors))
        else:
            loss = nn.functional.smooth_l1_loss(q_values, targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()
        ########## END OF YOUR CODE ##########

        if self.train_count % self.target_update_frequency == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        if self.train_count % 1000 == 0:
            print(f"[Train #{self.train_count}] Loss:{loss.item():.4f} "
                  f"Q_mean:{q_values.mean().item():.3f} "
                  f"Q_std:{q_values.std().item():.3f}")


# ===========================================================================
# Entry point  (also callable from Colab notebook)
# ===========================================================================

def build_args(overrides: dict = None):
    """
    Build a Namespace object with default hyper-parameters.
    Pass a dict of {key: value} to override any default.
    Useful when calling from a Colab cell instead of the command line.

    Example (Colab):
        from dqn import build_args, DQNAgent
        import wandb
        args = build_args({
            'env': 'ALE/Pong-v5',
            'use_double': True,
            'use_per': True,
            'multi_step': 3,
            'save_dir': '/content/drive/MyDrive/Lab5/task3',
            'student_id': '313551076',
        })
        wandb.init(project='DLP-Lab5-DQN', name='pong-enhanced')
        agent = DQNAgent(env_name=args.env, args=args)
        agent.run(episodes=2000)
    """
    import argparse
    defaults = dict(
        env                    = "CartPole-v1",
        save_dir               = "./results",
        student_id             = "StudentID",
        wandb_run_name         = "dqn-run",
        batch_size             = 32,
        memory_size            = 100_000,
        lr                     = 1e-4,
        discount_factor        = 0.99,
        epsilon_start          = 1.0,
        epsilon_decay          = 0.999_999,
        epsilon_min            = 0.05,
        target_update_frequency= 1_000,
        replay_start_size      = 50_000,
        max_episode_steps      = 10_000,
        train_per_step         = 1,
        use_double             = False,
        use_per                = False,
        per_alpha              = 0.6,
        per_beta               = 0.4,
        multi_step             = 1,
    )
    if overrides:
        defaults.update(overrides)
    return argparse.Namespace(**defaults)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DLP Lab5 – DQN Training")
    # paths & identity
    parser.add_argument("--save-dir",        type=str,   default="./results")
    parser.add_argument("--student-id",      type=str,   default="StudentID",
                        help="Your student ID, used in saved checkpoint filenames")
    parser.add_argument("--wandb-run-name",  type=str,   default="dqn-run")
    # environment
    parser.add_argument("--env",             type=str,   default="CartPole-v1")
    # core hyper-parameters
    parser.add_argument("--batch-size",      type=int,   default=32)
    parser.add_argument("--memory-size",     type=int,   default=100_000)
    parser.add_argument("--lr",              type=float, default=1e-4)
    parser.add_argument("--discount-factor", type=float, default=0.99)
    parser.add_argument("--epsilon-start",   type=float, default=1.0)
    parser.add_argument("--epsilon-decay",   type=float, default=0.999_999)
    parser.add_argument("--epsilon-min",     type=float, default=0.05)
    parser.add_argument("--target-update-frequency", type=int, default=1_000)
    parser.add_argument("--replay-start-size",        type=int, default=50_000)
    parser.add_argument("--max-episode-steps",        type=int, default=10_000)
    parser.add_argument("--train-per-step", type=int,   default=1)
    # Task 3 enhancements
    parser.add_argument("--use-double",  action="store_true")
    parser.add_argument("--use-per",     action="store_true")
    parser.add_argument("--per-alpha",   type=float, default=0.6)
    parser.add_argument("--per-beta",    type=float, default=0.4)
    parser.add_argument("--multi-step",  type=int,   default=1)
    args = parser.parse_args()

    wandb.init(project="DLP-Lab5-DQN", name=args.wandb_run_name, save_code=True)
    agent = DQNAgent(env_name=args.env, args=args)
    agent.run()
