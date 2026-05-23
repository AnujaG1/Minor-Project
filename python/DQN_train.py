"""
train_dqn.py
Standard DQN agent — no Dueling architecture, no Double DQN target trick.
Comparable baseline to DDQN in train_rl.py.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque

DEVICE = torch.device("cpu")
EPISODES = 500


class DQNNetwork(nn.Module):
    def __init__(self, input_dim=10, output_dim=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 128),       nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, 64),        nn.ReLU(),
            nn.Linear(64, 32),         nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.net(x)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        s, a, r, ns, d = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(DEVICE),
            torch.LongTensor(a).to(DEVICE),
            torch.FloatTensor(r).to(DEVICE),
            torch.FloatTensor(np.array(ns)).to(DEVICE),
            torch.FloatTensor(d).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:
    def __init__(self, input_dim=10, output_dim=3):
        self.output_dim    = output_dim
        self.online_net    = DQNNetwork(input_dim, output_dim).to(DEVICE)
        self.target_net    = DQNNetwork(input_dim, output_dim).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer     = optim.Adam(self.online_net.parameters(), lr=5e-4)
        self.criterion     = nn.SmoothL1Loss()
        self.buffer        = ReplayBuffer(20000)
        self.epsilon       = 1.0
        self.epsilon_decay = 0.997
        self.epsilon_min   = 0.05
        self.gamma         = 0.95
        self.batch_size    = 128

    def act(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, self.output_dim - 1)
        self.online_net.eval()
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            a = int(self.online_net(s).argmax(1).item())
        self.online_net.train()
        return a

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None
        s, a, r, ns, d = self.buffer.sample(self.batch_size)

        self.online_net.train()
        q = self.online_net(s).gather(1, a.unsqueeze(1)).squeeze(1)

        with torch.no_grad():
            # Standard DQN: target net picks AND evaluates (no double trick)
            qt     = self.target_net(ns).max(1)[0]
            target = r + self.gamma * qt * (1 - d)

        loss = self.criterion(q, target)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), 10.0)
        self.optimizer.step()
        return loss.item()

    def decay(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def sync_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path, episode, stats):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "episode":    episode,
            "stats":      stats,
        }, path)


def evaluate(agent, env) -> dict:
    """
    Walk the FULL test set sequentially from position 0.
    Mirrors exactly what the DDQN evaluate() does in train_rl.py.
    The original bug: env.reset() randomised current_pos each call,
    so most evaluations landed on a slice with zero attackers → 0/0.
    """
    agent.online_net.eval()
    saved_noise, env.noise_std = env.noise_std, 0.0
    env.reset_stats()

    # Sequential walk from the beginning — same as DDQN
    env.current_pos    = 0
    env.steps_taken    = 0
    env.episode_reward = 0.0

    for _ in range(len(env.states)):
        obs = env._obs()
        with torch.no_grad():
            action = int(
                agent.online_net(
                    torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                ).argmax(1).item()
            )
        env.step(action)

    stats = env.get_stats()
    env.noise_std = saved_noise
    agent.online_net.train()
    return stats


def train(
    csv_path   = "results/training_data.csv",
    model_path = "results/dqn_model.pth",
    episodes   = EPISODES,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split
    from environment import NetworkEnv5G

    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_attacker"]
    )
    train_path = csv_path.replace(".csv", "_train.csv")
    test_path  = csv_path.replace(".csv", "_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    from environment import NetworkEnv5G
    train_env = NetworkEnv5G(train_path, max_steps=300, noise_std=0.02)
    test_env  = NetworkEnv5G(test_path,  max_steps=99999, noise_std=0.0)
    agent     = DQNAgent(input_dim=10, output_dim=3)

    os.makedirs("results", exist_ok=True)

    best_detection = 0.0
    best_stats     = {}

    print(f"\nTraining standard DQN for {episodes} episodes...\n")
    print(f"{'Ep':>5} | {'DetRate':>8} | {'FAlarm':>7} | "
          f"{'Caught/Total':>12} | {'ε':>6} | {'Loss':>8}")
    print("-" * 65)

    for ep in range(1, episodes + 1):
        obs, _ = train_env.reset()
        done   = False
        ep_losses = []

        while not done:
            action = agent.act(obs)
            next_obs, reward, term, trunc, _ = train_env.step(action)
            agent.buffer.push(obs, action, reward, next_obs, float(term))
            loss = agent.train_step()
            if loss is not None:
                ep_losses.append(loss)
            obs  = next_obs
            done = term or trunc

        agent.decay()
        if ep % 20 == 0:
            agent.sync_target()

        if ep % 50 == 0 or ep == episodes:
            stats    = evaluate(agent, test_env)
            avg_loss = np.mean(ep_losses) if ep_losses else 0.0
            total_atk = stats['attacks_caught'] + stats['attacks_missed']

            print(
                f"{ep:5d} | "
                f"{stats['detection_rate']:7.1f}% | "
                f"{stats['false_alarm_rate']:6.1f}% | "
                f"{stats['attacks_caught']:4d}/{total_atk:<8d}| "
                f"{agent.epsilon:6.3f} | "
                f"{avg_loss:8.5f}"
            )
            train_env.reset_stats()

            if stats["detection_rate"] > best_detection:
                best_detection = stats["detection_rate"]
                best_stats     = stats.copy()
                agent.save(model_path, ep, stats)
                print(f"  *** New best detection rate: {best_detection:.1f}%")

    print(f"\nDQN Training done.")
    print(f"  Best detection rate : {best_stats.get('detection_rate', 0):.2f}%")
    print(f"  False alarm rate    : {best_stats.get('false_alarm_rate', 0):.2f}%")
    print(f"  Accuracy            : {best_stats.get('accuracy', 0):.2f}%")
    return best_stats


if __name__ == "__main__":
    train()