import torch , torch.nn as nn, torch.optim as optim
import numpy as np
import os, random
from collections import deque
from environment import NetworkEnv5G
import pandas as pd
from sklearn.model_selection import train_test_split

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

class DQNNetwork(nn.Module):
    def __init__(self, input_dim = 5, output_dim: int = 2):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        return self.network(x)

class ReplayBuffer:
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)).to(DEVICE),
            torch.LongTensor(actions).to(DEVICE),
            torch.FloatTensor(rewards).to(DEVICE),
            torch.FloatTensor(np.array(next_states)).to(DEVICE),
            torch.FloatTensor(dones).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    Double DQN:
      online_net  — learns every step
      target_net  — frozen copy, updates every 10 episodes
    Prevents training target from shifting too fast.
    """

    def __init__(self, input_dim=5, output_dim=2):
        self.online_net = DQNNetwork(input_dim, output_dim).to(DEVICE)
        self.target_net = DQNNetwork(input_dim, output_dim).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer  = optim.Adam(self.online_net.parameters(), lr=0.001)
        self.criterion  = nn.MSELoss()
        self.buffer     = ReplayBuffer(capacity=10000)
        self.epsilon             = 1.0    # start fully random
        self.epsilon_decay       = 0.995
        self.epsilon_min         = 0.01
        self.gamma               = 0.95
        self.batch_size          = 64

    def select_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 1)
        with torch.no_grad():
            s = torch.FloatTensor(state).unsqueeze(0).to(DEVICE)
            return int(self.online_net(s).argmax(1).item())

    def train_step(self):
        if len(self.buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)

        # Current Q values from online net
        q_current = self.online_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        # Target Q values from target net (Double DQN)
        with torch.no_grad():
            next_actions = self.online_net(next_states).argmax(1)
            q_next       = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_target = rewards + self.gamma * q_next * (1 - dones)

        loss = self.criterion(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_target(self):
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self, path, episode):
        torch.save({
            "online_net": self.online_net.state_dict(),
            "target_net": self.target_net.state_dict(),
            "epsilon"   : self.epsilon,
            "episode"   : episode,
        }, path)

def evaluate(agent, env):
    agent.online_net.eval()
    env_noise_std  = env.noise_std
    env.noise_std  = 0.0          # disable noise for evaluation
    env.reset_stats()

    obs, _ = env.reset()
    done   = False
    steps  = 0

    while not done and steps < len(env.states):
        with torch.no_grad():
            s      = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            action = int(agent.online_net(s).argmax(dim=1).item())

        obs, _, terminated, truncated, _ = env.step(action)
        done  = terminated or truncated
        steps += 1

    stats         = env.get_stats()
    env.noise_std = env_noise_std   # restore noise
    agent.online_net.train()
    return stats


# TRAINING 

def train(csv_path="results/training_data.csv", model_path="results/dqn_model.pth", episodes=500):

    print("[Train] Splitting data 80/20 train/test...")
    df       = pd.read_csv(csv_path)
    df["is_attacker"] = (df["is_attacker"] > 0.5).astype(int)

    attack_df = df[df["is_attacker"] == 1]
    normal_df = df[df["is_attacker"] == 0].sample(len(attack_df), random_state=42)

    df = pd.concat([attack_df, normal_df]).sample(frac=1, random_state=42)

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    train_path = csv_path.replace(".csv", "_train.csv")
    test_path  = csv_path.replace(".csv", "_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,  index=False)
    print(f"  Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    train_env = NetworkEnv5G(train_path, max_steps=200, noise_std=0.02)
    test_env  = NetworkEnv5G(test_path,  max_steps=9999, noise_std=0.0)

    agent = DQNAgent(input_dim=5)
    os.makedirs("results", exist_ok=True)

    best_test_acc  = 0.0
    best_episode   = 0

    print(f"\n[Train] Starting {episodes} episodes on {DEVICE}")
    print("-" * 60)

    for ep in range(1, episodes + 1):

        obs, _ = train_env.reset()
        ep_reward = 0.0
        done      = False

        while not done:
            action = agent.select_action(obs)
            next_obs, reward, terminated, truncated, _ = train_env.step(action)
            done = terminated or truncated

            agent.buffer.push(
                obs, action, reward,
                next_obs, float(terminated)
            )
            agent.train_step()
            obs        = next_obs
            ep_reward += reward

        agent.decay_epsilon()

        if ep % 10 == 0:
            agent.update_target()

        # ── Evaluate on test set every 50 episodes ──────────
        if ep % 50 == 0 or ep == episodes:
            test_stats  = evaluate(agent, test_env)
            train_stats = train_env.get_stats()
            train_env.reset_stats()

            test_acc  = test_stats["accuracy"]
            train_acc = train_stats["accuracy"] if train_stats["total_correct"] + train_stats["total_wrong"] > 0 else 0

            total_attacks = test_stats['attacks_caught'] + test_stats['attacks_missed']

            print(f"Ep {ep:4d} | "
                f"train_acc={train_acc:.1f}% | "
                f"test_acc={test_acc:.1f}% | "
                f"caught={test_stats['attacks_caught']} | "
                f"missed={test_stats['attacks_missed']} | "
                f"total_attacks={total_attacks} | "
                f"eps={agent.epsilon:.3f}")

            # Save best model based on TEST accuracy
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_episode  = ep
                agent.save(model_path, ep)
                print(f"  *** New best: {test_acc:.1f}% — saved to {model_path}")

    print(f"\n[Train] Done. Best test accuracy: {best_test_acc:.1f}% at episode {best_episode}")
    print(f"[Train] Model saved: {model_path}")
    
    return best_test_acc
if __name__ == "__main__":
    train()
   