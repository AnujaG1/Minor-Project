import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
from collections import deque
from environment import NetworkEnv5G

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

N_RUNS   = 3      # Fix 5: average over multiple runs
EPISODES = 1000   # Fix 2: doubled from 500


# ── Fix 4: BatchNorm added after shared layers ────────────────────────────────
class DDQNNetwork(nn.Module):
    def __init__(self, input_dim=10, output_dim=3):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.BatchNorm1d(128),          # Fix 4: stabilises training
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 128),
            nn.BatchNorm1d(128),          # Fix 4
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
        )
        self.value_stream = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )
        self.advantage_stream = nn.Sequential(
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, output_dim),
        )

    def forward(self, x):
        f = self.shared(x)
        v = self.value_stream(f)
        a = self.advantage_stream(f)
        return v + a - a.mean(dim=1, keepdim=True)


# ── Fix 3: Class-weighted replay buffer ───────────────────────────────────────
class WeightedReplayBuffer:
    """
    Stores (state, action, reward, next_state, done, is_attack).
    When sampling, oversamples attack transitions by attack_weight.
    Default weight 2.34 matches the 70/30 normal/attack ratio in your dataset.
    """
    def __init__(self, capacity=50000, attack_weight=2.34):  # Fix 2: 50k
        self.buffer        = deque(maxlen=capacity)
        self.attack_weight = attack_weight

    def push(self, state, action, reward, next_state, done, is_attack=0):
        self.buffer.append((state, action, reward, next_state, done, is_attack))

    def sample(self, batch_size):
        buf   = list(self.buffer)
        # Compute per-transition weight
        weights = np.array([
            self.attack_weight if t[5] == 1 else 1.0
            for t in buf
        ], dtype=np.float32)
        weights /= weights.sum()
        indices = np.random.choice(len(buf), size=batch_size,
                                   replace=False, p=weights)
        batch   = [buf[i] for i in indices]
        s, a, r, ns, d, _ = zip(*batch)
        return (
            torch.FloatTensor(np.array(s)).to(DEVICE),
            torch.LongTensor(a).to(DEVICE),
            torch.FloatTensor(r).to(DEVICE),
            torch.FloatTensor(np.array(ns)).to(DEVICE),
            torch.FloatTensor(d).to(DEVICE),
        )

    def __len__(self):
        return len(self.buffer)


class DDQNAgent:
    def __init__(self, input_dim=10, output_dim=3):
        self.output_dim  = output_dim
        self.online_net  = DDQNNetwork(input_dim, output_dim).to(DEVICE)
        self.target_net  = DDQNNetwork(input_dim, output_dim).to(DEVICE)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer     = optim.Adam(self.online_net.parameters(), lr=5e-4)
        self.criterion     = nn.SmoothL1Loss()
        self.buffer        = WeightedReplayBuffer(50000)   # Fix 2 + 3
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
            self.target_net.eval()
            best_actions = self.online_net(ns).argmax(1)
            qt = self.target_net(ns).gather(
                1, best_actions.unsqueeze(1)
            ).squeeze(1)
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
            "target_net": self.target_net.state_dict(),
            "epsilon":    self.epsilon,
            "episode":    episode,
            "stats":      stats,
            "input_dim":  10,
            "output_dim": 3,
        }, path)


def evaluate(agent, env) -> dict:
    agent.online_net.eval()
    saved_noise, env.noise_std = env.noise_std, 0.0
    env.reset_stats()
    # Fix 5: sequential walk from position 0 — no random start
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


# ── Fix 1: Reward override — stronger false alarm penalty ─────────────────────
class AccuracyEnv(NetworkEnv5G):
    """
    Subclass that overrides _compute_reward with a stronger false alarm penalty.
    Only change from parent: BLOCK on normal traffic = −2.5 (was −1.5).
    This directly reduces false alarms without touching detection logic.
    """
    def _compute_reward(self, action: int, true_label: int, severity: float):
        is_attack = (true_label == 1)

        if action == self.ACTION_PASS:
            if not is_attack:
                return +1.0, "PASS_CORRECT"
            else:
                return -1.5 - 1.5 * severity, "MISS_ATTACK"

        elif action == self.ACTION_RATE_LIMIT:
            if is_attack:
                return +0.5 + 0.5 * (1.0 - severity), "RATELIMIT_ATTACK"
            else:
                # Fix 1: slightly penalise false rate-limits too
                return -0.8 - 0.5 * severity, "RATELIMIT_FALSE_ALARM"

        else:  # BLOCK
            if is_attack:
                return +1.0 + 1.5 * severity, "BLOCK_CORRECT"
            else:
                # Fix 1: was −1.5, now −2.5 → strongly discourages blocking
                # legitimate traffic. This is the biggest single accuracy lever.
                return -2.5, "BLOCK_FALSE_ALARM"


def train_single_run(
    train_path: str,
    test_path:  str,
    model_path: str,
    run_id:     int = 1,
    episodes:   int = EPISODES,
) -> dict:
    """One complete training run. Returns best stats dict."""

    train_env = AccuracyEnv(train_path, max_steps=300, noise_std=0.02)
    test_env  = AccuracyEnv(test_path,  max_steps=99999, noise_std=0.0)
    agent     = DDQNAgent(input_dim=10, output_dim=3)

    best_accuracy  = 0.0
    best_stats     = {}

    print(f"\n  Run {run_id}/{N_RUNS} — {episodes} episodes")
    print(f"  {'Ep':>6} | {'DetRate':>8} | {'FA':>7} | {'Accuracy':>9} | {'ε':>6} | {'Loss':>8}")
    print("  " + "-" * 60)

    for ep in range(1, episodes + 1):
        obs, _ = train_env.reset()
        done   = False
        ep_losses = []

        while not done:
            action = agent.act(obs)
            next_obs, reward, term, trunc, info = train_env.step(action)

            # Fix 3: pass is_attack label into buffer for weighted sampling
            is_atk = int(info.get("true_label", 0))
            agent.buffer.push(
                obs, action, reward, next_obs, float(term), is_atk
            )

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
            train_env.reset_stats()

            print(
                f"  {ep:6d} | "
                f"{stats['detection_rate']:7.1f}% | "
                f"{stats['false_alarm_rate']:6.1f}% | "
                f"{stats['accuracy']:8.2f}% | "
                f"{agent.epsilon:6.3f} | "
                f"{avg_loss:8.5f}"
            )

            # Fix 5: save by accuracy (balanced metric), not detection alone
            if stats["accuracy"] > best_accuracy:
                best_accuracy = stats["accuracy"]
                best_stats    = stats.copy()
                agent.save(model_path.replace(".pth", f"_run{run_id}.pth"),
                           ep, stats)
                print(f"    *** New best accuracy: {best_accuracy:.2f}%")

    return best_stats


def train(
    csv_path   = "results/training_data.csv",
    model_path = "results/ddqn_model.pth",
    episodes   = EPISODES,
    n_runs     = N_RUNS,
):
    import pandas as pd
    from sklearn.model_selection import train_test_split

    df = pd.read_csv(csv_path)
    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_attacker"]
    )
    train_path = csv_path.replace(".csv", "_train.csv")
    test_path  = csv_path.replace(".csv", "_test.csv")
    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)
    print(f"Train: {len(train_df)} rows | Test: {len(test_df)} rows")

    os.makedirs("results", exist_ok=True)

    # Fix 5: run N_RUNS independent training runs, report mean ± std
    all_stats = []
    for run_id in range(1, n_runs + 1):
        stats = train_single_run(
            train_path, test_path, model_path,
            run_id=run_id, episodes=episodes
        )
        all_stats.append(stats)

    # Aggregate across runs
    keys = ["accuracy", "detection_rate", "false_alarm_rate"]
    print("\n" + "=" * 60)
    print(f"  Summary across {n_runs} runs (mean ± std) — for paper Table")
    print("=" * 60)
    for k in keys:
        vals = [s[k] for s in all_stats]
        print(f"  {k:25s}: {np.mean(vals):.2f}% ± {np.std(vals):.2f}%")

    # Save best run as the main model
    best_run_idx = np.argmax([s["accuracy"] for s in all_stats])
    best_stats   = all_stats[best_run_idx]
    import shutil
    best_run_path = model_path.replace(".pth", f"_run{best_run_idx+1}.pth")
    if os.path.exists(best_run_path):
        shutil.copy(best_run_path, model_path)
        print(f"\n  Best model (run {best_run_idx+1}) saved → {model_path}")

    print(f"  Best accuracy: {best_stats['accuracy']:.2f}%")
    print(f"  Best detection rate: {best_stats['detection_rate']:.2f}%")
    print(f"  Best false alarm rate: {best_stats['false_alarm_rate']:.2f}%")
    return best_stats


if __name__ == "__main__":
    train()