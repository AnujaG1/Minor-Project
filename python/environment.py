"""
environment.py
==============
Gymnasium environment for training the DQN agent.
Wraps training_data.csv as a step-through environment.

Changes from original:
  - feature_cols updated to 7 new behavioural features
  - observation_space shape (7,) not (6,)
  - Small Gaussian noise added on reset() to prevent memorisation
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class NetworkEnv5G(gym.Env):

    metadata = {"render_modes": ["human"]}

    # Asymmetric rewards — missing an attack is 4x worse than a false alarm
    REWARD_CORRECT_ATTACK = +2.0
    REWARD_CORRECT_NORMAL = +1.0
    REWARD_MISSED_ATTACK  = -2.0
    REWARD_FALSE_ALARM    = -0.5

    def __init__(self, csv_path: str, max_steps: int = 200,
                 render_mode=None, noise_std: float = 0.02):
        """
        csv_path   : path to training_data.csv
        max_steps  : steps per episode
        render_mode: "human" to print each step
        noise_std  : std dev of Gaussian noise added to observations
                     during training to prevent memorising exact values
                     set to 0.0 during evaluation
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_steps   = max_steps
        self.noise_std   = noise_std

        print(f"[Env] Loading {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"[Env] {len(self.df)} rows | "
              f"attacks={self.df['is_attacker'].sum()} | "
              f"normal={(self.df['is_attacker']==0).sum()}")

        # 7 behavioural features — no port flag, no sim_time
        self.feature_cols = [
            "f1_pkt_rate",
            "f2_pkt_size",
            "f3_interval",
            "f4_jitter",
            "f5_burst",
            "f6_size_unif",
            "f7_zscore",
        ]

        # Validate columns exist
        missing = [c for c in self.feature_cols if c not in self.df.columns]
        if missing:
            raise ValueError(
                f"[Env] Missing columns in CSV: {missing}\n"
                f"Available: {list(self.df.columns)}\n"
                f"Re-run data_collector.py to regenerate CSV."
            )

        self.states = self.df[self.feature_cols].values.astype(np.float32)
        self.labels = self.df["is_attacker"].values.astype(np.int64)
        self.nodes  = self.df["node"].values

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(7,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(2)

        # Episode state
        self.current_pos    = 0
        self.episode_reward = 0.0
        self.steps_taken    = 0

        # Statistics
        self.total_correct  = 0
        self.total_wrong    = 0
        self.attacks_caught = 0
        self.attacks_missed = 0
        self.false_alarms   = 0

        print(f"[Env] Ready — obs_space={self.observation_space.shape} "
              f"noise_std={self.noise_std}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        # Random start position so agent can't memorise episode order
        self.current_pos    = int(np.random.randint(0, len(self.states) - 1))
        self.episode_reward = 0.0
        self.steps_taken    = 0

        obs = self._get_obs()
        return obs, {"start_pos": self.current_pos}

    def _get_obs(self) -> np.ndarray:
        obs = self.states[self.current_pos].copy()
        if self.noise_std > 0:
            # Add Gaussian noise to prevent the model memorising exact values.
            # Forces learning the PATTERN not the specific numbers.
            # e.g. attacker always 0.667 rate → with noise: 0.651, 0.682, 0.659
            noise = np.random.normal(0, self.noise_std, size=obs.shape)
            obs   = np.clip(obs + noise, 0.0, 1.0).astype(np.float32)
        return obs

    def step(self, action: int):
        true_label   = self.labels[self.current_pos]
        current_node = self.nodes[self.current_pos]

        if action == 1 and true_label == 1:
            reward = self.REWARD_CORRECT_ATTACK
            self.attacks_caught += 1
            self.total_correct  += 1
            outcome = "CORRECT ATTACK"

        elif action == 0 and true_label == 0:
            reward = self.REWARD_CORRECT_NORMAL
            self.total_correct += 1
            outcome = "CORRECT NORMAL"

        elif action == 0 and true_label == 1:
            reward = self.REWARD_MISSED_ATTACK
            self.attacks_missed += 1
            self.total_wrong    += 1
            outcome = "MISSED ATTACK"

        else:
            reward = self.REWARD_FALSE_ALARM
            self.false_alarms += 1
            self.total_wrong  += 1
            outcome = "FALSE ALARM"

        self.current_pos    += 1
        self.episode_reward += reward
        self.steps_taken    += 1

        terminated = (self.current_pos >= len(self.states) - 1)
        truncated  = (self.steps_taken >= self.max_steps)

        if terminated:
            self.current_pos = 0

        obs = self._get_obs()

        info = {
            "true_label"  : int(true_label),
            "node"        : current_node,
            "action_taken": int(action),
            "outcome"     : outcome,
            "reward"      : reward,
            "step"        : self.steps_taken,
        }

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            node  = self.nodes[self.current_pos]
            label = self.labels[self.current_pos]
            print(f"Step {self.steps_taken:3d} | {node:<15} | label={label}")

    def close(self):
        pass

    def get_accuracy(self) -> float:
        total = self.total_correct + self.total_wrong
        return (self.total_correct / total * 100) if total > 0 else 0.0

    def get_stats(self) -> dict:
        return {
            "accuracy"      : round(self.get_accuracy(), 2),
            "attacks_caught": self.attacks_caught,
            "attacks_missed": self.attacks_missed,
            "false_alarms"  : self.false_alarms,
            "total_correct" : self.total_correct,
            "total_wrong"   : self.total_wrong,
        }

    def reset_stats(self):
        self.total_correct = self.total_wrong = 0
        self.attacks_caught = self.attacks_missed = self.false_alarms = 0


# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    CSV = os.path.join(os.path.dirname(__file__), "results", "training_data.csv")
    env = NetworkEnv5G(CSV, render_mode="human")

    obs, info = env.reset(seed=42)
    print(f"\nobs shape : {obs.shape}")
    print(f"obs values: {obs}")

    for i in range(5):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)
        print(f"  action={action} reward={reward:+.1f} outcome={info['outcome']}")

    env.close()
    print("\nEnvironment test passed!")