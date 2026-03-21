import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class NetworkEnv5G(gym.Env):

    REWARD_CORRECT_ATTACK = +10
    REWARD_CORRECT_NORMAL = +2
    REWARD_MISSED_ATTACK  = -10
    REWARD_FALSE_ALARM    = -1

    def __init__(self, csv_path, max_steps=200, render_mode=None, noise_std=0.02):
        super().__init__()

        self.render_mode = render_mode
        self.max_steps   = max_steps
        self.noise_std   = noise_std

        print(f"[Env] Loading {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"[Env] {len(self.df)} rows | "
              f"attacks={self.df['is_attacker'].sum()} | "
              f"normal={(self.df['is_attacker']==0).sum()}")

        self.feature_cols = [
            "f1_pkt_rate",
            "f2_pkt_size",
            "f3_interval",
            "f4_burst",
            "f5_zscore",
        ]

        # Validate columns exist
        missing = [c for c in self.feature_cols if c not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}. Re-run data_collector.py")

        self.states = self.df[self.feature_cols].values.astype(np.float32)
        self.labels = self.df["is_attacker"].values.astype(np.int64)
        self.nodes  = self.df["node"].values

        self.observation_space = spaces.Box(0.0, 1.0, shape=(5,), dtype=np.float32)
        self.action_space      = spaces.Discrete(2)

        # Episode state
        self.current_pos    = 0
        self.episode_reward = 0.0
        self.steps_taken    = 0
        self.total_correct  = 0
        self.total_wrong    = 0
        self.attacks_caught = 0
        self.attacks_missed = 0
        self.false_alarms   = 0

        print(f"[Env] Ready — obs=(5,) noise={noise_std}")

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)

        self.current_pos    = int(np.random.randint(0, len(self.states) - 1))
        self.episode_reward = 0.0
        self.steps_taken    = 0
        obs = self._get_obs()

        return obs, {"start_pos": self.current_pos}

    def _get_obs(self):
        obs = self.states[self.current_pos].copy()
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, size=obs.shape)  # adding gaussian noise to from model memorising
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

        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            node  = self.nodes[self.current_pos]
            label = self.labels[self.current_pos]
            print(f"Step {self.steps_taken:3d} | {node:<15} | label={label}")

    def close(self):
        pass

    def get_accuracy(self):
        total = self.total_correct + self.total_wrong
        return (self.total_correct / total * 100) if total > 0 else 0.0

    def get_stats(self):
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
