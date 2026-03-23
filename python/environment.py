"""
environment.py  —  v2
Gymnasium environment for DDQN training.

WHY THE OLD VERSION BEHAVED LIKE IF-ELSE
─────────────────────────────────────────
1. action_space = Discrete(2): only pass/block. No nuance possible.
   The agent had zero incentive to distinguish severity — blocking a slow
   DDoS and blocking a volumetric flood gave identical reward. So it learned
   one threshold rule for both: if pkt_rate > X → block.

2. Reward = f(label match): classification signal, not mitigation signal.
   reward = +2 if action==label else -2. This means the Q-values for
   different actions only differ by whether they match the label — not by
   what they do to the network. The agent learned to predict labels, not
   to mitigate attacks.

3. Random episode steps: self.current_pos = random each step.
   The agent never saw two consecutive states from the same node.
   It couldn't learn that a node ramping from 5→10→20 pkt/s is more
   dangerous than one already at 20 pkt/s — because it never saw the ramp.

4. 5 features from old extractor: f1..f5 were pre-computed by OMNeT++,
   including interval which was the config param not a measured IAT.
   The agent learned to separate config values, not traffic behaviour.

WHAT IS FIXED
─────────────
- action_space = Discrete(3): PASS / RATE_LIMIT / BLOCK
- Reward is QoS-based: considers attack severity, false alarm cost,
  proportionality (rate-limiting a flood is better than doing nothing,
  but worse than blocking it; blocking a normal UE is very costly)
- Episodes are sequential: agent sees consecutive states per node,
  allowing it to learn temporal patterns (ramp-up, persistence, pulsing)
- observation_space = Box(10,): matches feature_extractor.py get_state()
"""

import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces


class NetworkEnv5G(gym.Env):

    ACTION_PASS       = 0
    ACTION_RATE_LIMIT = 1
    ACTION_BLOCK      = 2
    ACTION_NAMES      = {0: "PASS", 1: "RATE_LIMIT", 2: "BLOCK"}

    N_FEATURES = 10   # must match feature_extractor.py get_state() output

    def __init__(self, csv_path, max_steps=300,
                 render_mode=None, noise_std=0.02):
        super().__init__()
        self.render_mode = render_mode
        self.max_steps   = max_steps
        self.noise_std   = noise_std

        print(f"[Env] Loading {csv_path}")
        df = pd.read_csv(csv_path)
        print(f"[Env] {len(df)} rows | "
              f"attacks={df['is_attacker'].sum()} | "
              f"normal={(df['is_attacker']==0).sum()}")

        # ── Feature columns — must match data_collector.py header ────────────
        feature_cols = [
            "f1_pkt_rate",
            "f2_mean_rate",
            "f3_burst_ratio",
            "f4_rate_change",
            "f5_rate_trend",
            "f6_flow_duration",
            "f7_activity_ratio",
            "f8_cell_zscore",
            "f9_consecutive",
            "f10_peak_rate",
        ]
        missing = [c for c in feature_cols if c not in df.columns]
        if missing:
            raise ValueError(
                f"Missing columns: {missing}\n"
                f"Re-run data_collector.py with the new feature_extractor.py"
            )

        # ── Sort by node + time so episodes are temporally coherent ──────────
        if "sim_time" in df.columns:
            df = df.sort_values(["node", "sim_time"]).reset_index(drop=True)

        self.states = df[feature_cols].values.astype(np.float32)
        self.labels = df["is_attacker"].values.astype(np.int64)
        self.nodes  = df["node"].values

        # ── Severity: derived from f1 (pkt_rate) and f2 (burst_ratio) ────────
        # Used to scale rewards — high-severity attacks cost more to miss
        pkt_rate   = self.states[:, 0]   # f1 normalised [0,1]
        burst      = self.states[:, 1]   # f2 normalised [0,1]
        self.severity = np.clip(pkt_rate * 0.6 + burst * 0.4, 0.0, 1.0)

        self.observation_space = spaces.Box(
            0.0, 1.0, shape=(self.N_FEATURES,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)   # PASS / RATE_LIMIT / BLOCK

        self._reset_counters()
        self.current_pos    = 0
        self.episode_reward = 0.0
        self.steps_taken    = 0

        print(f"[Env] Ready — obs=({self.N_FEATURES},) actions=3 "
              f"noise={noise_std}")

    # ── Gym interface ─────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed, options=options)
        # Start at a random position but step sequentially from there
        self.current_pos    = int(np.random.randint(0, len(self.states) - 1))
        self.episode_reward = 0.0
        self.steps_taken    = 0
        return self._obs(), {"start_pos": self.current_pos}

    def step(self, action):
        true_label = int(self.labels[self.current_pos])
        severity   = float(self.severity[self.current_pos])
        node       = self.nodes[self.current_pos]

        reward, outcome = self._compute_reward(action, true_label, severity)

        self._update_counters(action, true_label, outcome)

        # Sequential advance — agent sees the next state for the same node
        self.current_pos    += 1
        self.episode_reward += reward
        self.steps_taken    += 1

        terminated = self.current_pos >= len(self.states) - 1
        truncated  = self.steps_taken >= self.max_steps
        if terminated:
            self.current_pos = 0

        return self._obs(), reward, terminated, truncated, {
            "true_label": true_label,
            "node":       node,
            "outcome":    outcome,
            "severity":   round(severity, 3),
            "action":     self.ACTION_NAMES[action],
            "step":       self.steps_taken,
        }

    def render(self):
        if self.render_mode == "human":
            pos  = self.current_pos
            node = self.nodes[pos] if pos < len(self.nodes) else "?"
            print(f"Step {self.steps_taken:3d} | {node:<15} | "
                  f"label={self.labels[pos] if pos < len(self.labels) else '?'}")

    def close(self): pass

    # ── Reward function ───────────────────────────────────────────────────────

    def _compute_reward(self, action: int, true_label: int,
                        severity: float) -> tuple[float, str]:
        """
        QoS-based reward. Severity ∈ [0,1] scales attack-related rewards.

        Key design decisions:
        - Missing a high-severity attack is the worst outcome (-3 at max)
        - Blocking a legitimate UE is costly (disrupts real traffic: -1.5)
        - Rate-limiting a legitimate UE is a smaller false alarm (-0.5)
        - Rate-limiting an attack is always better than doing nothing,
          but worse than blocking it (teaches proportionality)
        - Passing normal traffic: small positive (encourages not over-blocking)

        This forces the agent to learn three things:
          1. Severity matters — not just presence/absence
          2. Rate-limit before block for borderline cases
          3. False alarms on heavy UEs are expensive
        """
        is_attack  = (true_label == 1)
        is_normal  = (true_label == 0)

        if action == self.ACTION_PASS:
            if is_normal:
                # Correct — let legitimate traffic flow
                reward  = +1.0
                outcome = "PASS_CORRECT"
            else:
                # Missed attack — cost scales with severity
                reward  = -1.5 - 1.5 * severity   # range: [-1.5, -3.0]
                outcome = "MISS_ATTACK"

        elif action == self.ACTION_RATE_LIMIT:
            if is_attack:
                # Partial mitigation — better than pass, worse than block
                # Reward scales with severity: high-severity attacks need blocking
                reward  = +0.5 + 0.5 * (1.0 - severity)  # range: [+0.5, +1.0]
                # for low severity attacker (slow DDoS): rate-limit is near-optimal
                # for high severity attacker (flood): rate-limit is suboptimal
                outcome = "RATELIMIT_ATTACK"
            else:
                # False alarm — rate-limiting a normal UE degrades QoS
                # Cost scales with UE activity (f7_flow_duration ≈ severity proxy)
                reward  = -0.5 - 0.5 * severity   # range: [-0.5, -1.0]
                outcome = "RATELIMIT_FALSE_ALARM"

        else:   # ACTION_BLOCK
            if is_attack:
                # Best mitigation — reward scales with severity
                # Blocking a high-severity attacker is maximally rewarded
                reward  = +1.0 + 1.5 * severity   # range: [+1.0, +2.5]
                outcome = "BLOCK_CORRECT"
            else:
                # Worst outcome — blocking a legitimate UE
                # This is more costly than missing a low-severity attack
                reward  = -1.5
                outcome = "BLOCK_FALSE_ALARM"

        return float(reward), outcome

    # ── Stats ─────────────────────────────────────────────────────────────────

    def _reset_counters(self):
        self.cnt = {
            "PASS_CORRECT":        0,
            "MISS_ATTACK":         0,
            "RATELIMIT_ATTACK":    0,
            "RATELIMIT_FALSE_ALARM": 0,
            "BLOCK_CORRECT":       0,
            "BLOCK_FALSE_ALARM":   0,
        }

    def _update_counters(self, action, true_label, outcome):
        if outcome in self.cnt:
            self.cnt[outcome] += 1

    def get_stats(self) -> dict:
        total_attack  = self.cnt["MISS_ATTACK"] + self.cnt["RATELIMIT_ATTACK"] + self.cnt["BLOCK_CORRECT"]
        total_normal  = self.cnt["PASS_CORRECT"] + self.cnt["RATELIMIT_FALSE_ALARM"] + self.cnt["BLOCK_FALSE_ALARM"]
        caught        = self.cnt["BLOCK_CORRECT"] + self.cnt["RATELIMIT_ATTACK"]
        missed        = self.cnt["MISS_ATTACK"]
        false_alarms  = self.cnt["RATELIMIT_FALSE_ALARM"] + self.cnt["BLOCK_FALSE_ALARM"]
        total         = sum(self.cnt.values())
        correct       = self.cnt["PASS_CORRECT"] + self.cnt["BLOCK_CORRECT"]

        return {
            "accuracy":          round(correct / total * 100, 2) if total else 0.0,
            "detection_rate":    round(caught / total_attack * 100, 2) if total_attack else 0.0,
            "false_alarm_rate":  round(false_alarms / total_normal * 100, 2) if total_normal else 0.0,
            "attacks_caught":    caught,
            "attacks_missed":    missed,
            "false_alarms":      false_alarms,
            "block_correct":     self.cnt["BLOCK_CORRECT"],
            "ratelimit_attack":  self.cnt["RATELIMIT_ATTACK"],
            "block_false_alarm": self.cnt["BLOCK_FALSE_ALARM"],
            "total":             total,
        }

    def reset_stats(self):
        self._reset_counters()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _obs(self) -> np.ndarray:
        obs = self.states[self.current_pos].copy()
        if self.noise_std > 0:
            obs = np.clip(
                obs + np.random.normal(0, self.noise_std, self.N_FEATURES),
                0.0, 1.0
            ).astype(np.float32)
        return obs