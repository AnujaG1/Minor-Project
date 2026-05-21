"""
rule_based.py
Rule-based threshold baseline for fair comparison with DQN and DDQN.

Logic (mirrors a real reactive IDS):
  - f1 (pkt_rate)   > PKT_RATE_THRESH  → suspect
  - f8 (cell_zscore) > ZSCORE_THRESH    → suspect
  - f3 (burst_ratio) > BURST_THRESH     → suspect
  - f9 (consecutive) >= CONSEC_THRESH   → suspect
  
  Scoring:
    0 suspicious features → PASS
    1 suspicious feature  → RATE_LIMIT
    2+ suspicious features → BLOCK

Thresholds are calibrated to your dataset's feature ranges [0,1].
No learning occurs — this is a fixed, hand-crafted policy.

Usage:  python3 rule_based.py
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from environment import NetworkEnv5G


# ── Tunable thresholds ────────────────────────────────────────────────────────
PKT_RATE_THRESH = 0.020   # f1: normalised packet rate (attackers ~0.026)
ZSCORE_THRESH   = 0.15    # f8: cell z-score (attackers ~0.45)
BURST_THRESH    = 0.065   # f3: burst ratio  (attackers ~0.075)
CONSEC_THRESH   = 1       # f9: consecutive high-rate steps


def rule_based_action(obs: np.ndarray) -> int:
    """
    Pure threshold policy — no learning.
    Returns 0 (PASS), 1 (RATE_LIMIT), or 2 (BLOCK).
    """
    f1  = obs[0]   # pkt_rate
    f3  = obs[2]   # burst_ratio
    f8  = obs[7]   # cell_zscore
    f9  = obs[8]   # consecutive

    flags = 0
    if f1  > PKT_RATE_THRESH: flags += 1
    if f8  > ZSCORE_THRESH:   flags += 1
    if f3  > BURST_THRESH:    flags += 1
    if f9  >= CONSEC_THRESH:  flags += 1

    if flags >= 2:
        return 2   # BLOCK
    elif flags == 1:
        return 1   # RATE_LIMIT
    else:
        return 0   # PASS


def evaluate_rule_based(env: NetworkEnv5G) -> dict:
    """Run the rule-based policy through the full test set."""
    env.reset_stats()
    obs, _ = env.reset()

    for _ in range(len(env.states)):
        action = rule_based_action(obs)
        obs, _, term, trunc, _ = env.step(action)
        if term or trunc:
            break

    return env.get_stats()


def run(csv_path="results/training_data.csv"):
    df = pd.read_csv(csv_path)
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_attacker"]
    )
    test_path = csv_path.replace(".csv", "_test.csv")
    test_df.to_csv(test_path, index=False)

    test_env = NetworkEnv5G(test_path, max_steps=9999, noise_std=0.0)
    stats    = evaluate_rule_based(test_env)

    total_atk = stats['attacks_caught'] + stats['attacks_missed']
    print("\n── Rule-Based Baseline Results ──────────────────────")
    print(f"  Detection rate : {stats['detection_rate']:.2f}%")
    print(f"  False alarm    : {stats['false_alarm_rate']:.2f}%")
    print(f"  Accuracy       : {stats['accuracy']:.2f}%")
    print(f"  Attacks caught : {stats['attacks_caught']} / {total_atk}")
    print(f"  False alarms   : {stats['false_alarms']}")
    print("─────────────────────────────────────────────────────")
    return stats


if __name__ == "__main__":
    run()