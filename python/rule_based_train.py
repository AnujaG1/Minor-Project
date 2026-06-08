"""
rule_based.py
Rule-based threshold baseline for fair comparison with DQN and DDQN.

Strategy: require a node to exceed 3 out of 4 thresholds to be BLOCKED.
This reduces false alarms significantly because legitimate UEs rarely exceed
more than 1-2 features simultaneously, even during burst periods.

Scoring:
  0-1 flags → PASS
  2 flags   → RATE_LIMIT
  3+ flags  → BLOCK
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from environment import NetworkEnv5G


# ── Thresholds ────────────────────────────────────────────────────────────────
# Strategy: set each threshold at the 90th percentile of UE values so that
# a legitimate UE can trip 1 flag during a burst but almost never trips 3+.
# Attackers consistently exceed all 4 once their flood is established.
#
# From your data distribution:
#   UE    f1 ~0.005-0.013   f3 ~0.036-0.092   f8 ~0.0-0.23   f10 ~0.009-0.013
#   Atk   f1 ~0.010-0.029   f3 ~0.096-0.117   f8 ~0.0-0.45   f10 ~0.008-0.029
#
# We also add f2 (mean_rate) as a 5th feature for a tiebreaker.
# Attackers have higher sustained mean rates than UEs.

PKT_RATE_THRESH  = 0.014   # f1:  just above typical UE max of 0.013
MEAN_RATE_THRESH = 0.012   # f2:  sustained rate; attackers ramp up over time
BURST_THRESH     = 0.093   # f3:  UE max ~0.092, attacker min ~0.096
ZSCORE_THRESH    = 0.25    # f8:  UE 90th pct ~0.23, attackers often >0.40
PEAK_THRESH      = 0.014   # f10: same reasoning as f1


def rule_based_action(obs: np.ndarray) -> int:
    """
    Five-feature threshold policy.
    obs indices (matching your CSV feature order):
      obs[0]  = f1_pkt_rate
      obs[1]  = f2_mean_rate
      obs[2]  = f3_burst_ratio
      obs[7]  = f8_cell_zscore
      obs[9]  = f10_peak_rate

    Scoring:
      0-1 flags → PASS        (likely legitimate)
      2 flags   → RATE_LIMIT  (suspicious, graduated response)
      3+ flags  → BLOCK       (confirmed attack pattern)

    Raising the BLOCK threshold from 2→3 flags is the key fix.
    A UE having one bad second trips at most 1-2 features.
    A sustained attacker trips 3-4 features consistently.
    """
    f1  = obs[0]   # pkt_rate
    f2  = obs[1]   # mean_rate
    f3  = obs[2]   # burst_ratio
    f8  = obs[7]   # cell_zscore
    f10 = obs[9]   # peak_rate

    flags = 0
    if f1  > PKT_RATE_THRESH:  flags += 1
    if f2  > MEAN_RATE_THRESH: flags += 1
    if f3  > BURST_THRESH:     flags += 1
    if f8  > ZSCORE_THRESH:    flags += 1
    if f10 > PEAK_THRESH:      flags += 1

    if   flags >= 3: return 2   # BLOCK
    elif flags == 2: return 1   # RATE_LIMIT
    else:            return 0   # PASS


def evaluate_rule_based(env: NetworkEnv5G) -> dict:
    """
    Walk the full test set sequentially from position 0.
    Same evaluation pattern as DDQN and DQN for fair comparison.
    """
    env.reset_stats()
    env.current_pos    = 0
    env.steps_taken    = 0
    env.episode_reward = 0.0

    for _ in range(len(env.states)):
        obs    = env._obs()
        action = rule_based_action(obs)
        env.step(action)

    return env.get_stats()


def run(csv_path="results/training_data.csv"):
    df = pd.read_csv(csv_path)
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_attacker"]
    )
    test_path = csv_path.replace(".csv", "_test.csv")
    test_df.to_csv(test_path, index=False)

    test_env = NetworkEnv5G(test_path, max_steps=99999, noise_std=0.0)
    stats    = evaluate_rule_based(test_env)

    total_atk = stats['attacks_caught'] + stats['attacks_missed']
    total_normal = stats.get('false_alarms', 0) + (
        stats.get('normal_total', 0) - stats.get('false_alarms', 0)
    )

    print("\n── Rule-Based Baseline Results ──────────────────────────────")
    print(f"  Detection rate  (attack recall) : {stats['detection_rate']:.2f}%")
    print(f"  False alarm rate                : {stats['false_alarm_rate']:.2f}%")
    print(f"  Overall accuracy                : {stats['accuracy']:.2f}%")
    print(f"  Attacks caught                  : {stats['attacks_caught']} / {total_atk}")
    print(f"  False alarms on normal traffic  : {stats['false_alarms']}")
    print("──────────────────────────────────────────────────────────────")
    print("\nThresholds used:")
    print(f"  f1  pkt_rate    > {PKT_RATE_THRESH}  (5 features scored)")
    print(f"  f2  mean_rate   > {MEAN_RATE_THRESH}")
    print(f"  f3  burst_ratio > {BURST_THRESH}")
    print(f"  f8  cell_zscore > {ZSCORE_THRESH}")
    print(f"  f10 peak_rate   > {PEAK_THRESH}")
    print(f"  BLOCK if flags >= 3, RATE_LIMIT if flags == 2, PASS otherwise")
    return stats


if __name__ == "__main__":
    run()