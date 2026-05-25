"""
compare_results.py
Runs Rule-Based, DQN, and DDQN on the same fixed test split and produces
a publication-quality grouped bar chart + CSV table for your paper.

Prerequisites:
  - Run prepare_splits.py once first (or any training script that creates
    training_data_train.csv and training_data_test.csv)
  - Train DQN:  python DQN_train.py   → saves results/dqn_model.pth
  - Train DDQN: python train_rl.py    → saves results/ddqn_model.pth

Then run:
  python compare_results.py

Output:
  results/comparison_bargraph.png  — 300 DPI figure for paper
  results/comparison_bargraph.pdf  — vector PDF for submission
  results/comparison_table.csv     — raw numbers for your paper table
"""

import os
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from environment import NetworkEnv5G
from rule_based_train import rule_based_action        # just the action function
from DQN_train import DQNAgent
from train_rl  import DDQNAgent

os.makedirs("results", exist_ok=True)

TEST_PATH  = "results/training_data_test.csv"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ── Shared evaluation helper ──────────────────────────────────────────────────
# All three methods use this exact same loop so evaluation is identical.
# No env.reset() — we always walk from position 0 through every row.

def evaluate_policy(action_fn, test_path=TEST_PATH):
    """
    action_fn: callable that takes obs (np.ndarray shape [10]) and returns
               an int action (0=PASS, 1=RATE_LIMIT, 2=BLOCK).

    Creates a fresh env each call so there is zero state leakage between
    the three methods. This is the most important correctness guarantee.
    """
    env = NetworkEnv5G(test_path, max_steps=99999, noise_std=0.0)
    env.reset_stats()
    env.current_pos    = 0
    env.steps_taken    = 0
    env.episode_reward = 0.0

    for _ in range(len(env.states)):
        obs    = env._obs()
        action = action_fn(obs)
        env.step(action)

    return env.get_stats()


# ── 1. Rule-Based ─────────────────────────────────────────────────────────────

def get_rule_based_stats():
    print("\n[1/3] Evaluating Rule-Based baseline...")
    stats = evaluate_policy(rule_based_action)
    print(f"      Detection : {stats['detection_rate']:.2f}%")
    print(f"      FalseAlarm: {stats['false_alarm_rate']:.2f}%")
    print(f"      Accuracy  : {stats['accuracy']:.2f}%")
    return stats


# ── 2. DQN ────────────────────────────────────────────────────────────────────

def get_dqn_stats(model_path="results/dqn_model.pth"):
    print("\n[2/3] Evaluating standard DQN...")

    if not os.path.exists(model_path):
        print("      No saved model — training DQN now...")
        from train_dqn import train as train_dqn
        return train_dqn()

    agent = DQNAgent(input_dim=10, output_dim=3)
    ckpt  = torch.load(model_path, map_location=DEVICE)
    agent.online_net.load_state_dict(ckpt["online_net"])
    agent.online_net.eval()

    def dqn_action(obs):
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            return int(agent.online_net(t).argmax(1).item())

    stats = evaluate_policy(dqn_action)
    print(f"      Detection : {stats['detection_rate']:.2f}%")
    print(f"      FalseAlarm: {stats['false_alarm_rate']:.2f}%")
    print(f"      Accuracy  : {stats['accuracy']:.2f}%")
    return stats


# ── 3. DDQN ───────────────────────────────────────────────────────────────────

def get_ddqn_stats(model_path="results/ddqn_model.pth"):
    print("\n[3/3] Evaluating DDQN (Dueling Double DQN)...")

    if not os.path.exists(model_path):
        print("      No saved model — training DDQN now...")
        from train_rl import train as train_ddqn
        return train_ddqn()

    agent = DDQNAgent(input_dim=10, output_dim=3)
    ckpt  = torch.load(model_path, map_location=DEVICE)
    agent.online_net.load_state_dict(ckpt["online_net"])
    agent.online_net.eval()

    def ddqn_action(obs):
        with torch.no_grad():
            t = torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
            return int(agent.online_net(t).argmax(1).item())

    stats = evaluate_policy(ddqn_action)
    print(f"      Detection : {stats['detection_rate']:.2f}%")
    print(f"      FalseAlarm: {stats['false_alarm_rate']:.2f}%")
    print(f"      Accuracy  : {stats['accuracy']:.2f}%")
    return stats


# ── 4. Bar chart ──────────────────────────────────────────────────────────────

def plot_comparison(rb_stats, dqn_stats, ddqn_stats):
    methods     = ["Rule-Based", "DQN", "DDQN (Dueling)"]
    metrics     = ["Detection Rate (%)", "False Alarm Rate (%)", "Accuracy (%)"]
    metric_keys = ["detection_rate", "false_alarm_rate", "accuracy"]

    values = np.array([
        [rb_stats[k]   for k in metric_keys],
        [dqn_stats[k]  for k in metric_keys],
        [ddqn_stats[k] for k in metric_keys],
    ])

    COLORS = ["#6B6B6B", "#3A7DC9", "#2BAE82"]

    plt.rcParams.update({
        "font.family":       "DejaVu Sans",
        "font.size":         11,
        "axes.spines.top":   False,
        "axes.spines.right": False,
        "axes.grid":         True,
        "grid.alpha":        0.35,
        "grid.linewidth":    0.6,
    })

    fig, ax = plt.subplots(figsize=(10, 6))

    x       = np.arange(len(metrics))
    bar_w   = 0.22
    offsets = np.array([-bar_w, 0, bar_w])

    for i, (method, color, offset) in enumerate(
            zip(methods, COLORS, offsets)):
        rects = ax.bar(
            x + offset, values[i], bar_w,
            label=method,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        for rect, val in zip(rects, values[i]):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.8,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=9, color=color,
            )

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_ylim(0, 115)
    ax.set_yticks(range(0, 101, 20))

    # Direction annotations below each metric group
    directions = ["↑ higher is better", "↓ lower is better", "↑ higher is better"]
    for xi, label in zip(x, directions):
        ax.annotate(
            label,
            xy=(xi, -0.11), xycoords=("data", "axes fraction"),
            ha="center", fontsize=8.5, color="#888888",
            annotation_clip=False,
        )

    ax.legend(
        loc="upper right",
        framealpha=0.9,
        fontsize=10,
        edgecolor="#cccccc",
    )

    ax.set_title(
        "Performance Comparison: Rule-Based vs. DQN vs. DDQN\n"
        "Attack Mitigation in 5G-Enabled Intent-Based Network",
        fontsize=12, pad=14,
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])

    png_path = "results/comparison_bargraph.png"
    pdf_path = "results/comparison_bargraph.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path,           bbox_inches="tight")
    print(f"\n  Saved → {png_path}")
    print(f"  Saved → {pdf_path}")
    plt.close(fig)


# ── 5. CSV table ──────────────────────────────────────────────────────────────

def save_table(rb_stats, dqn_stats, ddqn_stats):
    rows = []
    for name, stats in [
        ("Rule-Based",            rb_stats),
        ("DQN",                   dqn_stats),
        ("DDQN (Dueling Double)", ddqn_stats),
    ]:
        total_atk = stats["attacks_caught"] + stats["attacks_missed"]
        rows.append({
            "Algorithm":            name,
            "Detection Rate (%)":   round(stats["detection_rate"],   2),
            "False Alarm Rate (%)": round(stats["false_alarm_rate"], 2),
            "Accuracy (%)":         round(stats["accuracy"],         2),
            "Attacks Caught":       stats["attacks_caught"],
            "Total Attacks":        total_atk,
            "False Alarms":         stats["false_alarms"],
        })

    df   = pd.DataFrame(rows)
    path = "results/comparison_table.csv"
    df.to_csv(path, index=False)

    print("\n── Comparison Table ─────────────────────────────────────────────")
    print(df.to_string(index=False))
    print("─────────────────────────────────────────────────────────────────")
    print(f"  Saved → {path}")
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":

    # Verify test set exists
    if not os.path.exists(TEST_PATH):
        raise FileNotFoundError(
            f"{TEST_PATH} not found.\n"
            "Run any training script first (train_rl.py or DQN_train.py) "
            "to create the fixed train/test split."
        )

    # Verify test set is consistent
    test_df = pd.read_csv(TEST_PATH)
    print(f"Test set: {len(test_df)} rows | "
          f"attacks={test_df['is_attacker'].sum()} | "
          f"normal={(test_df['is_attacker']==0).sum()}")

    rb_stats   = get_rule_based_stats()
    dqn_stats  = get_dqn_stats()
    ddqn_stats = get_ddqn_stats()

    plot_comparison(rb_stats, dqn_stats, ddqn_stats)
    save_table(rb_stats, dqn_stats, ddqn_stats)

    print("\nDone — results/comparison_bargraph.png ready for your paper.")