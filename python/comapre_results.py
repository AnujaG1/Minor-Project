"""
compare_results.py
Runs Rule-Based, DQN, and DDQN on the same test split and produces
a publication-quality grouped bar chart comparing all three.

Usage (after you have trained DQN and DDQN):
    python3 compare_results.py

If trained models already exist, it loads them and skips retraining.
If not, it trains from scratch (same 500 episodes).

Output:
    results/comparison_bargraph.png  — high-res figure for paper
    results/comparison_bargraph.pdf  — vector PDF for Springer submission
    results/comparison_table.csv     — raw numbers for your paper table
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.model_selection import train_test_split
from environment import NetworkEnv5G
from rule_based import rule_based_train.py

os.makedirs("results", exist_ok=True)
CSV_PATH = "results/training_data.csv"


# ── 1. Prepare test environment (shared across all three methods) ─────────────

def get_test_env(csv_path=CSV_PATH):
    df = pd.read_csv(csv_path)
    _, test_df = train_test_split(
        df, test_size=0.2, random_state=42, stratify=df["is_attacker"]
    )
    test_path = csv_path.replace(".csv", "_test.csv")
    test_df.to_csv(test_path, index=False)
    return NetworkEnv5G(test_path, max_steps=9999, noise_std=0.0)


# ── 2. Evaluate Rule-Based ───────────────────────────────────────────────────

def get_rule_based_stats(test_env):
    print("\n[1/3] Evaluating Rule-Based baseline...")
    stats = evaluate_rule_based(test_env)
    print(f"      Detection: {stats['detection_rate']:.1f}%  "
          f"FalseAlarm: {stats['false_alarm_rate']:.1f}%  "
          f"Accuracy: {stats['accuracy']:.1f}%")
    return stats


# ── 3. Evaluate DQN ──────────────────────────────────────────────────────────

def get_dqn_stats(test_env, model_path="results/dqn_model.pth"):
    import torch
    from train_dqn import DQNAgent, DQNNetwork

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[2/3] Evaluating standard DQN...")

    if os.path.exists(model_path):
        print(f"      Loading saved model: {model_path}")
        agent = DQNAgent(input_dim=10, output_dim=3)
        ckpt  = torch.load(model_path, map_location=DEVICE)
        agent.online_net.load_state_dict(ckpt["online_net"])
        agent.online_net.eval()

        test_env.reset_stats()
        obs, _ = test_env.reset()
        for _ in range(len(test_env.states)):
            with torch.no_grad():
                action = int(
                    agent.online_net(
                        torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                    ).argmax(1).item()
                )
            obs, _, term, trunc, _ = test_env.step(action)
            if term or trunc:
                break
        stats = test_env.get_stats()
    else:
        print("      No saved model found — training DQN now (500 episodes)...")
        from train_dqn import train as train_dqn
        stats = train_dqn(csv_path=CSV_PATH, model_path=model_path)

    print(f"      Detection: {stats['detection_rate']:.1f}%  "
          f"FalseAlarm: {stats['false_alarm_rate']:.1f}%  "
          f"Accuracy: {stats['accuracy']:.1f}%")
    return stats


# ── 4. Evaluate DDQN ─────────────────────────────────────────────────────────

def get_ddqn_stats(test_env, model_path="results/ddqn_model.pth"):
    import torch
    from train_rl import DDQNNetwork, DDQNAgent

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("\n[3/3] Evaluating DDQN (Dueling Double DQN)...")

    if os.path.exists(model_path):
        print(f"      Loading saved model: {model_path}")
        agent = DDQNAgent(input_dim=10, output_dim=3)
        ckpt  = torch.load(model_path, map_location=DEVICE)
        agent.online_net.load_state_dict(ckpt["online_net"])
        agent.online_net.eval()

        test_env.reset_stats()
        obs, _ = test_env.reset()
        for _ in range(len(test_env.states)):
            with torch.no_grad():
                action = int(
                    agent.online_net(
                        torch.FloatTensor(obs).unsqueeze(0).to(DEVICE)
                    ).argmax(1).item()
                )
            obs, _, term, trunc, _ = test_env.step(action)
            if term or trunc:
                break
        stats = test_env.get_stats()
    else:
        print("      No saved model found — training DDQN now (500 episodes)...")
        from train_rl import train as train_ddqn
        stats = train_ddqn(csv_path=CSV_PATH, model_path=model_path)

    print(f"      Detection: {stats['detection_rate']:.1f}%  "
          f"FalseAlarm: {stats['false_alarm_rate']:.1f}%  "
          f"Accuracy: {stats['accuracy']:.1f}%")
    return stats


# ── 5. Build the bar chart ────────────────────────────────────────────────────

def plot_comparison(rb_stats, dqn_stats, ddqn_stats):
    """
    Grouped bar chart — 3 metrics × 3 algorithms.
    Springer-ready: 300 DPI, clean sans-serif, no chartjunk.
    """

    methods  = ["Rule-Based", "DQN", "DDQN\n(Dueling Double)"]
    metrics  = ["Detection Rate (%)", "False Alarm Rate (%)", "Accuracy (%)"]
    metric_keys = ["detection_rate", "false_alarm_rate", "accuracy"]

    # Gather values — rows = methods, cols = metrics
    values = np.array([
        [rb_stats[k]   for k in metric_keys],
        [dqn_stats[k]  for k in metric_keys],
        [ddqn_stats[k] for k in metric_keys],
    ])

    # ── Style ─────────────────────────────────────────────────────────────────
    COLORS = {
        "Rule-Based": "#6B6B6B",          # neutral gray
        "DQN":        "#3A7DC9",          # blue
        "DDQN":       "#2BAE82",          # teal/green
    }
    color_list = list(COLORS.values())

    plt.rcParams.update({
        "font.family":      "DejaVu Sans",
        "font.size":        11,
        "axes.spines.top":  False,
        "axes.spines.right":False,
        "axes.grid":        True,
        "grid.axis":        "y",
        "grid.alpha":       0.35,
        "grid.linewidth":   0.6,
    })

    fig, ax = plt.subplots(figsize=(9, 5.5))

    n_methods = len(methods)
    n_metrics = len(metrics)
    x         = np.arange(n_metrics)
    bar_w     = 0.22
    offsets   = np.array([-bar_w, 0, bar_w])

    bars_all = []
    for i, (method, color, offset) in enumerate(
            zip(methods, color_list, offsets)):
        vals  = values[i]
        rects = ax.bar(
            x + offset, vals, bar_w,
            label=method.replace("\n", " "),
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        bars_all.append(rects)

        # Value labels on top of each bar
        for rect, val in zip(rects, vals):
            ax.text(
                rect.get_x() + rect.get_width() / 2,
                rect.get_height() + 0.8,
                f"{val:.1f}%",
                ha="center", va="bottom",
                fontsize=9, fontweight="500",
                color=color,
            )

    # ── Axes ──────────────────────────────────────────────────────────────────
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.set_ylabel("Percentage (%)", fontsize=11)
    ax.set_ylim(0, 110)
    ax.set_yticks(range(0, 101, 20))
    ax.yaxis.set_tick_params(labelsize=10)

    # Highlight: lower false alarm is better — annotate the axis
    ax.annotate(
        "↓ lower is better",
        xy=(1, -0.13), xycoords=("data", "axes fraction"),
        ha="center", fontsize=8.5, color="#888888",
        annotation_clip=False,
    )
    ax.annotate(
        "↑ higher is better",
        xy=(0, -0.13), xycoords=("data", "axes fraction"),
        ha="center", fontsize=8.5, color="#888888",
        annotation_clip=False,
    )
    ax.annotate(
        "↑ higher is better",
        xy=(2, -0.13), xycoords=("data", "axes fraction"),
        ha="center", fontsize=8.5, color="#888888",
        annotation_clip=False,
    )

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_patches = [
        mpatches.Patch(color=c, label=m.replace("\n", " "))
        for m, c in zip(methods, color_list)
    ]
    ax.legend(
        handles=legend_patches,
        loc="upper right",
        framealpha=0.9,
        fontsize=10,
        edgecolor="#cccccc",
    )

    ax.set_title(
        "Performance Comparison: Rule-Based vs. DQN vs. DDQN\n"
        "5G-Enabled Intent-Based Network — Attack Mitigation",
        fontsize=12, pad=14,
    )

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    png_path = "results/comparison_bargraph.png"
    pdf_path = "results/comparison_bargraph.pdf"
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path,           bbox_inches="tight")
    print(f"\n  Saved → {png_path}")
    print(f"  Saved → {pdf_path}")
    plt.close(fig)


# ── 6. Save CSV table ─────────────────────────────────────────────────────────

def save_table(rb_stats, dqn_stats, ddqn_stats):
    rows = []
    for name, stats in [
        ("Rule-Based",          rb_stats),
        ("DQN",                 dqn_stats),
        ("DDQN (Dueling Double)", ddqn_stats),
    ]:
        rows.append({
            "Algorithm":          name,
            "Detection Rate (%)": stats["detection_rate"],
            "False Alarm Rate (%)": stats["false_alarm_rate"],
            "Accuracy (%)":       stats["accuracy"],
            "Attacks Caught":     stats["attacks_caught"],
            "Attacks Missed":     stats["attacks_missed"],
            "False Alarms":       stats["false_alarms"],
        })
    df = pd.DataFrame(rows)
    path = "results/comparison_table.csv"
    df.to_csv(path, index=False)
    print(f"  Saved → {path}")
    print("\n" + df.to_string(index=False))
    return df


# ── Main ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    test_env  = get_test_env()

    rb_stats   = get_rule_based_stats(test_env)
    dqn_stats  = get_dqn_stats(test_env)
    ddqn_stats = get_ddqn_stats(test_env)

    plot_comparison(rb_stats, dqn_stats, ddqn_stats)
    save_table(rb_stats, dqn_stats, ddqn_stats)

    print("\n✓ Done — results/comparison_bargraph.png ready for your paper.")