"""
Usage:
    python merge_data.py                        # auto-finds results/training_data_r*.csv
    python merge_data.py --reps 0 1 2 3         # specific repetitions
    python merge_data.py --no-balance           # keep natural class ratio
    python merge_data.py --min-attack 1000      # require at least N attack rows

What this script does:
  1. Load all repetition CSVs
  2. Validate columns and dtypes
  3. Remove bad rows (NaN, inf, out-of-range features)
  4. Remove duplicate rows (same node + same sim_time across reps)
  5. Report class distribution and feature statistics
  6. Balance classes (downsample majority to match minority)
  7. Shuffle and save to results/training_data.csv
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd

# ── Column definitions ────────────────────────────────────────────────────

FEATURE_COLS = [
    "f1_bytes_sec",
    "f2_interval_std",
    "f3_srv_count",
    "f4_diff_srv_rate",
    "f5_burst_ratio",
    "f6_cell_zscore",
    "f7_size_cv_inv",
    "f8_flow_duration",
    "f9_rate_accel",
]

REQUIRED_COLS = (
    ["sim_time", "node", "pkt_rate", "pkt_size", "interval",
     "dest_port", "burst_ratio", "cell_zscore", "label"]
    + FEATURE_COLS
)


def load_repetitions(rep_files: list) -> pd.DataFrame:
    """Load and tag each repetition CSV."""
    frames = []
    for i, path in enumerate(rep_files):
        if not os.path.exists(path):
            print(f"  [SKIP] {path} not found")
            continue
        df = pd.read_csv(path)
        df["rep"] = i
        frames.append(df)
        n_atk = (df["label"] == 1).sum()
        n_nrm = (df["label"] == 0).sum()
        print(f"  Loaded {path}: {len(df)} rows  "
              f"(attack={n_atk}, normal={n_nrm})")

    if not frames:
        raise ValueError("No CSV files found. Run data_collector.py first.")

    combined = pd.concat(frames, ignore_index=True)
    print(f"\n  Total after merge : {len(combined)} rows")
    return combined


def validate_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Check all required columns exist. Rename is_attacker→label if needed."""
    # Backward compatibility: old collector used is_attacker
    if "is_attacker" in df.columns and "label" not in df.columns:
        df = df.rename(columns={"is_attacker": "label"})
        print("  [Fix] Renamed is_attacker → label")

    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing columns: {missing}\n"
            f"Re-run data_collector.py with the updated feature_extractor.py"
        )

    # Ensure label is integer
    df["label"] = df["label"].astype(int)
    return df



def clean_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove rows with NaN, inf, or out-of-range feature values."""
    before = len(df)

    # Drop NaN in any feature or label column
    df = df.dropna(subset=FEATURE_COLS + ["label"])

    # Drop inf values
    for col in FEATURE_COLS:
        df = df[np.isfinite(df[col])]

    # Features must be in [0, 1] — values outside suggest extractor bug
    out_of_range = pd.Series(False, index=df.index)
    for col in FEATURE_COLS:
        out_of_range |= (df[col] < -0.01) | (df[col] > 1.01)
    n_bad = out_of_range.sum()
    if n_bad > 0:
        print(f"  [Clean] Dropping {n_bad} rows with features outside [0,1]")
        df = df[~out_of_range]

    # Label must be 0 or 1
    df = df[df["label"].isin([0, 1])]

    after = len(df)
    dropped = before - after
    if dropped > 0:
        print(f"  [Clean] Removed {dropped} bad rows  ({before} → {after})")
    else:
        print(f"  [Clean] All {after} rows are valid")

    return df.reset_index(drop=True)

def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove rows where the same node reported the same sim_time across reps.
    This happens when you accidentally collect two CSVs from the same seed.
    """
    before = len(df)
    df = df.drop_duplicates(subset=["sim_time", "node", "rep"], keep="first")
    dropped = before - len(df)
    if dropped > 0:
        print(f"  [Dedup] Removed {dropped} duplicate rows")
    return df.reset_index(drop=True)


def report_distribution(df: pd.DataFrame):
    """Print class and feature statistics."""
    n_atk = (df["label"] == 1).sum()
    n_nrm = (df["label"] == 0).sum()
    total = len(df)

    print(f"\n  Class distribution:")
    print(f"    Normal  (0) : {n_nrm:>6}  ({n_nrm/total*100:.1f}%)")
    print(f"    Attack  (1) : {n_atk:>6}  ({n_atk/total*100:.1f}%)")
    print(f"    Total       : {total:>6}")

    print(f"\n  Feature means by class:")
    print(f"  {'Feature':<22} {'Normal':>8} {'Attack':>8} {'Overlap?':>10}")
    print(f"  {'-'*52}")

    nrm = df[df["label"] == 0]
    atk = df[df["label"] == 1]

    for col in FEATURE_COLS:
        mn = nrm[col].mean()
        ma = atk[col].mean()
        # Simple overlap heuristic: if means within 0.15 of each other
        # the feature has non-trivial overlap (good for learning)
        diff = abs(ma - mn)
        if diff < 0.15:
            overlap = "HIGH (good)"
        elif diff < 0.40:
            overlap = "medium"
        else:
            overlap = "LOW — easy"
        print(f"  {col:<22} {mn:>8.3f} {ma:>8.3f} {overlap:>12}")

    print()

    # Warn if any feature has near-zero variance for one class
    for col in FEATURE_COLS:
        for cls, label_df in [("normal", nrm), ("attack", atk)]:
            std = label_df[col].std()
            if std < 0.005:
                print(f"  [WARN] {col} has std={std:.4f} for {cls} "
                      f"— almost no variance, may be trivially separable")


def balance_classes(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Downsample the majority class to match the minority class size.
    This prevents the DQN from learning the trivial "always predict normal"
    policy which achieves ~70% accuracy on imbalanced data.
    """
    n_atk = (df["label"] == 1).sum()
    n_nrm = (df["label"] == 0).sum()
    min_n = min(n_atk, n_nrm)

    balanced = pd.concat([
        df[df["label"] == 1].sample(min_n, random_state=seed),
        df[df["label"] == 0].sample(min_n, random_state=seed),
    ]).sample(frac=1, random_state=seed).reset_index(drop=True)

    print(f"  [Balance] {n_nrm} normal + {n_atk} attack  "
          f"→  {min_n} + {min_n} = {len(balanced)} rows")
    return balanced

def main():
    parser = argparse.ArgumentParser(
        description="Merge, clean, and balance training data repetitions"
    )
    parser.add_argument(
        "--reps", nargs="+", type=int, default=None,
        help="Repetition indices to merge (default: auto-detect r0,r1,...)"
    )
    parser.add_argument(
        "--input-dir", default="results",
        help="Directory containing training_data_rN.csv files"
    )
    parser.add_argument(
        "--output", default="results/training_data.csv",
        help="Output CSV path"
    )
    parser.add_argument(
        "--no-balance", action="store_true",
        help="Skip class balancing (keep natural ratio)"
    )
    parser.add_argument(
        "--min-attack", type=int, default=500,
        help="Minimum attack rows required before balancing (default: 500)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for sampling"
    )
    args = parser.parse_args()

    os.makedirs(args.input_dir, exist_ok=True)

    # ── Find input files ──────────────────────────────────────
    if args.reps is not None:
        rep_files = [
            os.path.join(args.input_dir, f"training_data_r{r}.csv")
            for r in args.reps
        ]
    else:
        rep_files = sorted(
            glob.glob(os.path.join(args.input_dir, "training_data_r*.csv"))
        )
        if not rep_files:
            # Fallback: look for single training_data.csv from old workflow
            fallback = os.path.join(args.input_dir, "training_data_raw.csv")
            if os.path.exists(fallback):
                rep_files = [fallback]
            else:
                print("[Error] No training_data_rN.csv files found in "
                      f"{args.input_dir}/")
                print("  Run: python data_collector.py "
                      "--output results/training_data_r0.csv")
                return

    print(f"\n{'='*60}")
    print(f" merge_data.py — merging {len(rep_files)} repetition(s)")
    print(f"{'='*60}\n")

    # ── Pipeline ──────────────────────────────────────────────
    print("[1/6] Loading repetitions...")
    df = load_repetitions(rep_files)

    print("\n[2/6] Validating columns...")
    df = validate_columns(df)

    print("\n[3/6] Cleaning bad rows...")
    df = clean_rows(df)

    print("\n[4/6] Removing duplicates...")
    df = remove_duplicates(df)

    print("\n[5/6] Distribution report:")
    report_distribution(df)

    # ── Check minimum attack rows ─────────────────────────────
    n_atk = (df["label"] == 1).sum()
    if n_atk < args.min_attack:
        print(f"[WARN] Only {n_atk} attack rows (need {args.min_attack}).")
        print("  Suggestions:")
        print("  - Check omnetpp.ini: earliest burst startTime should be")
        print("    uniform(5s,15s) so attackers appear before t=15s")
        print("  - Run more repetitions (--reps 0 1 2 3 4)")
        print("  - Increase sim-time-limit in omnetpp.ini")
        if n_atk < 100:
            print("  [ABORT] Too few attack rows to train meaningfully.")
            return

    # ── Balance ───────────────────────────────────────────────
    if not args.no_balance:
        print("[6/6] Balancing classes...")
        df = balance_classes(df, seed=args.seed)
    else:
        print("[6/6] Skipping balance (--no-balance set)")
        df = df.sample(frac=1, random_state=args.seed).reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────
    df.to_csv(args.output, index=False)
    print(f"\n{'='*60}")
    print(f" Saved {len(df)} rows → {args.output}")
    print(f" Ready for: python train_rl.py")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()