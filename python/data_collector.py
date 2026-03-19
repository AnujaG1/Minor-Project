"""
data_collector.py
=================
Run this ONCE alongside an OMNeT++ simulation to collect training data.
Saves everything to results/training_data.csv.

Usage:
  Terminal 1: python3 data_collector.py
  Terminal 2: run OMNeT++ simulation (any config)

CSV columns (updated for 7 behavioural features):
  sim_time, node, node_type,
  pkt_rate, pkt_size, interval, dest_port, is_attacker,
  jitter, burst_ratio, size_std, flow_duration, cell_zscore,
  f1, f2, f3, f4, f5, f6, f7
"""

import csv
import os
import time
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

from feature_extractor import UDPReceiver, FeatureExtractor


def collect_training_data(
        output_file: str = "results/training_data.csv",
        duration:    int = 110,
        host:        str = "127.0.0.1",
        port:        int = 9999,
):
    os.makedirs("results", exist_ok=True)

    receiver  = UDPReceiver(host=host, port=port)
    extractor = FeatureExtractor()

    with open(output_file, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)

        # Header — raw values + pre-computed behavioural values + features
        writer.writerow([
            "sim_time", "node", "node_type",
            # raw observed
            "pkt_rate", "pkt_size", "interval", "dest_port", "is_attacker",
            # behavioural (computed in C++)
            "jitter", "burst_ratio", "size_std", "flow_duration", "cell_zscore",
            # normalised features fed to DQN
            "f1_pkt_rate", "f2_pkt_size", "f3_interval",
            "f4_jitter", "f5_burst", "f6_size_unif", "f7_zscore",
        ])

        print(f"Saving to {output_file}")
        print("Start OMNeT++ simulation now!")
        print("-" * 50)

        receiver.start()
        start_time   = time.time()
        rows_written = 0

        while time.time() - start_time < duration:
            batch = receiver.get_data()

            for raw in batch:
                features, label, node = extractor.extract(raw)

                writer.writerow([
                    raw.get("time",          0),
                    raw.get("node",          ""),
                    raw.get("type",          ""),
                    # raw observed
                    raw.get("pkt_rate",      0),
                    raw.get("pkt_size",      0),
                    raw.get("interval",      0),
                    raw.get("dest_port",     0),
                    label,
                    # behavioural
                    raw.get("jitter",        0),
                    raw.get("burst_ratio",   1),
                    raw.get("size_std",      0),
                    raw.get("flow_duration", 0),
                    raw.get("cell_zscore",   0),
                    # 7 normalised features
                    features[0], features[1], features[2],
                    features[3], features[4], features[5], features[6],
                ])
                rows_written += 1

                if raw.get("pkt_rate", 0) > 100:
                    print(f"t={raw['time']:.2f}s | "
                          f"{raw['node']:<15} | "
                          f"ATTACK rate={raw['pkt_rate']:.0f} "
                          f"jitter={raw.get('jitter',0):.4f} "
                          f"burst={raw.get('burst_ratio',1):.2f}")

            csvfile.flush()
            time.sleep(0.01)

            if receiver.get_stats()["sim_ended"]:
                print("Simulation ended — stopping collection")
                break

        receiver.stop()

    print(f"\nDone. Rows saved: {rows_written} → {output_file}")
    return output_file


if __name__ == "__main__":
    collect_training_data()