import csv
import os
import time, argparse
import sys

from feature_extractor import UDPReceiver, FeatureExtractor


def collect_training_data(
        output_file: str = "results/training_data.csv",
        duration:    int = 600,
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
            "pkt_rate", "pkt_size", "interval", "dest_port",
            "burst_ratio", "cell_zscore", "is_attacker",
            "f1_pkt_rate", "f2_pkt_size", "f3_interval",
            "f4_burst", "f5_zscore",
        ])

        print(f"Saving to {output_file}")
        print("Start OMNeT++ simulation now!")
        print("-" * 50)

        receiver.start()
        start  = time.time()
        rows_written = 0
        sim_ended = False

        while True:
            if time.time() - start > duration:
                print("[Collector] Timeout reached")
                break
            
            batch = receiver.get_data()
            for raw in batch:
                if raw.get("pkt_rate", 0) == 0:
                    continue 
                features, label, node = extractor.extract(raw)

                writer.writerow([
                    raw.get("time",          0),
                    raw.get("node",          ""),
                    raw.get("type",          ""),
                    raw.get("pkt_rate",      0),
                    raw.get("pkt_size",      0),
                    raw.get("interval",      0),
                    raw.get("dest_port",     0),
                    raw.get("burst_ratio",   1),
                    raw.get("cell_zscore",   0),
                    label,
                    features[0], features[1], features[2],
                    features[3], features[4],
                ])
                rows_written += 1

                if raw.get("pkt_rate", 0) > 100:
                    print(f"t={raw['time']:.2f}s | "
                          f"{raw['node']:<15} | "
                          f"PACKET rate={raw['pkt_rate']:.0f} "
                          f"burst={raw.get('burst_ratio',1):.2f}")

            csvfile.flush()

            if receiver.get_stats()["sim_ended"] and not sim_ended:
                sim_ended = True
                print("[Collector] SIM_END — draining queue...")
                time.sleep(1.0)
 
            if sim_ended and not batch:
                print("[Collector] Done")
                break
            time.sleep(0.005)

        receiver.stop()
        s = receiver.get_stats()
        print(f"\nReceived : {s['packets_received']}")
        print(f"Dropped  : {s['packets_dropped']}")
        print(f"Rows     : {rows_written} → {output_file}")

    return output_file


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output",   default="results/training_data.csv")
    p.add_argument("--duration", type=int, default=600)
    args = p.parse_args()
    collect_training_data(output_file=args.output, duration=args.duration)