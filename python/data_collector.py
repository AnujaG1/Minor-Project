"""
data_collector.py  —  v4  FINAL
Fixed: uses sim_time (not wall clock) for deduplication.
Fixed: only writes rows where at least one node in the current tick has pkt_rate > 0.
Fixed: writes ALL ticks from SocketStreamer without skipping.
"""
import csv, os, time, argparse, logging
from collections import defaultdict
from feature_extractor import UDPReceiver, FeatureExtractor

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)


def collect(output_file="results/training_data.csv",
            duration=600, host="127.0.0.1", port=9999):

    os.makedirs("results", exist_ok=True)
    receiver  = UDPReceiver(host=host, port=port)
    extractor = FeatureExtractor()

    # Track last sim_time written per node — use SIM time not wall clock
    last_sim_written = defaultdict(float)

    with open(output_file, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "sim_time", "node", "is_attacker",
            "f1_pkt_rate", "f2_mean_rate", "f3_burst_ratio",
            "f4_rate_change", "f5_rate_trend", "f6_flow_duration",
            "f7_activity_ratio", "f8_cell_zscore",
            "f9_consecutive", "f10_peak_rate",
        ])

        logger.info(f"Saving to {output_file}")
        logger.info("Start OMNeT++ simulation now!")

        receiver.start()
        start     = time.time()
        rows      = 0
        sim_ended = False

        # Buffer: collect all raw ticks at the same sim_time,
        # then write them all together as one consistent snapshot
        tick_buffer = defaultdict(dict)  # sim_time -> {node -> raw}

        while True:
            if time.time() - start > duration:
                logger.info("Timeout reached")
                break

            batch = receiver.get_data()

            for raw in batch:
                if raw.get("type") == "SIM_END":
                    sim_ended = True
                    continue

                node     = raw.get("node", "unknown")
                sim_time = float(raw.get("time", 0.0))

                # Ingest into feature extractor history
                extractor.ingest(raw)

                # Buffer this tick
                tick_buffer[sim_time][node] = raw

            # Process complete ticks — a tick is complete when we have
            # received data for ALL nodes (8 nodes: 5 UE + 3 attacker)
            # or when a newer tick has arrived (meaning the old one is done)
            complete_ticks = []
            all_times = sorted(tick_buffer.keys())

            for i, t in enumerate(all_times):
                # Consider a tick complete if there's a newer tick after it
                if i < len(all_times) - 1:
                    complete_ticks.append(t)

            for t in complete_ticks:
                nodes_at_t = tick_buffer.pop(t)

                # Check if ANY node at this tick has nonzero rate
                any_active = any(
                    float(raw.get("pkt_rate", 0)) > 0
                    for raw in nodes_at_t.values()
                )

                for node, raw in nodes_at_t.items():
                    # Skip if we already wrote this node at this sim_time
                    if t <= last_sim_written[node]:
                        continue
                    last_sim_written[node] = t

                    state       = extractor.get_state(node, t)
                    is_attacker = extractor.get_label(node)
                    pkt_rate    = float(raw.get("pkt_rate", 0))

                    writer.writerow([
                        round(t, 3), node, is_attacker,
                        *[round(float(v), 6) for v in state],
                    ])
                    rows += 1

                    if pkt_rate > 0:
                        logger.info(
                            f"t={t:.0f}s | {node:<15} | "
                            f"rate={pkt_rate:.0f} pkt/s | "
                            f"attacker={is_attacker}"
                        )

            f.flush()

            if sim_ended and not receiver.get_data():
                # Flush remaining buffer
                for t in sorted(tick_buffer.keys()):
                    for node, raw in tick_buffer[t].items():
                        if t <= last_sim_written[node]:
                            continue
                        last_sim_written[node] = t
                        state       = extractor.get_state(node, t)
                        is_attacker = extractor.get_label(node)
                        writer.writerow([
                            round(t, 3), node, is_attacker,
                            *[round(float(v), 6) for v in state],
                        ])
                        rows += 1
                logger.info(f"Done. {rows} rows written.")
                break

            time.sleep(0.005)

        receiver.stop()
        s = receiver.get_stats()
        logger.info(f"UDP received: {s['packets_received']} | "
                    f"Rows written: {rows} -> {output_file}")

    return output_file


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--output",   default="results/training_data.csv")
    p.add_argument("--duration", type=int, default=600)
    args = p.parse_args()
    collect(output_file=args.output, duration=args.duration)
