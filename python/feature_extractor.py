"""
feature_extractor.py
====================
Two classes in one file:
  UDPReceiver     — listens on UDP port 9999 for OMNeT++ JSON packets
  FeatureExtractor — converts raw JSON → 7 normalised features

7 features (all behavioural — no port/sim_time cheating):
  f1  pkt_rate      rate vs max                    high = suspicious
  f2  pkt_size      size vs max                    high = suspicious
  f3  interval      inverted send interval         high = fast = suspicious
  f4  jitter        inverted inter-arrival jitter  high = mechanical = suspicious
  f5  burst_ratio   rate vs own rolling avg        high = spike = suspicious
  f6  size_std      inverted size std dev           high = uniform = suspicious
  f7  cell_zscore   rate vs all UEs this tick       high = outlier = suspicious

f4, f5, f6, f7 are computed in C++ by SocketStreamer and arrive
pre-computed in the JSON. Python just normalises them to [0,1].
"""

import numpy as np
from collections import defaultdict
import socket
import json
import threading
import queue
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
# FeatureExtractor
# ─────────────────────────────────────────────────────────────

class FeatureExtractor:
    """
    Converts one raw JSON dict from SocketStreamer into a
    numpy array of 7 normalised features.

    All behavioural features (jitter, burst_ratio, size_std,
    cell_zscore) are computed in C++ and arrive pre-computed.
    This class only normalises them to [0, 1].
    """

    MAX_PKT_RATE  = 1500.0   # pkt/s
    MAX_PKT_SIZE  = 1500.0   # bytes
    MIN_INTERVAL  = 0.001    # s  (fastest attacker)
    MAX_INTERVAL  = 0.1      # s  (slowest normal UE)
    MAX_JITTER    = 0.05     # s
    MAX_BURST     = 10.0     # ratio (10x own avg = saturated)
    MAX_SIZE_STD  = 500.0    # bytes
    MAX_ZSCORE    = 5.0      # std devs

    def __init__(self):
        self.node_history = defaultdict(list)

    def extract(self, raw_data: dict):
        """
        Convert one raw JSON dict to a 7-element feature vector.

        Parameters
        ----------
        raw_data : dict — one parsed JSON packet from SocketStreamer

        Returns
        -------
        (features, is_attacker, node_name)
          features    : np.float32 array shape (7,)
          is_attacker : int 0 or 1  (ground truth label from OMNeT++)
          node_name   : str e.g. "attacker[0]", "ue[1]"
        """
        try:
            pkt_rate    = float(raw_data.get("pkt_rate",    0))
            pkt_size    = float(raw_data.get("pkt_size",    512))
            interval    = float(raw_data.get("interval",    0.1))
            sim_time    = float(raw_data.get("time",        0.0))
            is_attacker = int(  raw_data.get("is_attacker", 0))
            node        =       raw_data.get("node", "unknown")

            # Behavioural fields — computed by SocketStreamer C++
            jitter      = float(raw_data.get("jitter",      0.0))
            burst_ratio = float(raw_data.get("burst_ratio", 1.0))
            size_std    = float(raw_data.get("size_std",    0.0))
            cell_zscore = float(raw_data.get("cell_zscore", 0.0))

            # f1: packet rate
            # Normal ~10 pkt/s → 0.007 | Attacker ~1000 pkt/s → 0.667
            f1 = min(pkt_rate / self.MAX_PKT_RATE, 1.0)

            # f2: packet size
            # Normal 512B → 0.341 | Attacker 1500B → 1.0
            f2 = min(pkt_size / self.MAX_PKT_SIZE, 1.0)

            # f3: send interval (inverted)
            # Normal 0.1s → 0.0 (slow) | Attacker 0.001s → 1.0 (fast)
            f3 = 1.0 - min(
                (interval - self.MIN_INTERVAL) /
                (self.MAX_INTERVAL - self.MIN_INTERVAL), 1.0
            )
            f3 = max(f3, 0.0)

            # f4: inter-arrival jitter (inverted)
            # Flood tools: perfectly fixed interval → jitter ≈ 0 → f4 = 1.0
            # Real traffic: OS scheduling variation  → jitter > 0 → f4 < 1.0
            # Does NOT use port number — purely behavioural
            f4 = 1.0 - min(jitter / self.MAX_JITTER, 1.0)

            # f5: burst ratio vs own rolling average
            # Normal: ratio ≈ 1.0 → f5 = 0.1
            # Attack ramp-up: ratio >> 1 → f5 → 1.0
            # Self-normalising — works at any simulation time
            f5 = min(burst_ratio / self.MAX_BURST, 1.0)

            # f6: size uniformity (inverted std dev)
            # Flood: all same size → std=0  → f6 = 1.0
            # Real:  varying payload → std>0 → f6 < 1.0
            f6 = 1.0 - min(size_std / self.MAX_SIZE_STD, 1.0)

            # f7: cell z-score (outlier vs peers)
            # How many std devs above the cell-wide mean rate?
            # Catches attackers even at moderate absolute rates
            f7 = min(max(cell_zscore / self.MAX_ZSCORE, 0.0), 1.0)

            features = np.array(
                [f1, f2, f3, f4, f5, f6, f7],
                dtype=np.float32
            )

            # Store history for offline analysis
            self.node_history[node].append({
                "time"       : sim_time,
                "pkt_rate"   : pkt_rate,
                "interval"   : interval,
                "pkt_size"   : pkt_size,
                "is_attacker": is_attacker,
            })
            if len(self.node_history[node]) > 100:
                self.node_history[node].pop(0)

            return features, is_attacker, node

        except Exception as e:
            logger.warning(f"[FeatureExtractor] Error: {e}")
            return np.zeros(7, dtype=np.float32), 0, "unknown"

    def extract_batch(self, raw_data_list: list) -> list:
        results = []
        for raw in raw_data_list:
            features, label, node = self.extract(raw)
            results.append({
                "state": features,
                "label": label,
                "node" : node,
                "time" : raw.get("time", 0),
                "raw"  : raw,
            })
        return results


# ─────────────────────────────────────────────────────────────
# UDPReceiver
# ─────────────────────────────────────────────────────────────

class UDPReceiver:
    """
    Listens on UDP port 9999 for JSON packets from SocketStreamer.
    Runs a background thread. Call get_data() to drain the queue.
    """

    def __init__(self, host="127.0.0.1", port=9999, buffer_size=10000):
        self.host        = host
        self.port        = port
        self.buffer_size = buffer_size
        self.data_queue  = queue.Queue(maxsize=buffer_size)
        self.running     = False
        self.thread      = None
        self.stats = {
            "packets_received": 0,
            "packets_dropped" : 0,
            "parse_errors"    : 0,
            "sim_ended"       : False,
        }

    def start(self):
        self.running = True
        self.thread  = threading.Thread(target=self._listen_loop, daemon=True)
        self.thread.start()
        logger.info(f"UDP Receiver started on {self.host}:{self.port}")

    def stop(self):
        self.running = False
        logger.info("UDP Receiver stopped")

    def _listen_loop(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(1.0)
        logger.info("Waiting for OMNeT++ simulation data...")

        while self.running:
            try:
                raw_data, _ = sock.recvfrom(4096)
                json_str    = raw_data.decode("utf-8")

                if '"type":"SIM_END"' in json_str:
                    logger.info("Simulation ended signal received")
                    self.stats["sim_ended"] = True
                    continue

                parsed = json.loads(json_str)

                try:
                    self.data_queue.put_nowait(parsed)
                    self.stats["packets_received"] += 1
                except queue.Full:
                    self.data_queue.get_nowait()
                    self.data_queue.put_nowait(parsed)
                    self.stats["packets_dropped"] += 1

            except socket.timeout:
                continue
            except json.JSONDecodeError as e:
                self.stats["parse_errors"] += 1
                logger.warning(f"Bad JSON: {e}")
            except Exception as e:
                logger.error(f"Receiver error: {e}")
                break

        sock.close()

    def get_data(self) -> list:
        items = []
        while not self.data_queue.empty():
            try:
                items.append(self.data_queue.get_nowait())
            except queue.Empty:
                break
        return items

    def get_stats(self) -> dict:
        return self.stats.copy()


# ─────────────────────────────────────────────────────────────
# Self-test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fe = FeatureExtractor()

    normal  = {"time":5.3,"node":"ue[0]","type":"ue",
               "pkt_rate":10,"pkt_size":512,"interval":0.098,
               "is_attacker":0,"dest_port":5000,
               "jitter":0.012,"burst_ratio":1.1,
               "size_std":45.0,"flow_duration":5.3,"cell_zscore":0.2}

    attack  = {"time":5.3,"node":"attacker[0]","type":"attacker",
               "pkt_rate":1000,"pkt_size":1500,"interval":0.001,
               "is_attacker":1,"dest_port":4000,
               "jitter":0.0,"burst_ratio":8.5,
               "size_std":0.0,"flow_duration":0.3,"cell_zscore":4.8}

    fn, _, _ = fe.extract(normal)
    fa, _, _ = fe.extract(attack)

    names = ["f1 rate","f2 size","f3 interval","f4 jitter",
             "f5 burst","f6 size_unif","f7 zscore"]
    print(f"{'Feature':<14} {'Normal':>8} {'Attack':>8}")
    print("-" * 32)
    for i, n in enumerate(names):
        print(f"{n:<14} {fn[i]:>8.3f} {fa[i]:>8.3f}")
    print("\nAll attack features should be near 1.0")
    print("All normal features should be near 0.0")