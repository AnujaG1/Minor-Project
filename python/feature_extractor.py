# converts raw_data data => normlaised features
# UDPReceiver listens on port 9999.

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

class FeatureExtractor:

    MAX_PKT_RATE = 1500.0
    MAX_PKT_SIZE = 1500.0
    MIN_INTERVAL = 0.001
    MAX_INTERVAL = 2.0
    MAX_BURST    = 20.0
    MAX_ZSCORE   = 5.0

    def __init__(self):
        self.node_history = defaultdict(list)

    def extract(self, raw_data: dict):
        try:
            pkt_rate    = float(raw_data.get("pkt_rate",    0))
            pkt_size    = float(raw_data.get("pkt_size",    512))
            interval    = float(raw_data.get("interval",    0.1))
            burst_ratio = float(raw_data.get("burst_ratio", 1.0))
            cell_zscore = float(raw_data.get("cell_zscore", 0.0))
            sim_time    = float(raw_data.get("time",        0.0))
            node        = raw_data.get("node", "unknown")

            is_attacker = 1 if "attacker" in node.lower() else 0

            f1 = min(pkt_rate / self.MAX_PKT_RATE, 1.0)
            f2 = min(pkt_size / self.MAX_PKT_SIZE, 1.0)
            f3 = min(1.0 / interval, 1000) / 1000
            f4 = min(burst_ratio / self.MAX_BURST, 1.0)
            f5 = min(max(cell_zscore / self.MAX_ZSCORE, 0.0), 1.0)

            features = np.array(
                [f1, f2, f3, f4, f5],
                dtype=np.float32
            )

            # Store history for offline analysis
            self.node_history[node].append({
                "time"       : sim_time,
                "pkt_rate"   : pkt_rate,
                "interval"   : interval,
                "pkt_size"   : pkt_size,
            })
            if len(self.node_history[node]) > 100:
                self.node_history[node].pop(0)

            return features, is_attacker, node

        except Exception as e:
            logger.warning(f"[FeatureExtractor] Error: {e}")
            return np.zeros(5, dtype=np.float32), 0, "unknown"

    def extract_batch(self, raw_data_list: list) -> list:
        results = []
        for raw_data in raw_data_list:
            features, label, node = self.extract(raw_data)
            results.append({
                "state": features,
                "label": label,
                "node" : node,
                "time" : raw_data.get("time", 0),
                "raw_data"  : raw_data,
            })
        return results

class UDPReceiver:
    """
    Listens on UDP port 9999 for JSON packets from SocketStreamer.
    Runs a background thread. Call get_data() to drain the queue.
    """

    def __init__(self, host="127.0.0.1", port=9999, buffer_size=50000):
        self.host        = host
        self.port        = port
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
        self.thread  = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        logger.info(f"UDP Receiver started on {self.host}:{self.port}")

    def stop(self):
        self.running = False

    def _listen(self):
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(1.0)
        logger.info("Waiting for OMNeT++ simulation data...")
        while self.running:
            try:
                data, _ = sock.recvfrom(4096)
                json_str    = data.decode("utf-8")
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
