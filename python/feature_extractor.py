# feature_extractor.py
# PURPOSE: Converts raw JSON data from OMNeT++ into

# Raw JSON:  {"time":10.01, "node":"attacker[0]",
#             "pkt_rate":1000.0, "pkt_size":1500, ...}
# Output:    numpy array [0.99, 1.0, 0.99, 1.0, 0.1]
#            (normalized 0-1 values for neural network)

import numpy as np
from collections import defaultdict

class FeatureExtractor:
    def __init__(self):
        self.MAX_PKT_RATE = 1500.0
        # Attacker sends ~1000 pkt/s, so 1500 gives headroom
        # Used to normalize: actual_rate / MAX_PKT_RATE = 0 to 1

        self.MAX_PKT_SIZE = 1500.0
        # ↑ Maximum Ethernet packet size in bytes
        # Normal: 512B → 0.34, Attacker: 1500B → 1.0

        self.MIN_INTERVAL = 0.001
        self.MAX_INTERVAL = 0.1

        self.node_history = defaultdict(list)

    def extract(self, raw_data):
        # ↑ Main function - converts one raw JSON dict to feature vector
        # raw_data = one dictionary from udp_receiver
        # Returns numpy array of 6 normalized features
        try:
            pkt_rate = float(raw_data.get("pkt_rate", 0))
            pkt_size = float(raw_data.get("pkt_size", 512))
            interval = float(raw_data.get("interval", 0.1))
            port = int(raw_data.get("port", 5000))
            sim_time = float(raw_data.get("time",0))
            is_attacker = int(raw_data.get("is_attacker", 0))

            #Feature 1: Packet Rate (normalized)
            pkt_rate_norm = min(pkt_rate / self.MAX_PKT_RATE, 1.0)
            # Normal UE:  ~10/1500  = 0.007 (very low)
            # Attacker:   1000/1500 = 0.667 (high)

            #  Feature 2: Packet Size (normalized) 
            pkt_size_norm = min(pkt_size / self.MAX_PKT_SIZE, 1.0)
            # ↑ Normal: 512/1500  = 0.341
            # Attacker: 1500/1500 = 1.0

            # Feature 3: Send interval (inverted, normalized)
            interval_norm = 1.0 -min (
                (interval - self.MIN_INTERVAL)/
                (self.MAX_INTERVAL - self.MIN_INTERVAL), 1.0
            )
            # Normal interval=0.1s  → norm = 1.0 - 1.0 = 0.0  (slow sender)
            # Attacker interval=0.001s → norm = 1.0 - 0.0 = 1.0 (fast sender)

            #  Feature 4: Destination Port
            port_flag = 1.0 if port == 4000 else 0.0
            # Port 4000 = attacker target port
            # Port 5000 = normal traffic port

            # Feature 5: simulation time 
            time_norm = min(sim_time / 100.0 , 1.0)

            # Feature 6: Packet Data Trend 
            pkt_delta = float(raw_data.get("pkt_delta", 0))
            delta_norm = min(pkt_delta / 20.0, 1.0)

            state = np.array([
                pkt_rate_norm,
                pkt_size_norm,
                interval_norm,
                port_flag,
                time_norm,
                delta_norm,
            ], dtype=np.float32)

            #store in history for trend analysis
            node = raw_data.get("node", "unknown")
            self.node_history[node].append({
                "time": sim_time,
                "pkt_rate": pkt_rate,
                "state": state,
                "is_attacker": is_attacker
            })

            if len(self.node_history[node]) > 100:
                self.node_history[node].pop(0)

            return state, is_attacker, node
            # ↑ Returns tuple:
            # state       = numpy array for RL agent
            # is_attacker = ground truth label (0 or 1)
            # node        = node name string for logging

        except Exception as e:
            return np.zeros(6, dtype=np.float32), 0, "unknown"
        
    def extract_batch(self, raw_data_list):
        results = []
        for raw in raw_data_list:
            state, label, node = self.extract(raw)
            results.append({
                "state": state,
                "label": label,
                "node": node,
                "time": raw.get("time", 0),
                "raw": raw
            })

        return results
    # Each item has state vector + metadata
        
                




