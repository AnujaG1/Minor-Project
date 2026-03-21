"""
Two threads:
  Thread 1 (UDPListenerThread) — receives packets from SocketStreamer → raw_queue
  Thread 2 (_process_loop)     — drains queue, runs inference, updates state

PipelineState is thread-safe shared memory that Flask reads every 500ms.
"""

import json, time, torch, threading, traceback
import numpy as np
from collections import deque

from feature_extractor import FeatureExtractor, UDPReceiver
from train_rl import DQNNetwork
from intent_engine import IntentEngine, Intent

class DQNInference:
    """Loads saved model, greedy inference only — no training."""

    def __init__(self, model_path="results/dqn_model.pth", input_dim=5):
        self.net = DQNNetwork(input_dim)
        checkpoint = torch.load(model_path, map_location="cpu")

        # Model saved by DQNAgent.save() as full checkpoint dict
        if "online_net" in checkpoint:
            self.net.load_state_dict(checkpoint["online_net"])
        else:
            self.net.load_state_dict(checkpoint)

        self.net.eval()
        print(f"[DQN] Loaded from {model_path}")

    def predict(self, features):
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            return int(self.net(x).argmax(dim=1).item())

class UDPListenerThread(threading.Thread):
    """Wraps UDP packets => pushes packets onto shared deque."""

    def __init__(self, raw_queue: deque, host="127.0.0.1", port=9999):
        super().__init__(daemon=True)
        self.receiver  = UDPReceiver(host=host, port=port)
        self.raw_queue = raw_queue
        self._stop_evt = threading.Event()

    def run(self):
        self.receiver.start()
        print("[UDP] Listening for OMNeT++ packets on port 9999...")
        while not self._stop_evt.is_set():
            for pkt in self.receiver.get_data():
                self.raw_queue.append(pkt)
            time.sleep(0.005)

    def stop(self):
        self._stop_evt.set()
        self.receiver.stop()

class PipelineState:
 # Thread-safe shared state between pipeline and Flask.

    def __init__(self):
        self._lock           = threading.Lock()
        self.recent_intents  = deque(maxlen=100)
        self.node_timeseries = {}          # node → deque[(sim_time, pkt_rate)]
        self.alert_log       = deque(maxlen=50)
        self.counters        = {"total": 0, "attacks": 0, "normal": 0}
        self.sim_running     = False

    def update(self, intent: Intent, raw: dict):
        node     = intent.node
        sim_time = intent.sim_time
        pkt_rate = float(raw.get("pkt_rate", 0))

        with self._lock:
            if node not in self.node_timeseries:
                self.node_timeseries[node] = deque(maxlen=200)
            self.node_timeseries[node].append((sim_time, pkt_rate))

            self.recent_intents.append(intent.to_dict())

            if intent.action_type.value in ("BLOCK", "MONITOR"):
                self.alert_log.append(intent.to_dict())

            self.counters["total"] += 1
            if intent.action_type.value == "BLOCK":
                self.counters["attacks"] += 1
            else:
                self.counters["normal"] += 1

            self.sim_running = True

    def mark_sim_ended(self):
        with self._lock:
            self.sim_running = False

    def snapshot(self):
        with self._lock:
            return {
                "counters"       : dict(self.counters),
                "recent_intents" : list(self.recent_intents)[-20:],
                "alert_log"      : list(self.alert_log)[-10:],
                "node_timeseries": {
                    node: list(ts)
                    for node, ts in self.node_timeseries.items()
                },
                "sim_running"    : self.sim_running,
            }

    def reset(self):
        with self._lock:
            self.recent_intents.clear()
            self.node_timeseries.clear()
            self.alert_log.clear()
            self.counters = {"total": 0, "attacks": 0, "normal": 0}
            self.sim_running = False

class RealtimePipeline:
 # Wires everything together: UDP → FeatureExtractor → DQN → IntentEngine → PipelineState
    
    def __init__(self, model_path="results/dqn_model.pth"):
        self.raw_queue = deque(maxlen=500)
        self.extractor = FeatureExtractor()
        self.dqn       = DQNInference(model_path, input_dim=5)
        self.engine    = IntentEngine()
        self.state     = PipelineState()
        self._listener = UDPListenerThread(self.raw_queue)
        self._running  = False
        self._thread   = None

    def start(self):
        self._listener.start()
        self._running = True
        self._thread  = threading.Thread(target=self._process_loop, daemon=True)
        self._thread.start()
        print("[Pipeline] Started — waiting for OMNeT++ simulation...")

    def stop(self):
        self._running = False
        self._listener.stop()
        print("[Pipeline] Stopped.")

    def _process_loop(self):
        while self._running:
            if not self.raw_queue:
                time.sleep(0.005)
                continue

            raw = self.raw_queue.popleft()

            if raw.get("type") == "SIM_END":
                print("[Pipeline] Simulation ended.")
                self.state.mark_sim_ended()
                continue

            try:
                # Step 1: extract 5 behavioural features
                features, _, node = self.extractor.extract(raw)

                # Step 2: DQN predicts independently from ground truth
                action = self.dqn.predict(features)

                # Step 3: Intent engine maps action → structured policy
                intent = self.engine.process(
                    dqn_action = action,
                    node       = node,
                    features   = features.tolist(),
                    sim_time   = float(raw.get("time", 0.0)),
                    raw_data   = raw,
                )

                # Step 4: update Flask-readable state
                self.state.update(intent, raw)
                if action == 1:
                    print(intent)

            except Exception as e:
                print(f"[Pipeline] Error: {e}")
                traceback.print_exc()

if __name__ == "__main__":
    import socket as _socket

    pipeline = RealtimePipeline()
    pipeline.start()
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    print("[Test] Sending fake packets...")

    for i in range(20):
        t = round(5.0 + i * 0.2, 2)
        pkt = {"time": t, "node": "attacker[0]" if i % 3 == 0 else f"ue[{i%3}]",
               "pkt_rate": 1000.0 if i % 3 == 0 else 10.0,
               "pkt_size": 1500 if i % 3 == 0 else 512,
               "interval": 0.001 if i % 3 == 0 else 0.1,
               "burst_ratio": 8.5 if i % 3 == 0 else 1.1,
               "cell_zscore": 4.8 if i % 3 == 0 else 0.2}
        sock.sendto(json.dumps(pkt).encode(), ("127.0.0.1", 9999))
        time.sleep(0.05)
    time.sleep(0.5)
    snap = pipeline.state.snapshot()
    print(f"Counters: {snap['counters']}")
    print(f"Nodes:    {list(snap['node_timeseries'].keys())}")
    pipeline.stop()
 