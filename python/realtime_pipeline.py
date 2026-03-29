"""
realtime_pipeline.py  —  v2
Live pipeline: OMNeT++ UDP → FeatureExtractor → DDQN → IntentEngine → Flask

Key changes from v1:
  - Uses extractor.ingest(raw) + extractor.get_state(node, time) instead of
    the old extractor.extract(raw) which read pre-cooked OMNeT++ fields.
  - input_dim=10 to match the trained model.
  - Processes events per-node on a 1-second cadence matching training.
  - IBN feedback: action taken is sent back as a network state change
    so the next state reflects the mitigation effect.
"""
import json
import time
import torch
import threading
import numpy as np
import socket as _socket
from collections import deque, defaultdict

from feature_extractor import FeatureExtractor, UDPReceiver
from train_rl          import DDQNNetwork
from intent_engine     import IntentEngine, Intent


class DDQNInference:
    """Load saved DDQN model, greedy inference only."""

    def __init__(self, model_path="results/ddqn_model.pth", input_dim=10,
                 output_dim=3):
        self.net = DDQNNetwork(input_dim, output_dim)
        ck = torch.load(model_path, map_location="cpu", weights_only=True)
        self.net.load_state_dict(
            ck["online_net"] if "online_net" in ck else ck
        )
        self.net.eval()
        saved_dims = (ck.get("input_dim"), ck.get("output_dim"))
        print(f"[DDQN] Loaded {model_path} "
              f"(dims saved={saved_dims}, using={input_dim},{output_dim})")

    def predict(self, features: np.ndarray) -> int:
        with torch.no_grad():
            x = torch.FloatTensor(features).unsqueeze(0)
            return int(self.net(x).argmax(1).item())

    def predict_with_confidence(self, features: np.ndarray) -> tuple[int, float]:
        """Returns (action, confidence) where confidence = softmax max."""
        with torch.no_grad():
            x      = torch.FloatTensor(features).unsqueeze(0)
            q_vals = self.net(x).squeeze(0)
            probs  = torch.softmax(q_vals, dim=0)
            action = int(probs.argmax().item())
            conf   = float(probs.max().item())
        return action, conf


class UDPListenerThread(threading.Thread):
    """Receives UDP packet events → pushes onto raw_queue."""

    def __init__(self, raw_queue, host="127.0.0.1", port=9999):
        super().__init__(daemon=True)
        self.receiver  = UDPReceiver(host=host, port=port)
        self.raw_queue = raw_queue
        self._stop     = threading.Event()

    def run(self):
        self.receiver.start()
        print("[UDP] Listening on port 9999...")
        while not self._stop.is_set():
            for pkt in self.receiver.get_data():
                self.raw_queue.append(pkt)
            time.sleep(0.005)

    def stop(self):
        self._stop.set()
        self.receiver.stop()


class PipelineState:
    """Thread-safe shared state between pipeline and Flask dashboard."""

    def __init__(self):
        self._lock           = threading.Lock()
        self.recent_intents  = deque(maxlen=200)
        self.node_timeseries = {}
        self.alert_log       = deque(maxlen=100)
        self.counters        = {"total": 0, "blocked": 0,
                                "rate_limited": 0, "allowed": 0}
        self.sim_running     = False

    def update(self, intent: Intent, features: np.ndarray, action: int):
        with self._lock:
            node = intent.node
            if node not in self.node_timeseries:
                self.node_timeseries[node] = deque(maxlen=300)

            # Store (time, pkt_rate_unnormalised, action) for dashboard
            pkt_rate_raw = float(features[0]) * 2000.0
            self.node_timeseries[node].append(
                (intent.sim_time, pkt_rate_raw, action)
            )
            self.recent_intents.append(intent.to_dict())

            if intent.action_type.value in ("BLOCK", "THROTTLE"):
                self.alert_log.append(intent.to_dict())

            self.counters["total"] += 1
            if action == 2:
                self.counters["blocked"] += 1
            elif action == 1:
                self.counters["rate_limited"] += 1
            else:
                self.counters["allowed"] += 1

            self.sim_running = True

    def mark_ended(self):
        with self._lock:
            self.sim_running = False

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "counters"       : dict(self.counters),
                "recent_intents" : list(self.recent_intents)[-20:],
                "alert_log"      : list(self.alert_log)[-10:],
                "node_timeseries": {
                    n: list(ts)
                    for n, ts in self.node_timeseries.items()
                },
                "sim_running": self.sim_running,
            }

    def reset(self):
        with self._lock:
            self.recent_intents.clear()
            self.node_timeseries.clear()
            self.alert_log.clear()
            self.counters    = {"total": 0, "blocked": 0,
                                "rate_limited": 0, "allowed": 0}
            self.sim_running = False


class RealtimePipeline:
    """
    Wires everything together:
      UDP events → FeatureExtractor.ingest()
                → per-second: get_state() → DDQN → IntentEngine → State

    The 1-second cadence matches the training data collection cadence,
    ensuring the state vector the agent sees at inference is the same
    distribution as what it was trained on.
    """

    STEP_INTERVAL = 1.0   # seconds — must match data_collector window_sec

    def __init__(self, model_path="results/ddqn_model.pth"):
        self.raw_queue    = deque(maxlen=5000)
        self.extractor    = FeatureExtractor(window_sec=self.STEP_INTERVAL)
        self.dqn          = DDQNInference(model_path, input_dim=10, output_dim=3)
        self.engine       = IntentEngine()
        self.state        = PipelineState()
        self._listener    = UDPListenerThread(self.raw_queue)
        self._running     = False
        self._last_step   = time.time()

        # IBN feedback: track which nodes are currently rate-limited or blocked
        # so we can reduce their contribution to cell_zscore accordingly
        self._mitigated: dict[str, int] = {}   # node → action applied

        # Enforcement socket — sends policy commands back to OMNeT++
        self._cmd_sock   = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
        self._omnet_host = "127.0.0.1"
        self._omnet_cmd_port = 9998          # must match SocketStreamer.ned cmdPort
        self._applied_policies: dict[str, int] = {}   # node → last action sent


    def start(self):
        self._listener.start()
        self._running = True
        threading.Thread(target=self._ingest_loop, daemon=True).start()
        threading.Thread(target=self._decision_loop, daemon=True).start()
        print("[Pipeline] Started — waiting for simulation packets...")

    def stop(self):
        self._running = False
        self._listener.stop()

    # ── Thread 1: ingest every packet as it arrives ───────────────────────────
    def _ingest_loop(self):
        while self._running:
            if not self.raw_queue:
                time.sleep(0.002)
                continue
            raw = self.raw_queue.popleft()
            if raw.get("type") == "SIM_END":
                self.state.mark_ended()
                print("[Pipeline] Simulation ended.")
                continue
            try:
                self.extractor.ingest(raw)
            except Exception as e:
                print(f"[Ingest error] {e}")

    # ── Thread 2: make decisions every STEP_INTERVAL seconds ─────────────────
    def _decision_loop(self):
        while self._running:
            now = time.time()
            if now - self._last_step < self.STEP_INTERVAL:
                time.sleep(0.05)
                continue
            self._last_step = now

            # Get the latest sim_time from any node
            current_sim_time = self._latest_sim_time()
            if current_sim_time is None:
                continue

            # Make one decision per known node
            for node in list(self.extractor.node_history.keys()):
                try:
                    self._decide_for_node(node, current_sim_time)
                except Exception as e:
                    import traceback
                    print(f"[Decision error] {node}: {e}")
                    traceback.print_exc()

    def _decide_for_node(self, node: str, sim_time: float):
        features = self.extractor.get_state(node, sim_time)
        action, confidence = self.dqn.predict_with_confidence(features)

        # IBN feedback: if this node is currently mitigated, reduce
        # its effective pkt_rate feature so the agent sees the effect
        # of its previous action in the next state
        if node in self._mitigated:
            prev_action = self._mitigated[node]
            if prev_action == 2:   # blocked → rate should appear as 0
                features = features.copy()
                features[0] *= 0.0    # f1 pkt_rate → 0
                features[9] *= 0.0    # f10 pkt_count → 0
            elif prev_action == 1:  # rate-limited → rate reduced to 50%
                features = features.copy()
                features[0] *= 0.5
                features[9] *= 0.5

        # Update IBN mitigation state
        if action == 0:
            self._mitigated.pop(node, None)
        else:
            self._mitigated[node] = action

        self._enforce(node, action) 
        intent = self.engine.process(
            dqn_action = action,
            node       = node,
            features   = features.tolist(),
            sim_time   = sim_time,
            raw_data   = {"confidence": confidence},
        )
        self.state.update(intent, features, action)

        # Log significant actions
        if action >= 1:
            action_name = {1: "RATE_LIMIT", 2: "BLOCK"}[action]
            print(
                f"t={sim_time:.1f}s | {node:<15} | "
                f"{action_name} | conf={confidence:.2f} | "
                f"rate={features[0]*2000:.0f}pkt/s"
            )

    def _enforce(self, node: str, action: int):
        """Send enforcement command to OMNeT++ SocketStreamer."""
        last = self._applied_policies.get(node, -1)
        if action == last:
            return   # no change — don't spam

        import json
        cmd = json.dumps({"node": node, "action": action})
        self._cmd_sock.sendto(
            cmd.encode(),
            (self._omnet_host, self._omnet_cmd_port)
        )
        self._applied_policies[node] = action

        action_name = {0: "PASS", 1: "RATE-LIMIT", 2: "BLOCK"}[action]
        print(f"[ENFORCE] {node} → {action_name}")

    def _latest_sim_time(self) -> float | None:
        latest = None
        for history in self.extractor.node_history.values():
            if history:
                t = history[-1]["time"]
                if latest is None or t > latest:
                    latest = t
        return latest


if __name__ == "__main__":
    import socket as _socket

    pipeline = RealtimePipeline()
    pipeline.start()

    # Send test packet events (new format: just node + time + pkt_size)
    sock = _socket.socket(_socket.AF_INET, _socket.SOCK_DGRAM)
    print("[Test] Sending fake raw packet events...")

    # Simulate attacker[0] sending many fast packets
    for i in range(60):
        t   = round(1.0 + i * 0.05, 3)
        pkt = {
            "node":     "attacker[0]" if i % 4 != 0 else f"ue[{i%5}]",
            "time":     t,
            "pkt_size": 512,
        }
        sock.sendto(json.dumps(pkt).encode(), ("127.0.0.1", 9999))
        time.sleep(0.02)

    time.sleep(2.0)
    snap = pipeline.state.snapshot()
    print(f"\nCounters: {snap['counters']}")
    print(f"Nodes seen: {list(snap['node_timeseries'].keys())}")
    pipeline.stop()