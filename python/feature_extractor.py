import numpy as np
import socket, json, threading, queue, logging
from collections import defaultdict

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger(__name__)

MAX_PKT_RATE = 1000.0
MAX_FLOW_DUR = 300.0
MAX_ZSCORE   = 5.0
MAX_BURST    = 20.0
HISTORY_LEN  = 30


class FeatureExtractor:
    def __init__(self, window_sec=1.0, history_limit=500):
        self.window_sec    = window_sec
        self.history_limit = history_limit
        self.node_history  = defaultdict(list)
        self.cell_rates    = {}

    def ingest(self, raw: dict):
        node = raw.get("node", "unknown")
        self.node_history[node].append({
            "time":     float(raw.get("time",      0.0)),
            "pkt_rate": float(raw.get("pkt_rate",  0.0)),
            "pkt_delta":float(raw.get("pkt_delta", 0.0)),
        })
        h = self.node_history[node]
        if len(h) > self.history_limit:
            del h[:len(h) - self.history_limit]
        self.cell_rates[node] = float(raw.get("pkt_rate", 0.0))

    def get_state(self, node: str, current_time: float) -> np.ndarray:
        h = self.node_history.get(node, [])
        if not h:
            return np.zeros(10, dtype=np.float32)

        rates = [e["pkt_rate"] for e in h]
        pkt_rate  = h[-1]["pkt_rate"]
        mean_rate = float(np.mean(rates[-10:])) if rates else 0.0

        # f1: current rate
        f1 = min(pkt_rate / MAX_PKT_RATE, 1.0)

        # f2: smoothed mean rate (last 10 ticks)
        f2 = min(mean_rate / MAX_PKT_RATE, 1.0)

        # f3: burst ratio — current vs own mean
        f3 = min((pkt_rate / mean_rate) / MAX_BURST, 1.0) \
             if mean_rate > 0.1 else 0.0

        # f4: rate acceleration — current tick vs previous
        f4 = min(abs(h[-1]["pkt_rate"] - h[-2]["pkt_rate"]) / MAX_PKT_RATE, 1.0) \
             if len(h) >= 2 else 0.0

        # f5: rate trend — is rate going up over last 5 ticks?
        if len(h) >= 5:
            early = float(np.mean([e["pkt_rate"] for e in h[-5:-2]]))
            late  = float(np.mean([e["pkt_rate"] for e in h[-2:]]))
            f5 = float(np.clip((late - early) / (MAX_PKT_RATE + 1e-6)
                               * 0.5 + 0.5, 0.0, 1.0))
        else:
            f5 = 0.5

        # f6: flow duration
        f6 = min((current_time - h[0]["time"]) / MAX_FLOW_DUR, 1.0)

        # f7: activity ratio — fraction of last 30 ticks with rate > 0
        recent = h[-HISTORY_LEN:]
        f7 = sum(1 for e in recent if e["pkt_rate"] > 0) / len(recent)

        # f8: cell z-score — outlier vs all current nodes
        all_rates = list(self.cell_rates.values())
        if len(all_rates) >= 2:
            mu  = float(np.mean(all_rates))
            sig = float(np.std(all_rates))
            f8  = float(np.clip(
                (pkt_rate - mu) / (sig + 1e-6) / MAX_ZSCORE,
                0.0, 1.0))
        else:
            f8 = 0.0

        # f9: consecutive active ticks (persistence)
        consec = 0
        for e in reversed(h):
            if e["pkt_rate"] > 0: consec += 1
            else: break
        f9 = min(consec / HISTORY_LEN, 1.0)

        # f10: peak rate ever seen for this node
        f10 = min(max(rates) / MAX_PKT_RATE, 1.0)

        return np.array([f1,f2,f3,f4,f5,f6,f7,f8,f9,f10],
                        dtype=np.float32)

    def get_label(self, node: str) -> int:
        return 1 if "attacker" in node.lower() else 0

    def extract_window(self, current_time: float) -> list:
        return [{"node": n, "time": current_time,
                 "state": self.get_state(n, current_time),
                 "label": self.get_label(n)}
                for n in list(self.node_history.keys())]


class UDPReceiver:
    def __init__(self, host="127.0.0.1", port=9999, buffer_size=50000):
        self.host=host; self.port=port
        self.data_queue=queue.Queue(maxsize=buffer_size)
        self.running=False; self.thread=None
        self.stats={"packets_received":0,"packets_dropped":0,
                    "parse_errors":0,"sim_ended":False}

    def start(self):
        self.running=True
        self.thread=threading.Thread(target=self._listen, daemon=True)
        self.thread.start()
        logger.info(f"UDP Receiver on {self.host}:{self.port}")

    def stop(self): self.running=False

    def _listen(self):
        sock=socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        sock.bind((self.host, self.port))
        sock.settimeout(1.0)
        logger.info("Waiting for OMNeT++ packets...")
        while self.running:
            try:
                data,_=sock.recvfrom(4096)
                s=data.decode("utf-8")
                if '"type":"SIM_END"' in s:
                    self.stats["sim_ended"]=True
                    continue
                parsed=json.loads(s)
                try:
                    self.data_queue.put_nowait(parsed)
                    self.stats["packets_received"]+=1
                except queue.Full:
                    self.data_queue.get_nowait()
                    self.data_queue.put_nowait(parsed)
                    self.stats["packets_dropped"]+=1
            except socket.timeout: continue
            except json.JSONDecodeError: self.stats["parse_errors"]+=1
            except Exception as e:
                logger.error(f"Receiver: {e}")
                break
        sock.close()

    def get_data(self):
        items=[]
        while not self.data_queue.empty():
            try: items.append(self.data_queue.get_nowait())
            except queue.Empty: break
        return items

    def get_stats(self): return self.stats.copy()

if __name__ == "__main__":
    import time

    print("="*55)
    print("  FeatureExtractor + UDPReceiver — Live Test")
    print("="*55)
    print("  Listening on 127.0.0.1:9999 for OMNeT++ packets...")
    print("  Start your OMNeT++ simulation now.")
    print("  Press Ctrl+C to stop.\n")

    receiver  = UDPReceiver(host="127.0.0.1", port=9999)
    extractor = FeatureExtractor()

    receiver.start()

    try:
        while True:
            packets = receiver.get_data()

            for raw in packets:
                extractor.ingest(raw)
                node  = raw.get("node", "unknown")
                t     = float(raw.get("time", 0.0))
                state = extractor.get_state(node, t)
                label = extractor.get_label(node)

                print(f"\n  [{t:>8.3f}s] {node:<22} label={'ATTACK' if label else 'NORMAL'}")
                print(f"    f1  pkt_rate      = {state[0]:.4f}")
                print(f"    f2  mean_rate     = {state[1]:.4f}")
                print(f"    f3  burst_ratio   = {state[2]:.4f}")
                print(f"    f4  rate_change   = {state[3]:.4f}")
                print(f"    f5  rate_trend    = {state[4]:.4f}")
                print(f"    f6  flow_duration = {state[5]:.4f}")
                print(f"    f7  activity_ratio= {state[6]:.4f}")
                print(f"    f8  cell_zscore   = {state[7]:.4f}")
                print(f"    f9  consecutive   = {state[8]:.4f}")
                print(f"    f10 peak_rate     = {state[9]:.4f}")

            stats = receiver.get_stats()
            if stats["sim_ended"]:
                print("\n[INFO] SIM_END received from OMNeT++.")
                print(f"  Packets received : {stats['packets_received']}")
                print(f"  Packets dropped  : {stats['packets_dropped']}")
                print(f"  Parse errors     : {stats['parse_errors']}")
                print("  Test PASSED — pipeline is working correctly.")
                break

            time.sleep(0.1)

    except KeyboardInterrupt:
        stats = receiver.get_stats()
        print("\n[INFO] Stopped by user.")
        print(f"  Packets received : {stats['packets_received']}")
        print(f"  Packets dropped  : {stats['packets_dropped']}")
        print(f"  Parse errors     : {stats['parse_errors']}")
        if stats['packets_received'] > 0:
            print("  Test PASSED — packets are being received from OMNeT++.")
        else:
            print("  Test FAILED — no packets received. Check OMNeT++ is running and sending to 127.0.0.1:9999.")

    finally:
        receiver.stop()
