"""
intent_engine.py
IBN layer — translates DQN actions into structured network policy intents.

Actions:
  0 = ALLOW    → normal traffic
  1 = THROTTLE → suspicious, rate-limit
  2 = BLOCK    → confirmed attack, drop

Confidence is computed from all 9 features, weighted by NSL-KDD importance.
Low-confidence BLOCKs are automatically downgraded to THROTTLE.
"""

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


class Severity(Enum):
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class ActionType(Enum):
    ALLOW    = "ALLOW"
    THROTTLE = "THROTTLE"
    BLOCK    = "BLOCK"
    MONITOR  = "MONITOR"


@dataclass
class Intent:
    intent_str:   str
    policy:       str
    action:       str
    action_type:  ActionType
    severity:     Severity
    node:         str
    sim_time:     float
    active_since: float
    duration:     float
    features:     list
    confidence:   float
    metadata:     dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "intent":       self.intent_str,
            "policy":       self.policy,
            "action":       self.action,
            "action_type":  self.action_type.value,
            "severity":     self.severity.value,
            "node":         self.node,
            "sim_time":     round(self.sim_time, 3),
            "active_since": round(self.active_since, 3),
            "duration":     round(self.duration, 3),
            "features":     [round(f, 4) for f in self.features],
            "confidence":   round(self.confidence, 3),
            "metadata":     self.metadata,
        }

    def __str__(self) -> str:
        return (
            f"╔══ INTENT ENGINE ══════════════════════════════════╗\n"
            f"  Intent    : {self.intent_str}\n"
            f"  Policy    : {self.policy}\n"
            f"  Action    : {self.action}\n"
            f"  Severity  : {self.severity.value}\n"
            f"  Node      : {self.node}\n"
            f"  Sim Time  : t={self.sim_time:.3f}s\n"
            f"  Duration  : {self.duration:.2f}s\n"
            f"  Confidence: {self.confidence*100:.1f}%\n"
            f"╚══════════════════════════════════════════════════╝"
        )


class IntentEngine:

    # Feature weights for confidence — ordered f1..f9, from NSL-KDD importance
    # Initialised in __init__ once numpy is imported.
    ATTACK_WEIGHTS = None

    CONFIDENCE_THRESHOLD_BLOCK = 0.60   # below this → THROTTLE instead of BLOCK

    INTENT_TEMPLATES = {
        "attack":  "Protect 5G network from DDoS flood",
        "flood":   "Mitigate UDP flood on gNB uplink",
        "normal":  "Maintain QoS for authorised UEs",
        "monitor": "Monitor suspicious traffic pattern",
    }

    POLICY_TEMPLATES = {
        "block":    "DROP all traffic from {node} — CRITICAL flood",
        "port_block": "BLOCK high-rate UDP flood on port {port} from {node}",
        "throttle": "RATE-LIMIT {node} to 50 Kbps",
        "monitor":  "LOG and monitor {node} — borderline traffic",
        "allow":    "PASS traffic from verified UE {node}",
    }

    def __init__(self):
        import numpy as _np
        self._np = _np
        self.ATTACK_WEIGHTS = _np.array(
            [0.20, 0.15, 0.15, 0.12, 0.12, 0.10, 0.08, 0.05, 0.03],
            dtype=_np.float32
        )
        self.node_state:    dict = {}
        self.blocked_nodes: set  = set()
        self.intent_log:    list = []

    # ── Public API ────────────────────────────────────────────────────

    def process(
            self,
            dqn_action: int,
            node:       str,
            features:   list,
            sim_time:   float,
            raw_data:   Optional[dict] = None,
    ) -> Intent:
        raw_data = raw_data or {}
        self._update_node_state(node, dqn_action, sim_time)
        state = self.node_state[node]

        confidence = self._confidence(features, dqn_action)
        severity   = self._severity(features, dqn_action, state)
        duration   = (sim_time - state["first_attack_time"]
                      if state["first_attack_time"] is not None else 0.0)

        # IBN override: downgrade low-confidence BLOCKs
        effective_action = dqn_action
        if dqn_action == 2 and confidence < self.CONFIDENCE_THRESHOLD_BLOCK:
            effective_action = 1

        intent_str, policy, action_str, action_type = self._build_intent(
            effective_action, node, features, severity, raw_data
        )

        if action_type == ActionType.BLOCK:
            self.blocked_nodes.add(node)
        elif action_type == ActionType.ALLOW:
            self.blocked_nodes.discard(node)

        intent = Intent(
            intent_str   = intent_str,
            policy       = policy,
            action       = action_str,
            action_type  = action_type,
            severity     = severity,
            node         = node,
            sim_time     = sim_time,
            active_since = state["first_attack_time"] or sim_time,
            duration     = duration,
            features     = features,
            confidence   = confidence,
            metadata     = self._metadata(node, raw_data, state),
        )
        self.intent_log.append(intent)
        return intent

    def get_blocked_nodes(self) -> list:
        return list(self.blocked_nodes)

    def get_attack_summary(self) -> dict:
        return {
            "total_nodes_seen": len(self.node_state),
            "total_blocked":    len(self.blocked_nodes),
            "blocked_nodes":    list(self.blocked_nodes),
            "intent_log_size":  len(self.intent_log),
            "node_stats": {
                node: {
                    "total_attacks": s["total_attacks"],
                    "total_normal":  s["total_normal"],
                    "first_seen":    s["first_seen"],
                    "first_attack":  s["first_attack_time"],
                }
                for node, s in self.node_state.items()
            },
        }

    def get_recent_intents(self, n: int = 20) -> list:
        return [i.to_dict() for i in self.intent_log[-n:]]

    def reset(self):
        self.node_state.clear()
        self.blocked_nodes.clear()
        self.intent_log.clear()

    # ── Internal helpers ──────────────────────────────────────────────

    def _update_node_state(self, node: str, action: int, sim_time: float):
        if node not in self.node_state:
            self.node_state[node] = {
                "first_seen":          sim_time,
                "first_attack_time":   None,
                "last_seen":           sim_time,
                "consecutive_attacks": 0,
                "consecutive_normal":  0,
                "total_attacks":       0,
                "total_normal":        0,
            }
        s = self.node_state[node]
        s["last_seen"] = sim_time

        if action >= 1:                  # THROTTLE or BLOCK = attack signal
            s["total_attacks"]       += 1
            s["consecutive_normal"]   = 0
            s["consecutive_attacks"] += 1
            if s["first_attack_time"] is None:
                s["first_attack_time"] = sim_time
        else:
            s["total_normal"]        += 1
            s["consecutive_normal"]  += 1
            s["consecutive_attacks"]  = 0

    def _confidence(self, features: list, dqn_action: int) -> float:
        np = self._np
        f = np.array(features[:9], dtype=np.float32)
        if len(f) < 9:
            f = np.pad(f, (0, 9 - len(f)))
        attack_score = float(np.dot(f, self.ATTACK_WEIGHTS))
        if dqn_action >= 1:
            return min(1.0, 0.4 + attack_score)
        else:
            return min(1.0, 0.4 + (1.0 - attack_score))

    def _severity(self, features: list, dqn_action: int,
                  state: dict) -> Severity:
        if dqn_action == 0:
            return Severity.LOW
        f1 = features[0]   # bytes_sec
        f5 = features[4]   # burst_ratio
        consec = state["consecutive_attacks"]
        if f1 > 0.6 and f5 > 0.7 and consec >= 5:
            return Severity.CRITICAL
        elif f1 > 0.4 or f5 > 0.7:
            return Severity.HIGH
        elif f1 > 0.1:
            return Severity.MEDIUM
        return Severity.LOW

    def _build_intent(self, action: int, node: str, features: list,
                      severity: Severity, raw_data: dict):
        port     = int(raw_data.get("dest_port", raw_data.get("port", 0)))
        pkt_rate = float(raw_data.get("pkt_rate", 0))

        if action == 0:
            return (
                self.INTENT_TEMPLATES["normal"],
                self.POLICY_TEMPLATES["allow"].format(node=node),
                f"ALLOW {node}",
                ActionType.ALLOW,
            )

        if action == 1:
            return (
                self.INTENT_TEMPLATES["monitor"],
                self.POLICY_TEMPLATES["throttle"].format(node=node),
                f"THROTTLE {node}",
                ActionType.THROTTLE,
            )

        # action == 2 (BLOCK)
        intent_str = (self.INTENT_TEMPLATES["flood"]
                      if port == 4000
                      else self.INTENT_TEMPLATES["attack"])

        if severity == Severity.CRITICAL:
            return (intent_str,
                    self.POLICY_TEMPLATES["block"].format(node=node),
                    f"DROP ALL {node}", ActionType.BLOCK)
        
        elif severity in (Severity.HIGH, Severity.MEDIUM):
            return (intent_str,
                    self.POLICY_TEMPLATES["port_block"].format(
                        port=port, node=node),
                    f"BLOCK {node}", ActionType.BLOCK)
        else:
            return (intent_str,
                    self.POLICY_TEMPLATES["monitor"].format(node=node),
                    f"MONITOR {node}", ActionType.MONITOR)

    def _metadata(self, node: str, raw_data: dict, state: dict) -> dict:
        return {
            "dest_port":           raw_data.get("dest_port"),
            "pkt_rate":            raw_data.get("pkt_rate"),
            "pkt_size":            raw_data.get("pkt_size"),
            "interval":            raw_data.get("interval"),
            "total_attacks":       state["total_attacks"],
            "total_normal":        state["total_normal"],
            "consecutive_attacks": state["consecutive_attacks"],
            "is_blocked":          node in self.blocked_nodes,
        }


if __name__ == "__main__":
    engine = IntentEngine()
    test_cases = [
        (2, "attacker[0]",
         [0.50, 0.01, 0.8, 0.9, 0.8, 0.7, 0.9, 0.1, 0.8], 5.2,
         {"dest_port": 4000, "pkt_rate": 1000, "pkt_size": 1500, "interval": 0.001}),
        (0, "ue[0]",
         [0.01, 0.6, 0.1, 0.2, 0.3, 0.4, 0.1, 0.9, 0.5], 5.3,
         {"dest_port": 5000, "pkt_rate": 10,   "pkt_size": 512,  "interval": 0.1}),
        (1, "ue[2]",
         [0.12, 0.3, 0.4, 0.5, 0.4, 0.5, 0.5, 0.5, 0.6], 5.4,
         {"dest_port": 5000, "pkt_rate": 180,  "pkt_size": 800,  "interval": 0.02}),
    ]
    for action, node, feats, t, raw in test_cases:
        intent = engine.process(action, node, feats, t, raw)
        print(intent)
        print()