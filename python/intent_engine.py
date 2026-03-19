import time 
from dataclasses import dataclass, field
from typing import Optional
from enum import Enum

# 1. Enums -> fixed vocabulary for intents
class Severity(Enum):
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"

class ActionType(Enum):
    ALLOW  = "ALLOW"   # DQN action=0, normal traffic
    BLOCK  = "BLOCK"   # DQN action=1, confirmed attack
    ALERT  = "ALERT"   # DQN action=1 but low confidence (future use)
    MONITOR = "MONITOR" # Suspicious but not yet confirmed
 
# 2. Intent Dataclass -> structured output
@dataclass
class Intent:
    intent_str : str
    policy : str
    action : str
    action_type : str
    severity : Severity
    node : str
    sim_time : float
    active_since : float
    duration : float
    features : list
    confidence : float
    metadata : dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "intent" : self.intent_str,
            "policy" : self.policy,
            "action" : self.action,
            "action_type" : self.action_type.value,
            "severity" : self.severity.value,
            "node" : self.node,
            "sim_time" : round(self.sim_time, 3),
            "active_since" : round(self.active_since, 3),
            "duration" : round(self.duration, 3),
            "features" : [round(f,4) for f in self.features],
            "confidence" : round(self.confidence, 3),
            "metadata": self.metadata,
        }
    
    def __str__(self) -> str:
        lines = [
            f"╔══ INTENT ENGINE OUTPUT ══════════════════════════╗",
            f"  Intent   : {self.intent_str}",
            f"  Policy   : {self.policy}",
            f"  Action   : {self.action}",
            f"  Severity : {self.severity.value}",
            f"  Node     : {self.node}",
            f"  Sim Time : t={self.sim_time:.3f}s",
            f"  Duration : Active since t={self.active_since:.3f}s "
            f"({self.duration:.2f}s ago)",
            f"  Confidence: {self.confidence*100:.1f}%",
            f"╚══════════════════════════════════════════════════╝",
        ]
        return "\n".join(lines)
    
# Part 3: IntentEngine : main class 
class IntentEngine:

    # High-level goal strings — rotated based on context
    INTENT_TEMPLATES = {
        "attack" : "Protect 5G network from DDoS",
        "flood"  : "Mitigate UDP flood on gNB uplink",
        "normal" : "Maintain quality of service for authorized UEs",
        "monitor": "Monitor suspicious traffic pattern",
    }

    # Policy templates — filled in with node/port details
    POLICY_TEMPLATES = {
        "port_block" : "Block high-rate UDP flood on port {port}",
        "rate_limit" : "Rate-limit node {node} exceeding {rate:.0f} pkt/s",
        "full_block" : "Drop all traffic from node {node} {CRITICAL flood}",
        "allow" : "Pass traffic from verified UE {node}",
        "monitor" : "Log and monitor {node} - borderline traffic",
    }

    def __init__(self):
        self.node_state : dict = {}
        self.intent_log : list = []
        self.blocked_nodes : set = set()

    def process(
            self,
            dqn_action : int,
            node : str,
            features : list,
            sim_time : float,
            raw_data : Optional[dict] = None,
    ) -> Intent:
        
        raw_data = raw_data or {}

        # Step 1: Update intternal state for this node
        self._update_node_state(node, dqn_action, sim_time)
        state = self.node_state[node]

        # Step 2: Compute derived values
        confidence = self._compute_confidence(features, dqn_action)
        severity = self._compute_severity(features, dqn_action, state)
        duration = sim_time - state["first_attack_time"] if state["first_attack_time"] is not None else 0.0

        # Step 3: Build intent strings
        intent_str, policy, action_str, action_type = self._build_intent(
            dqn_action, node, features, severity, raw_data
        )

        # Step 4: Update blocked set
        if action_type == ActionType.BLOCK:
            self.blocked_nodes.add(node)
        elif action_type == ActionType.ALLOW and node in self.blocked_nodes:
            self.blocked_nodes.discard(node)

        # Step 5: Assemble and log Intent
        intent = Intent(
            intent_str = intent_str,
            policy = policy,
            action = action_str,
            action_type = action_type,
            severity = severity,
            node = node,
            sim_time = sim_time,
            active_since = state["first_attack_time"] or sim_time,
            duration = duration,
            features = features,
            confidence = confidence,
            metadata = self._build_metadata(node, raw_data, state),
        )

        self.intent_log.append(intent)
        return intent
    
    # NODE STATE TRACKING

    def _update_node_state(self, node:str, action:int, sim_time: float):
        if node not in self.node_state:
            self.node_state[node] = {
                "first_seen" : sim_time,
                "first_attack_time" : None,    # set when first attack detected
                "last_seen" : sim_time,
                "consecutive_attacks" : 0,
                "consecutive_normal" : 0,
                "total_attacks" : 0,
                "total_normal" : 0,
                "severity_history" : [],
            }

        s = self.node_state[node]
        s["last_seen"] = sim_time

        if action == 1:
            s["total_attacks"] += 1
            s["consecutive_normal"] = 0
            s["consecutive_attacks"] += 1
            if s["first_attack_time"] is None:
                s["first_attack_time"] = sim_time
            else:
                s["total_normal"] += 1
                s["consecutive_normal"] += 1
                s["consecutive_attacks"] = 0

    # CONFIDENCE SCORE

    def _compute_confidence(self, features: list, dqn_action: int) -> float:
        f1, f2, f3, f4 = features[0], features[1], features[2], features[3]

        attack_score = (0.35*f1) + (0.25*f2) + (0.25*f3) + (0.15*f4)
        if dqn_action == 1:
            return min(1.0, 0.5 + attack_score)
        else:
            return min(1.0, 0.5+(1.0 - attack_score))
        
    # Severity Classification

    def _compute_severity(
            self, features: list, dqn_action: int, state: dict
    ) -> Severity:
        if dqn_action == 0:
            return Severity.LOW
        
        f1, f2, f3, f4 = features[0], features[1], features[2], features[3]
        consec = state["consecutive_attacks"]

        if f1 > 0.6 and f4 == 1.0 and consec > 5:
            return Severity.CRITICAL
        elif f1 > 0.4 and f4 == 1.0:
            return Severity.HIGH
        elif f1 > 0.1 or f3 > 0.5:
            return Severity.MEDIUM
        else:
            return Severity.LOW
 
    # Policy string builder

    def _build_intent(
            self, 
            dqn_action : int,
            node : str,
            features : list,
            severity : Severity,
            raw_data : dict,
    ):
        port = int(raw_data.get("port", 0))
        pkt_rate = float(raw_data.get("pkt_rate", features[0] * 1500))

        if dqn_action == 0:
            # NORMAL TRAFFIC
            intent_str = self.INTENT_TEMPLATES["normal"]
            policy_str = self.POLICY_TEMPLATES["allow"].format(node=node)
            action_str = f"ALLOW {node}"
            action_type = ActionType.ALLOW

        else:
            # ATTACK TRAFFIC 
            if port == 4000:
                intent_str = self.INTENT_TEMPLATES["flood"]
            else:
                intent_str = self.INTENT_TEMPLATES["attack"]
            
            if severity == Severity.CRITICAL:
                policy_str = self.POLICY_TEMPLATES["full_block"].format(node=node)
                action_str = f"DROP ALL {node}"
                action_type = ActionType.BLOCK

            elif severity == Severity.HIGH:
                policy_str = self.POLICY_TEMPLATES["port_block"].format(port=port)
                action_str = f"BLOCK {node}"
                action_type = ActionType.BLOCK

            elif severity == Severity.MEDIUM:
                policy_str = self.POLICY_TEMPLATES["rate_limit"].format(node=node, rate=pkt_rate)
                action_str = f"RATE-LIMIT {node}"
                action_type = ActionType.BLOCK

            else :
                policy_str = self.POLICY_TEMPLATES["monitor"].format(node=node)
                action_str = f"MONITOR {node}"
                action_type = ActionType.MONITOR

        return intent_str, policy_str, action_str, action_type
    
    # Metadata Builder
    def _build_metadata(self, node:str, raw_data: dict, state: dict) -> dict:
        return {
            "port" : raw_data.get("port"),
            "pkt_rate": raw_data.get("pkt_rate"),
            "pkt_size": raw_data.get("pkt_size"),
            "interval": raw_data.get("interval"),
            "total_attacks": state["total_attacks"],
            "total_normal": state["total_normal"],
            "consecutive_attacks": state["consecutive_attacks"],
            "is_blocked": node in self.blocked_nodes,
        }
    
    def get_blocked_nodes(self) -> list:
        return list(self.blocked_nodes)
    
    def get_attack_summary(self) -> dict:
        return {
            "total_nodes_seen" : len(self.node_state),
            "total_blocked" : len(self.blocked_nodes),
            "blocked_nodes" : list(self.blocked_nodes),
            "intent_log_size" : len(self.intent_log),
            "node_stats" : {
                node: {
                    "total_attacks" : s["total_attacks"],
                    "total_normal" : s["total_normal"],
                    "first_seen" : s["first_seen"],
                    "first_attack" : s["first_attack_time"],
                }
                for node, s in self.node_state.items()
            },
        }
    
    def get_recent_intents(self, n: int=20) -> list:
        return [i.to_dict() for i in self.intent_log[-n:]]
    
    def reset(self):
        self.node_state.clear()
        self.intent_log.clear()
        self.blocked_nodes.clear()

if __name__ == "__main__":
    engine = IntentEngine()
 
    test_cases = [
        # (action, node,           features,                            time,  raw)
        (1, "attacker[0]", [0.667, 1.0, 1.0, 1.0, 0.052, 0.5], 5.20,
         {"port":4000,"pkt_rate":1000,"pkt_size":1500,"interval":0.001}),
 
        (0, "ue[0]",       [0.007, 0.341, 0.0, 0.0, 0.053, 0.05], 5.30,
         {"port":5000,"pkt_rate":10,"pkt_size":512,"interval":0.1}),
 
        (1, "attacker[1]", [0.667, 1.0, 1.0, 1.0, 0.054, 0.48], 5.40,
         {"port":4000,"pkt_rate":1000,"pkt_size":1500,"interval":0.001}),
 
        # Repeated attacks from attacker[0] to trigger CRITICAL
        *[
            (1, "attacker[0]", [0.667, 1.0, 1.0, 1.0, 0.05+i*0.01, 0.5], 5.5+i,
             {"port":4000,"pkt_rate":1000,"pkt_size":1500,"interval":0.001})
            for i in range(7)
        ],
    ]
 
    for action, node, feats, t, raw in test_cases:
        intent = engine.process(action, node, feats, t, raw)
        print(intent)
        print()
 
    print("\n=== Attack Summary ===")
    import json
    print(json.dumps(engine.get_attack_summary(), indent=2))
        
