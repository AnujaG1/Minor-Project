import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class NetworkEnv5G(gym.Env):

    metadata = {"render_modes": ["human"]}

    # rewards
    REWARD_CORRECT_ATTACK = +2.0 
    REWARD_CORRECT_NORMAL = +1.0
    REWARD_MISSED_ATTACK = -2.0
    REWARD_FALSE_ALARM = -0.5    # Said attack to normal traffic

    def __init__(self, csv_path, max_steps=200, render_mode=None):
        """
        csv_path  : path to training_data.csv
        max_steps : steps per episode , default = 200
        render_mode : "human" to print each step, none for silen
        """
        super().__init__()

        self.render_mode = render_mode
        self.max_steps = max_steps

        print(f"[Environment] Loading data from {csv_path}")
        self.df = pd.read_csv(csv_path)
        print(f"[Environment] Loaded {len(self.df)} rows")
        print(f"[Environment] Attack rows: {self.df['is_attacker'].sum()}")
        print(f"[Environment] Normal rows: {(self.df['is_attacker']==0).sum()}")

        # feature column are states (input) to DQN
        self.feature_cols = [
            'f1_pkt_rate',
            'f2_pkt_size',
            'f3_interval',
            'f4_port',
            'f5_time',
            'f6_delta'
        ]

        self.states = self.df[self.feature_cols].values.astype(np.float32)

        self.labels = self.df['is_attacker'].values.astype(np.int64)

        self.nodes = self.df['node'].values

        self.node_windows = {}     # stores last N states per node to detect train

        # Define observation space
        self.observation_space = spaces.Box(
            low = 0.0,
            high = 1.0,
            shape = (6,),
            dtype = np.float32
        )

        # define action space  -- agent can choose 0 or 1
        self.action_space = spaces.Discrete(2)

        # Episode tracking 
        self.current_pos = 0     #current position in csv
        self.episode_reward = 0
        self.steps_taken = 0

        # statistics tracking
        self.total_correct = 0
        self.total_wrong = 0
        self.attacks_caught = 0
        self.attacks_missed = 0
        self.false_alarms = 0

        print(f"[Env] observation_space ={self.observation_space},"
              f" [Env] action_space = {self.action_space}")
        print(f"[Env] Ready! ")


    def reset(self, seed=None, options=None):
        # seed    : random seed for reproducibility
        # Handle random seed 
        super().reset(seed=seed, options=options)
        if seed is not None:
                np.random.seed(seed)

        # randomly start a new episode so that the agent doesn't memorize it
        self.current_pos = int(np.random.randint(0, len(self.states)-1))
        self.episode_reward = 0.0
        self.steps_taken = 0

        observation = self.states[self.current_pos]
        info = {"start_pos": self.current_pos}

        return observation, info
        
    def step(self, action):
        # get groung truth for current rows
        true_label = self.labels[self.current_pos]
        current_node = self.nodes[self.current_pos]

        # calculate reward based on action
        if action == 1 and true_label == 1:
            reward = self.REWARD_CORRECT_ATTACK
            self.attacks_caught += 1
            self.total_correct += 1
            outcome = "CORRECT ATTACK DETECTED"
            
        elif action == 0 and true_label == 0:
            reward = self.REWARD_CORRECT_NORMAL
            self.total_correct += 1
            outcome = "CORRECT NORMAL"

        elif action == 0 and true_label == 1:
            reward = self.REWARD_MISSED_ATTACK
            self.attacks_missed += 1
            self.total_wrong += 1
            outcome = "MISSED ATTACK"

        else :
            reward = self.REWARD_FALSE_ALARM
            self.false_alarms += 1
            self.total_wrong += 1
            outcome = "FALSE ALARM"

        # Move to next row
        self.current_pos += 1
        self.episode_reward += reward
        self.steps_taken += 1

        terminated = (self.current_pos >= len(self.states)-1)
        truncated = (self.steps_taken >= self.max_steps)

        if terminated:
            self.current_pos = 0

        # get next observation
        observation = self.states[self.current_pos]

        # build information dictionary for logging
        info = {
            "true label": int(true_label),
            "node": current_node,
            "action_taken": int(action),
            "outcome" : outcome,
            "reward" : reward,
            "step" : self.steps_taken
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, truncated, info
            
    def render(self):
        # print current state only when render mode = human
        if self.render_mode == "human":
                node = self.nodes[self.current_pos]
                state = self.states[self.current_pos]
                label = self.labels[self.current_pos]
                print(f" Step {self.steps_taken:3d} | "
                      f"{node:<15} | "
                      f"label={label}")
                
    def close(self):
            pass
        
    def get_accuracy(self):
            total = self.total_correct + self.total_wrong
            if total == 0:
                return 0.0
            return (self.total_correct / total) * 100
        
    def get_stats(self):
            return {
                "Accuracy" : round(self.get_accuracy(),2),
                "attacks_caught" : self.attacks_caught,
                "attacks_missed" : self.attacks_missed,
                "false_alarms": self.false_alarms,
                "total_correct": self.total_correct,
                "total_wrong": self.total_wrong
            }
        
    def reset_stats(self):
            # resets statistics counters (call between train/test phases)
            self.total_correct = 0
            self.total_wrong = 0
            self.attacks_caught = 0
            self.attacks_missed = 0
            self.false_alarms = 0


# ── Quick test ────────────────────────────────────────────────
if __name__ == "__main__":
    import os
    CSV = os.path.join(os.path.dirname(__file__), "results", "training_data.csv")
 
    env = NetworkEnv5G(CSV, render_mode="human")
 
    # Test reset
    obs, info = env.reset(seed=42)
    print(f"\nInitial obs shape: {obs.shape}")
    print(f"Initial obs:       {obs}")
    print(f"Info:              {info}")
 
    # Test 5 steps
    print("\nRunning 5 steps...")
    for i in range(5):
        action = env.action_space.sample()  # random action
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"  Action={action} | Reward={reward:+.1f} | "
              f"Outcome={info['outcome']}")
 
    env.close()
    print("\n✅ Environment test passed!")
 
        



