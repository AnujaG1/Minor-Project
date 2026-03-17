import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import os
import sys
from collections import deque

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, "results")
CSV_PATH = os.path.join(RESULTS_DIR, "training_data.csv")
MODEL_PATH = os.path.join(RESULTS_DIR, "dqn_model.pth")

sys.path.append(SCRIPT_DIR)
from environment import NetworkEnv5G

# set device (CPU or GPU if available)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Part 1: Neural Network

class DQNNetwork(nn.Module):
    def __init__(self, state_size=6, action_size=2):
        super(DQNNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_size, 64),
            # Linear layer: 6 inputs → 64 outputs
            # Each of 64 neurons sees all 6 features
            # Learns weights: output = weight * input + bias

            nn.ReLU(),
            nn.Linear(64,64),     # 64 → 64: learns combinations of patterns from layer 1
            nn.ReLU(),

            nn.Linear(64,32),      # 64 → 32: narrowing down to most important patterns
            nn.ReLU(),

            nn.Linear(32, action_size)     # 32 → 2: final Q-values [Q(normal), Q(attack)]
            
        )
    
    def forward(self, x):
        return self.network(x)


# part 2 : Replay Buffer

class ReplayBuffer:

    """
    Stores past experiences for training(memory).
 
    WHY NEEDED:
    Problem: If we train on consecutive steps (t=1, t=2, t=3...),
    the data is highly correlated — the network memorizes sequences,
    not patterns.
 
    Solution: Store all experiences in a buffer.
    Randomly sample from it during training.
    This breaks correlation → network learns actual patterns.
    """

    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        # Unzip: [(s1,a1,r1,ns1,d1), (s2,...)] → ([s1,s2], [a1,a2], 

        return (
            torch.FloatTensor(np.array(states)).to(device),
            torch.LongTensor(actions).to(device),
            torch.FloatTensor(rewards).to(device),
            torch.FloatTensor(np.array(next_states)).to(device),
            torch.FloatTensor(dones).to(device)
        )
    
    def __len__(self):
        """Returns how many experiences are stored."""
        return len(self.buffer)
    

# Part 3 : DQN Agent

class DQNAgent:
    """
DOUBLE DQN TRICK:
  We use TWO identical networks:
 
    1. online_net  → makes decisions, gets trained every step
    2. target_net  → provides stable learning targets, updated slowly
"""

    def __init__(self,state_size=6, action_size=2):
        self.state_size = state_size
        self.action_size = action_size

        self.online_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net = DQNNetwork(state_size, action_size).to(device)
        self.target_net.load_state_dict(self.online_net.state_dict())

        self.target_net.eval()
        # eval() = inference mode, never trains directly
        # Disables dropout/batchnorm if present

        # Optimizer
        self.optimizer = optim.Adam(
            self.online_net.parameters(), lr=0.001
        )

        # Loss function
        self.criterion = nn.MSELoss()

        # Memory
        self.memory = ReplayBuffer(capacity=10000)

        # Hyperparameters
        self.gamma = 0.95

        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        # Multiply epsilon by this after each episode
        # Episode 1:   epsilon = 1.000
        # Episode 100: epsilon = 1.0 * 0.995^100 = 0.606
        # Episode 300: epsilon = 1.0 * 0.995^300 = 0.223
        # Episode 500: epsilon = 1.0 * 0.995^500 = 0.082

        self.batch_size = 64
        self.update_target_every = 10
        self.episode_count = 0

    def act(self, state):
        # choose action using epsilon-greedy policy.
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
        # unsqueeze(0): adds batch dimension
        # state shape: (6,) → (1, 6)
        # Network expects batch format even for single sample

        with torch.no_grad():
            q_values = self.online_net(state_tensor)
        
        return q_values.argmax().item()
    
    def remember(self, state, action, reward, next_state, done):
        # store the experience in replay buffer
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return 0.0
            # Need at least 64 experiences before training starts
            # First ~64 steps: just collecting data, no learning yet

        # Sample random batch from memory
        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        # Each: tensor of shape (64,6) or (64,)

        # What did our network predict ?
        current_q = self.online_net(states)
        # Shape: (64, 2) — Q-values for both actions for each sample

        current_q_values = current_q.gather(1, actions.unsqueeze(1)).squeeze(1)
        # gather: pick only the Q-value for the action actually taken

        # what should the network have predicted ?
        with torch.no_grad():
            next_q = self.target_net(next_states)
            max_next_q = next_q.max(1)[0]    # Best Q-value available in next state

            target_q_values = rewards + self.gamma * max_next_q * (1 - dones)
            # BELLMAN EQUATION:
            # target = reward + 0.95 * best_future_q * (1 if not done)
            # If done=1 (episode ended): target = reward only
            # If done=0 (episode continues): target = reward + future

        loss = self.criterion(current_q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()   #Backward Propagation to reduce loss

        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(),1.0)
        self.optimizer.step()
        return loss.item()
    
    def update_epsilon(self):
        """
        Decay exploration rate after each episode.
        WHY NEEDED:
        Early training: explore randomly to discover patterns
        Late training:  exploit learned knowledge
        This function gradually shifts from explore to exploit.
        """
        self.epsilon = max(
            self.epsilon_min, 
            self.epsilon * self.epsilon_decay
        )
        # Multiply by 0.995 each episode
        # Never go below 0.01
    
    def update_target_network(self):
        # copy online network weightsto target network.
        self.target_net.load_state_dict(self.online_net.state_dict())

    def save(self,path):
        # save trained model to disk. so that we can use it in real time without retraining
        torch.save({
            'online_net': self.online_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'epsilon':  self.epsilon,
            'episode': self.episode_count
        }, path)
        print(f"[DQN] Model saved -> {path}")

    def load(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.online_net.load_state_dict(checkpoint['online_net'])
        self.target_net.load_state_dict(checkpoint['target_net'])
        self.epsilon = checkpoint['epsilon']
        print(f"[DQN] Model loaded <- {path}")

# Part 4: Training Loop

def train(episodes=500):
    print("\n" + "="*60)
    print(" PHASE 4: DQN TRAINING - 5G Attack Detection")
    print("="*60)

    env = NetworkEnv5G(CSV_PATH)
    agent = DQNAgent(state_size=6, action_size=2)
    episode_rewards = []
    episode_losses = []
    accuracies = []
    best_accuracy = 0.0

    print(f"\n Training for {episodes} episodes.")
    print(f"Each episode = 200 steps through network traffic data")
    print(f"Total Decisions = {episodes * 200:,}")
    print("-"*60)

    # main loop
    for episodes in range(1, episodes + 1 ):
        state, info = env.reset()
        total_reward = 0.0
        total_loss = 0.0
        steps = 0

        # Episode loop = 200 steps
        while True:
            #step 1 : Agent decides action
            action = agent.act(state)

            # Step 2: Environment grades the decision
            next_state, reward, terminated, truncated, info = env.step(action)

            # Step 3: Store experience in memory
            done = terminated or truncated
            agent.remember(state, action, reward, next_state, float(done))

            # Step 4: Learn from random past experiences
            loss = agent.train_step()

            # Step 5: Update tracking
            total_reward += reward
            total_loss += loss
            steps += 1
            state = next_state
            # Move to next state for next iteration

            if done:
                break

            # END OF EPISODE 

        agent.update_epsilon()
        agent.episode_count += 1

        if episodes % agent.update_target_every == 0:
            agent.update_target_network()

        # Record metrics
        accuracy = env.get_accuracy()
        avg_loss = total_loss / max(steps, 1)
        episode_rewards.append(total_reward)
        accuracies.append(accuracy)

        if accuracy > best_accuracy and episodes > 50:
            best_accuracy = accuracy
            agent.save(MODEL_PATH)

        if episodes % 50 == 0:
            stats = env.get_stats()
            avg_reward = np.mean(episode_rewards[-50:])
            avg_acc = np.mean(accuracies[-50:])

            print(f"Episode {episodes:4d}/{episodes} | "
                  f"Avg Reward: {avg_reward:8.2f} | "
                  f"Accuracy: {avg_acc:6.2f}% | "
                  f"Epsilon: {agent.epsilon:.3f} | "
                  f"Loss: {avg_loss:.4f}")
            print(f" Caught: {stats['attacks_caught']} | "
                  f"Missed: {stats['attacks_missed']} | "
                  f"False alarms: {stats['false_alarms']}")
            
            env.reset_stats()

    # TRAINING COMPLETE
    print("\n" + "="*60)
    print(f"Training Complete!")
    print(f"Best Accuracy achieved: {best_accuracy:.2f}%")
    print(f"Model saved to: {MODEL_PATH}")
    print("="*60)

    #Run final evaluation on full dataset
    print("\n Running final evaluation on full dataset...")
    test(agent,env)

    return agent

# PART 5: TEST FUNCTION
def test(agent, env):
    
    """
    WHY NEEDED:
    Training accuracy can be misleading (agent sees same data repeatedly).
    This function does a clean single pass through ALL data
    with epsilon=0 (no random actions) to get true performance.
    """
    agent.epsilon = 0.0
    env.reset_stats()
    state, _ = env.reset()
    env.current_pos = 0

    for i in range(len(env.states) - 1):
        action = agent.act(state)
        next_state, reward, terminated, truncated, info = env.step(action)
        state = next_state
        if terminated:
            break

    # Print results
    stats = env.get_stats()
    total = stats['total_correct'] + stats['total_wrong']

    print("\n── FINAL TEST RESULTS ──────────────────────────────")
    print(f"  Total samples evaluated : {total}")
    print(f"  Overall Accuracy        : {stats['Accuracy']:.2f}%")
    print(f"  Attacks caught          : {stats['attacks_caught']}")
    print(f"  Attacks missed          : {stats['attacks_missed']}")
    print(f"  False alarms            : {stats['false_alarms']}")
    print("────────────────────────────────────────────────────")

    if stats['Accuracy'] >= 90:
        print("  ✅ EXCELLENT — Ready for Phase 5 Intent Engine!")
    elif stats['Accuracy'] >= 75:
        print("  ⚠️  GOOD — Consider collecting more training data")
    else:
        print("  ❌ NEEDS WORK — Run more episodes or collect more data")
    print("────────────────────────────────────────────────────")

if __name__ == "__main__":
    os.makedirs(RESULTS_DIR, exist_ok=True)
    train(episodes=500)


        

           




        

 


        
        
 


