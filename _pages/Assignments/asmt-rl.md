---
layout: assignment
permalink: Assignments/ReinforcementLearning
title: "Assignment: Reinforcement Learning — From Scratch, with Libraries, and a Creative Challenge"

info:
  points: 100
  goals:
    - Derive and implement core RL algorithms from first principles.
    - Implement tabular Value Iteration and Q-Learning from scratch on toy MDPs.
    - Use a library environment (e.g., Gymnasium) with a baseline RL agent.
    - Analyze exploration, convergence, and policy quality with clear metrics.
    - Complete a scaffolded creative RL mini-project.
  purpose: "This assignment develops practical proficiency with Reinforcement Learning, connecting Bellman equations to working agents. You will code tabular algorithms, then use libraries for scaled tasks, and finally solve a small creative problem."
  concepts:
    - Markov Decision Processes (MDPs), policies, returns
    - Bellman expectation/optimality equations and contraction
    - Value Iteration and Q-Learning
    - Exploration strategies (\(\epsilon\)-greedy, softmax), learning rates
    - Evaluation: returns, sample efficiency, state visitation
  tasks:
    - Implement Value Iteration and Q-Learning from scratch on a gridworld.
    - Train an agent with a library environment (Gymnasium) and compare results.
    - Complete a creative mini-project with instrumentation and analysis.
  rubric:
    - weight: 30
      description: Implementation
      preemerging: Partial algorithms or unstable code.
      beginning: Working Value Iteration and Q-Learning with basic tests.
      progressing: Modular, well-documented implementations and plots.
      proficient: Solid engineering, reproducibility, and insightful instrumentation.
    - weight: 30
      description: Algorithmic Understanding
      preemerging: Minimal derivations.
      beginning: Correct Bellman equations and update rules.
      progressing: Clear proofs/sketches for convergence and hyperparameter effects.
      proficient: Deep reasoning with ablations and edge-case discussion.
    - weight: 20
      description: Analysis & Evaluation
      preemerging: Limited metrics or discussion.
      beginning: Episodic return curves and simple comparisons.
      progressing: State visitation, policy heatmaps, and sample-efficiency analysis.
      proficient: Thorough analyses with well-argued conclusions.
    - weight: 10
      description: Code Quality & Documentation
      preemerging: Sparse comments.
      beginning: Basic docstrings and structure.
      progressing: Clean modular code with plotting utilities.
      proficient: Excellent organization, type hints, and reproducibility.
    - weight: 10
      description: Submission Completeness
      preemerging: Missing artifacts.
      beginning: Includes code and results.
      progressing: Includes plots and discussion.
      proficient: Fully reproducible with seeds and instructions.

tags:
  - reinforcement-learning
  - dynamic-programming
  - temporal-difference
  - exploration
---

# Overview

In Reinforcement Learning (RL), an **agent** interacts with an **environment** modeled as an MDP and seeks a policy maximizing expected return. You will implement **Value Iteration** and **Q-Learning** from scratch, then use a **library environment** (e.g., Gymnasium) to benchmark a baseline agent. Finally, you will complete a **creative challenge** that requires design decisions and analysis.

---

## Stage 0 — Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
np.set_printoptions(precision=4, suppress=True)
```

(Optional libraries for Stage 3):
```python
# You may need: pip install gymnasium
try:
    import gymnasium as gym
    HAS_GYM = True
except Exception as e:
    HAS_GYM = False
    print("Gymnasium not available; you can still complete Stages 1–2.")
```

---

## Stage 1 — From Scratch: Value Iteration on a Toy Gridworld

Let an MDP be $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ with discount $\gamma\in[0,1)$. The **Bellman optimality** update is
$$
V_{k+1}(s) \leftarrow \max_{a\in\mathcal{A}} \Big[ R(s,a) + \gamma \sum_{s'} P(s'\mid s,a) V_k(s') \Big].
$$

### 1.1 Gridworld Definition

We define a $4\times4$ grid with terminal states and a step cost.

```python
@dataclass
class Gridworld:
    H: int = 4
    W: int = 4
    terminals: tuple = (0, 15)
    step_cost: float = -1.0
    gamma: float = 0.95

    def states(self):
        return list(range(self.H*self.W))

    def actions(self):
        return [0,1,2,3]  # up,right,down,left

    def step(self, s, a):
        if s in self.terminals:
            return s, 0.0
        y, x = divmod(s, self.W)
        if a == 0: y = max(0, y-1)
        elif a == 1: x = min(self.W-1, x+1)
        elif a == 2: y = min(self.H-1, y+1)
        elif a == 3: x = max(0, x-1)
        s2 = y*self.W + x
        r = 0.0 if s2 in self.terminals else self.step_cost
        return s2, r
```

### 1.2 Value Iteration

```python
def value_iteration(env: Gridworld, theta=1e-6):
    S = env.states(); A = env.actions()
    V = np.zeros(len(S))
    policy = np.zeros(len(S), dtype=int)
    while True:
        delta = 0.0
        for s in S:
            v = V[s]
            q = []
            for a in A:
                s2, r = env.step(s, a)
                q.append(r + env.gamma * V[s2])
            V[s] = np.max(q)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    # Greedy policy
    for s in S:
        q = []
        for a in A:
            s2, r = env.step(s, a)
            q.append(r + env.gamma * V[s2])
        policy[s] = int(np.argmax(q))
    return V, policy

env = Gridworld()
V_star, pi_star = value_iteration(env)
print("V* (reshape):\n", V_star.reshape(env.H, env.W))
print("π* (0=U,1=R,2=D,3=L):\n", pi_star.reshape(env.H, env.W))
```

**Checkpoint:** Verify $\lVert V_{k+1}-V_k \rVert_\infty$ decreases and the policy arrows point toward terminal rewards.

---

## Stage 2 — From Scratch: Tabular Q-Learning

Q-Learning learns $Q^*(s,a)$ from samples without knowing $P$ or $R$:
$$
Q(s,a) \leftarrow Q(s,a) + \alpha \Big[ r + \gamma\max_{a'} Q(s',a') - Q(s,a) \Big].
$$

We run episodes on the same gridworld using \(\epsilon\)-greedy exploration.

```python
def run_q_learning(env: Gridworld, episodes=5000, alpha=0.1, gamma=0.95, eps_start=1.0, eps_end=0.05, eps_decay=0.995):
    S = len(env.states()); A = len(env.actions())
    Q = np.zeros((S, A))
    eps = eps_start
    returns = []
    for ep in range(episodes):
        s = np.random.randint(0, S)
        G = 0.0
        for t in range(500):
            if np.random.rand() < eps:
                a = np.random.randint(0, A)
            else:
                a = int(np.argmax(Q[s]))
            s2, r = env.step(s, a)
            Q[s,a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s,a])
            s = s2
            G += r
            if s in env.terminals:
                break
        eps = max(eps_end, eps * eps_decay)
        returns.append(G)
    policy = np.argmax(Q, axis=1)
    return Q, policy, np.array(returns)

Q, pi_q, ret = run_q_learning(env, episodes=3000)
print("Policy from Q-Learning (reshaped):\n", pi_q.reshape(env.H, env.W))
plt.plot(ret, alpha=0.6); plt.title("Episode Return (Q-Learning)"); plt.xlabel("Episode"); plt.ylabel("Return"); plt.show()
```

**Checkpoint:** Compare $\pi^*$ from Value Iteration to the learned policy from Q-Learning. How many episodes are needed for near-optimal returns?

---

## Stage 3 — Library Baseline: Gymnasium Environment (FrozenLake-v1)

Train a simple tabular Q-Learning agent on **FrozenLake-v1** (slippery), if Gymnasium is available.

```python
if HAS_GYM:
    env = gym.make("FrozenLake-v1", is_slippery=True)
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))

    def epsilon_greedy(Q, s, eps):
        return env.action_space.sample() if np.random.rand() < eps else int(np.argmax(Q[s]))

    episodes, alpha, gamma = 5000, 0.8, 0.99
    eps, eps_min, eps_decay = 1.0, 0.05, 0.999
    rewards = []

    for ep in range(episodes):
        s, _ = env.reset()
        done = False; G = 0.0
        while not done:
            a = epsilon_greedy(Q, s, eps)
            s2, r, terminated, truncated, _ = env.step(a)
            done = terminated or truncated
            Q[s,a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s,a])
            s = s2; G += r
        eps = max(eps_min, eps * eps_decay)
        rewards.append(G)

    print("Average return (last 100):", np.mean(rewards[-100:]))
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode="valid"))
    plt.title("FrozenLake Q-Learning (Moving Average Return)"); plt.xlabel("Episode"); plt.ylabel("Avg Return"); plt.show()
else:
    print("Gymnasium not available; skip this stage or replace with another local environment.")
```

**Checkpoint:** Report the moving-average return; adjust exploration and learning rates to improve stability.

---

## Stage 4 — Creative Mini-Project (Scaffold): **Warehouse Robot Navigator**

Design a small **warehouse grid** with moving hazards and a pickup/dropoff task. You will define the MDP, implement the reward structure, and train a Q-Learning agent. Instrument your code to analyze learning dynamics.

### 4.1 Environment Scaffold

```python
@dataclass
class Warehouse:
    H: int = 6
    W: int = 8
    pickup: tuple = (0, 0)
    dropoff: tuple = (5, 7)
    hazards_init: tuple = ((2,2), (3,5))
    gamma: float = 0.99
    step_cost: float = -0.01
    pickup_reward: float = 1.0
    drop_reward: float = 5.0
    hazard_penalty: float = -2.0

    def __post_init__(self):
        self.reset_hazards()

    def reset_hazards(self):
        self.hazards = list(self.hazards_init)

    def to_state(self, y, x, carrying):
        return y*self.W + x + (self.H*self.W)*carrying

    def from_state(self, s):
        cells = self.H*self.W
        carrying = 1 if s >= cells else 0
        s0 = s - cells if carrying else s
        y, x = divmod(s0, self.W)
        return y, x, carrying

    def step(self, s, a):
        # a: 0=up,1=right,2=down,3=left,4=pickup,5=drop
        y, x, carrying = self.from_state(s)
        if a == 0: y = max(0, y-1)
        elif a == 1: x = min(self.W-1, x+1)
        elif a == 2: y = min(self.H-1, y+1)
        elif a == 3: x = max(0, x-1)
        elif a == 4 and (y,x)==self.pickup and carrying==0:
            carrying = 1
        elif a == 5 and (y,x)==self.dropoff and carrying==1:
            # dropping off completes an episode
            s2 = self.to_state(y, x, 0)
            return s2, self.drop_reward, True

        s2 = self.to_state(y, x, carrying)
        r = self.step_cost

        # hazards move horizontally (wrap-around)
        self.hazards = [ (hy, (hx+1) % self.W) for (hy, hx) in self.hazards ]
        if (y,x) in self.hazards:
            r += self.hazard_penalty

        if (y,x)==self.pickup and carrying==1:
            r += self.pickup_reward

        done = False
        return s2, r, done

    @property
    def n_states(self):
        return self.H*self.W*2
    @property
    def n_actions(self):
        return 6
```

### 4.2 Q-Learning Scaffold

```python
def train_warehouse(episodes=5000, alpha=0.2, gamma=0.99, eps_start=1.0, eps_end=0.05, eps_decay=0.999):
    env = Warehouse()
    Q = np.zeros((env.n_states, env.n_actions))
    rng = np.random.default_rng(0)
    eps = eps_start
    ep_returns = []

    for ep in range(episodes):
        env.reset_hazards()
        # random start
        y, x = rng.integers(0, env.H), rng.integers(0, env.W)
        s = env.to_state(y, x, carrying=0)
        G = 0.0
        for t in range(1000):
            a = rng.integers(0, env.n_actions) if rng.random() < eps else int(np.argmax(Q[s]))
            s2, r, done = env.step(s, a)
            Q[s,a] += alpha * (r + gamma * np.max(Q[s2]) - Q[s,a])
            s = s2; G += r
            if done:
                break
        eps = max(eps_end, eps * eps_decay)
        ep_returns.append(G)

    return Q, np.array(ep_returns)

Qw, Rw = train_warehouse()
plt.plot(np.convolve(Rw, np.ones(100)/100, mode="valid"))
plt.title("Warehouse Navigator — Moving Avg Return")
plt.xlabel("Episode"); plt.ylabel("Avg Return"); plt.show()
```

### 4.3 Your Tasks

1. **Shaping & Hazards.** Adjust rewards (pickup/drop/penalty) and hazard motion to improve sample efficiency without reward hacking.  
2. **Exploration.** Compare \(\epsilon\)-greedy vs. softmax exploration; plot state visitation heatmaps.  
3. **Ablation.** Vary $\alpha$, $\gamma$, and hazard dynamics; quantify effects on convergence and final policy quality.  
4. **Policy Visualization.** Render the greedy policy over the grid for carrying/not-carrying.  
5. **Stretch.** Replace Q-Learning with **SARSA** and compare stability.

---

# What to Submit

1. A notebook or scripts implementing **Value Iteration** and **Q-Learning** from scratch (Stages 1–2).  
2. Results for the **Gymnasium** task (Stage 3), or a substitute local environment if Gymnasium is unavailable.  
3. For the **creative project** (Stage 4): code, plots (returns, visitation), and a short analysis (2–3 pages) with ablations.  
4. Reproducibility: random seeds, hyperparameters, and clear documentation.

---
