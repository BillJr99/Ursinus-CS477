# Reinforcement Learning: Value Iteration, Q-Learning, and Genetic Algorithms
<!--
author:   William M. Mongan
language: en
narrator: US English Male
comment:  Render via https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-rl.md
import:   https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md
link:     https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-5
           https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Reinforcement Learning: Value Iteration, Q-Learning, and Genetic Algorithms

This module explores three pillars of **Reinforcement Learning (RL)** — **dynamic programming**, **temporal-difference learning**, and **evolutionary search** — to reveal how agents learn optimal behavior through experience or simulated evolution.

---

## Open Colab: RL from Scratch (Toy Gridworld)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/RL_FromScratch_ToyGridworld.ipynb)

---

## Open Colab: Q-Learning vs. Value Iteration (Gridworld)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/QLearning_vs_ValueIteration_Gridworld.ipynb)

---

## Open Colab: Genetic Algorithm (From Scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/GeneticAlgorithm_FromScratch_ToyExample.ipynb)

---

# 1. Reinforcement Learning Overview

**Reinforcement Learning (RL)** is a framework in which an **agent** interacts with an **environment** modeled as a **Markov Decision Process (MDP)**, aiming to learn a policy that maximizes expected cumulative reward.

At each time step $t$:
- Agent observes state $S_t$
- Chooses action $A_t$
- Receives reward $R_{t+1}$
- Transitions to $S_{t+1}$

Objective: maximize discounted return

$$
G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}, \quad 0 \leq \gamma < 1
$$

**Two complementary on-ramps before the math:**

[![Reinforcement Learning from scratch](https://img.youtube.com/vi/vXtfdGphr3c/0.jpg)](https://www.youtube.com/watch?v=vXtfdGphr3c)

*Video summary (text equivalent):* a short animated cartoon that builds RL intuition with no equations — an agent tries actions, gets rewards, and gradually prefers what worked — ending with how the same trial-and-error loop, scaled up, trained AlphaGo.

[![Reinforcement Learning, by the Book (Mutual Information)](https://img.youtube.com/vi/NFo9v_yKQXA/0.jpg)](https://www.youtube.com/watch?v=NFo9v_yKQXA)

*Video summary (text equivalent):* a visual walkthrough of the standard (Sutton & Barto) formalism — states, actions, rewards, returns, policies, and value functions — matching the notation used throughout this module, with gridworld animations for each definition.

---

# 2. Value Functions and Optimality

- **State-value function** under policy $\pi$:  
  $$V^{\pi}(s) = \mathbb{E}_{\pi}[G_t | S_t = s]$$
- **Action-value function:**  
  $$Q^{\pi}(s,a) = \mathbb{E}_{\pi}[G_t | S_t = s, A_t = a]$$

Bellman expectation equation:
$$
V^{\pi}(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^{\pi}(s')]
$$

The **optimal value** and **optimal policy** satisfy:
$$
V^*(s) = \max_a Q^*(s,a), \quad Q^*(s,a) = \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma \max_{a'} Q^*(s',a')]
$$

---

# 3. Value Iteration — Dynamic Programming Approach

## 3.1 Algorithm

Value Iteration iteratively applies the Bellman optimality operator:

$$
V_{k+1}(s) \leftarrow \max_a \Big[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s') \Big].
$$

Converges to $V^*$ and yields optimal policy $\pi^*(s) = \arg\max_a Q^*(s,a)$.

## 3.2 Code Example (from Colab)

```python
import numpy as np

gamma = 0.9
V = np.zeros(num_states)
for i in range(1000):
    delta = 0
    for s in range(num_states):
        v = V[s]
        V[s] = max(sum(p*(r + gamma*V[s2]) for (p,s2,r) in transitions[s][a])
                   for a in actions)
        delta = max(delta, abs(v - V[s]))
    if delta < 1e-6: break
policy = {s: np.argmax([sum(p*(r + gamma*V[s2]) for (p,s2,r) in transitions[s][a])
                        for a in actions]) for s in range(num_states)}
```

---

# 4. Q-Learning — Model-Free Reinforcement Learning

## 4.1 Core Idea

Q-learning learns $Q^*(s,a)$ **without knowing** transition probabilities or reward functions.

Update rule:

$$
Q(s,a) \leftarrow Q(s,a) + \alpha [R + \gamma \max_{a'} Q(s',a') - Q(s,a)]
$$

- $\alpha$ – learning rate  
- $\gamma$ – discount factor

Exploration is typically achieved via $\epsilon$-greedy action selection.

---

## 4.2 Pseudocode

```
Initialize Q(s,a) arbitrarily
for each episode:
    Initialize s
    repeat until terminal:
        Choose a from s using ε-greedy(Q)
        Take action a, observe r, s'
        Q[s,a] ← Q[s,a] + α [r + γ max_a' Q[s',a'] − Q[s,a]]
        s ← s'
```

---

## 4.3 Practical Comparison (Colab Integration)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/QLearning_vs_ValueIteration_Gridworld.ipynb)

This notebook visualizes convergence of Value Iteration and Q-learning on the same gridworld.  
Observe how model-free learning (Q-learning) gradually approximates $Q^*$ through exploration.

[[MC]]
Which statement is **true** about Q-learning?
- (x) It does not require knowledge of transition probabilities.
- ( ) It requires a known reward model.
- ( ) It computes exact expectations instead of samples.

---

# 5. From Tabular to Continuous Spaces

In continuous or high-dimensional spaces, we approximate $Q(s,a)$ with neural networks — leading to **Deep Q-Networks (DQNs)**.  
The loss function becomes:

$$
L(\theta) = \mathbb{E}_{(s,a,r,s')} [(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s,a;\theta))^2]
$$

Key innovations:
- Experience replay buffer
- Target network for stability
- ε-decay schedules for exploration

**Deep dives on combining RL with neural networks:**

[![Reinforcement Learning with Neural Networks: Essential Concepts (StatQuest)](https://img.youtube.com/vi/9hbQieQh7-o/0.jpg)](https://www.youtube.com/watch?v=9hbQieQh7-o)

*Video summary (text equivalent):* StatQuest explains, at the concept level, how a neural network can serve as a policy — mapping states to action probabilities — and how rewards tell us *which direction* to nudge the network's weights, the idea behind policy-gradient methods.

[![Reinforcement Learning with Neural Networks: Mathematical Details (StatQuest)](https://img.youtube.com/vi/DVGmsnxB2UQ/0.jpg)](https://www.youtube.com/watch?v=DVGmsnxB2UQ)

*Video summary (text equivalent):* the companion video works one weight update end to end — backpropagating through the policy network exactly as in our Neural Networks module, but with the reward signal standing in for a supervised label. Watch it after the Essential Concepts video.

---

# 6. Genetic Algorithms (GAs) for RL

## 6.1 Overview

A **Genetic Algorithm** (GA) is an **evolutionary search** method inspired by natural selection.  
It evolves a population of candidate solutions using:
- Selection (survival of the fittest)
- Crossover (recombination)
- Mutation (diversification)

GAs can optimize RL policies **without gradients or differentiable models**.

---

## 6.2 Code Excerpt (From Scratch)

```python
import numpy as np

def fitness(policy):
    return evaluate_policy(policy)  # environment reward

def crossover(p1, p2):
    point = np.random.randint(len(p1))
    return np.concatenate((p1[:point], p2[point:]))

def mutate(child, rate=0.05):
    mask = np.random.rand(len(child)) < rate
    child[mask] += np.random.normal(0, 0.1, mask.sum())
    return np.clip(child, -1, 1)

population = [np.random.uniform(-1,1,policy_dim) for _ in range(50)]
for generation in range(100):
    scores = np.array([fitness(p) for p in population])
    elite_idx = np.argsort(scores)[-10:]
    new_pop = [population[i] for i in elite_idx]
    while len(new_pop) < 50:
        p1, p2 = np.random.choice(new_pop, 2, replace=False)
        child = mutate(crossover(p1, p2))
        new_pop.append(child)
    population = new_pop
```

---

## 6.3 GA Notebook Integration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/GeneticAlgorithm_FromScratch_ToyExample.ipynb)

In this notebook:
- Policies are encoded as weight vectors.
- Fitness corresponds to cumulative environment reward.
- Demonstrates evolution toward higher fitness over generations.

[[MC]]
What is the **primary difference** between Q-learning and Genetic Algorithms?
- (x) Q-learning updates values using temporal differences; GAs evolve populations of policies.
- ( ) Both rely on explicit model gradients.
- ( ) GAs require exact transition probabilities.

---

# 7. Integrating GAs with RL

**Hybrid methods** leverage both paradigms:
- Use GAs to evolve hyperparameters or neural network initializations for Q-learning.
- Alternate gradient-based and evolutionary updates.
- “Neuroevolution” approaches (e.g., NEAT) evolve architectures directly.

---

# 8. Exploration vs. Exploitation Across Methods

| Method | Type | Exploration Mechanism | Model-Free | Key Advantage |
|--------|------|------------------------|-------------|----------------|
| Value Iteration | DP | Full state sweeps | No | Exact for small MDPs |
| Q-Learning | TD | ε-Greedy sampling | Yes | Model-free learning |
| Genetic Algorithm | Evolutionary | Population diversity | Yes | No gradient needed |

---

# 9. Discussion and Exercises

1. Implement asynchronous updates for Q-learning (online).  
2. Add softmax action selection and compare learning stability.  
3. Implement GA-based policy evolution on the same gridworld and compare convergence speed.  
4. Explore mutation rates: what trade-offs arise between exploration and convergence?  

---

# 10. Open Colab Links (for convenience)

- RL From Scratch (Toy Gridworld): [Open in Colab](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/RL_FromScratch_ToyGridworld.ipynb)
- Q-Learning vs. Value Iteration (Gridworld): [Open in Colab](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/QLearning_vs_ValueIteration_Gridworld.ipynb)
- Genetic Algorithm (From Scratch): [Open in Colab](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/GeneticAlgorithm_FromScratch_ToyExample.ipynb)
