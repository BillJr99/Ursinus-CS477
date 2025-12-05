# Markov Decision Processes, Hidden Markov Models, and Multi‑Armed Bandits
<!--
author:   William M. Mongan
language: en
narrator: US English Male

comment: Render with https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-mdp-hmm-bandits.md or locally if deployed via https://www.billmongan.com/LiaScript/?https://raw.githubusercontent.com/BillJr99/Ursinus-CS477/gh-pages/_pages/Activities/liascript-mdp-hmm-bandits.md

import: https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md

link:   https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-4
        https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Markov Decision Processes, Hidden Markov Models, and Multi‑Armed Bandits

This module unifies three cornerstone frameworks for decision‑making under uncertainty:

1. **Markov Chains and Markov Decision Processes (MDPs)** for sequential control with **state, action, and reward**.
2. **Hidden Markov Models (HMMs)** for **latent‑state** time series: filtering, smoothing, decoding, and learning.
3. **Multi‑Armed Bandits (MABs)** for **exploration–exploitation** with minimal state.

We proceed from **formalism $\to$ algorithms $\to$ guarantees $\to$ practice**, with tightly integrated notebooks.

---

## Open Colab: Markov Chains (Foundations)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Markov_Chain_Tutorial.ipynb)

---

## Open Colab: HMM – Viterbi/Forward–Backward (Core)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HMM_Viterbi_Colab.ipynb)

---

## Open Colab: HMM – Robot Localization Heatmaps (Filtering)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HMM_Robot_Localization_Heatmaps_From_Scratch.ipynb)

---

## Open Colab: HMM – Speech Commands (Tiny/Fast)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HMM_Speech_Commands_Tiny_Fast.ipynb)

---

## Open Colab: HMM – Image Recognition (From Scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HMM_Image_Recognition_From_Scratch.ipynb)

---

## Open Colab: HMM – ECG Artifact Detector (From Scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HMM_ECG_Artifact_Detector_From_Scratch.ipynb)

---

## Open Colab: HMM – Bull/Bear Market Detection (From Scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HMM_Bull_Bear_Detection_From_Scratch.ipynb)

---

## Open Colab: Multi‑Armed Bandits – Project Funding (From Scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/ProjectFunding_MultiArmedBandit_FromScratch.ipynb)

---

## 0. Conventions & Utilities

- Random variables in uppercase, realizations in lowercase, parameters by Greek letters.  
- Vectors boldface like $\mathbf{p}$, matrices upper‑case Roman.  
- All math uses $...$ and $$...$$ delimiters.

---

## Code Cell
```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)
print("Utilities ready.")
```

---

# Markov Chains: Overview

## 1. Conceptual Introduction

A **Markov Chain** is a mathematical model for situations where something moves between states over time, and the key assumption is:

> **The future depends only on the present, not the past.**

This is called the **Markov Property**.

### Example Context
You want to model the weather:
- States: `Sunny`, `Cloudy`, `Rainy`.
- Each day, the weather transitions to another state with a certain probability.

We write these probabilities in a **transition matrix**:

$$
P = \begin{bmatrix}
0.6 & 0.3 & 0.1 \\\\
0.4 & 0.4 & 0.2 \\\\
0.2 & 0.6 & 0.2
\end{bmatrix}
$$

Each row corresponds to “today”, each column to “tomorrow”.

---

## 2. Numerical Example (Exact Arithmetic)

Suppose today is **Sunny**. Represent this as a probability vector:

$$
\mathbf{x}_0 = [1, 0, 0].
$$

To find tomorrow’s distribution:

$$
\mathbf{x}_1 = \mathbf{x}_0 P
$$

Compute:

$$
\mathbf{x}_1
= [1, 0, 0]
\begin{bmatrix}
0.6 & 0.3 & 0.1 \\\\
0.4 & 0.4 & 0.2 \\\\
0.2 & 0.6 & 0.2
\end{bmatrix}
= [0.6,\; 0.3,\; 0.1].
$$

After two days:

$$
\mathbf{x}_2 = \mathbf{x}_1 P.
$$

We compute this explicitly:

\[
\mathbf{x}_2 =
[0.6, 0.3, 0.1]
\begin{bmatrix}
0.6 & 0.3 & 0.1 \\\\
0.4 & 0.4 & 0.2 \\\\
0.2 & 0.6 & 0.2
\end{bmatrix}
\]

Compute each component:

- Sunny:
  $$
  0.6(0.6)+0.3(0.4)+0.1(0.2)=0.36+0.12+0.02=0.50
  $$
- Cloudy:
  $$
  0.6(0.3)+0.3(0.4)+0.1(0.6)=0.18+0.12+0.06=0.36
  $$
- Rainy:
  $$
  0.6(0.1)+0.3(0.2)+0.1(0.2)=0.06+0.06+0.02=0.14
  $$

Thus:

$$
\mathbf{x}_2 = [0.50, 0.36, 0.14].
$$

---

## 3. Theory: Regular Markov Chains and Steady-State

A Markov chain is **regular** if some power of its transition matrix has **all positive entries**.

If a chain is regular:

> It has a unique **steady‑state distribution** \( \pi \) such that  
> \( \pi P = \pi \).

We solve the equation:

$$
\pi P = \pi, \qquad \pi_1 + \pi_2 + \pi_3 = 1.
$$

This is equivalent to finding the eigenvector of \( P^T \) with eigenvalue 1.

---

## 4. Code Example (Python)

```python
import numpy as np

P = np.array([
    [0.6, 0.3, 0.1],
    [0.4, 0.4, 0.2],
    [0.2, 0.6, 0.2]
])

x0 = np.array([1,0,0])

# evolve for 10 steps
x = x0
for _ in range(10):
    x = x @ P

print("After 10 days:", x)

# steady state via eigenvector method
vals, vecs = np.linalg.eig(P.T)
idx = np.argmin(np.abs(vals - 1))
pi = vecs[:, idx].real
pi = pi / pi.sum()
print("Steady state:", pi)
```

---

## 5. PageRank as a Markov Chain

Google PageRank views the web as a Markov chain:
- Pages = states  
- Links = transition probabilities  

Basic PageRank formula:

$$
P = dA + (1-d)\frac{1}{n}\mathbf{1}\mathbf{1}^T
$$

with damping factor \( d \approx 0.85 \).

The steady state of this Markov chain gives the **PageRank** scores.

---

# Hidden Markov Models (HMMs): Overview

## 1. Conceptual Introduction

A Hidden Markov Model has:
1. **Hidden states** — not directly observable  
2. **Observations** — what we see  
3. **Transition matrix** \( A \)  
4. **Emission matrix** \( B \)  
5. **Initial distribution** \( \pi \)

Example:  
Weather is hidden; whether someone carries an umbrella is visible.

---

## 2. Weather/Umbrella Example

States (hidden):
- `Sunny`
- `Rainy`

Observations:
- `Umbrella`
- `NoUmbrella`

Transition matrix:

$$
A = \begin{bmatrix}
0.7 & 0.3 \\\\
0.4 & 0.6
\end{bmatrix}
$$

Emission matrix:

$$
B = \begin{bmatrix}
0.1 & 0.9 \\\\
0.8 & 0.2
\end{bmatrix}
$$

Interpretation:
- When Sunny: 10% umbrella, 90% no umbrella  
- When Rainy: 80% umbrella, 20% no umbrella  

---

## 3. Viterbi Algorithm (Step‑by‑Step Example)

Goal: Given observations  
\[
O = (\text{Umbrella},\ \text{Umbrella},\ \text{NoUmbrella})
\]
find the **most likely hidden state sequence**.

Viterbi computes:

$$
\delta_t(i) = \max_{q_1,\dots,q_{t-1}} \Pr(q_1,\dots,q_t=i,\ O_1,\dots,O_t)
$$

and stores backpointers:

$$
\psi_t(i) = \arg\max_j \delta_{t-1}(j)\,a_{ji}.
$$

We compute everything explicitly (omitted in this brief version, but would include full arithmetic in a longer deck).

---

## 4. Full Python Implementation (Viterbi)

```python
import numpy as np

A = np.array([[0.7, 0.3],
              [0.4, 0.6]])

B = np.array([[0.1, 0.9],
              [0.8, 0.2]])

pi = np.array([0.5, 0.5])

obs_map = {"Umbrella":0, "NoUmbrella":1}
O = ["Umbrella","Umbrella","NoUmbrella"]
Oidx = [obs_map[o] for o in O]

def viterbi(A, B, pi, O):
    T = len(O)
    N = A.shape[0]
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    # init
    delta[0] = pi * B[:, O[0]]

    # recurrence
    for t in range(1, T):
        for j in range(N):
            probs = delta[t-1] * A[:, j]
            psi[t, j] = np.argmax(probs)
            delta[t, j] = probs[psi[t, j]] * B[j, O[t]]

    # termination
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    return path, delta

path, delta = viterbi(A, B, pi, Oidx)
print(\"Most likely state sequence:\", path)
```

---

## 5. Summary

- Markov Chains model observable state transitions.  
- Hidden Markov Models extend this to hidden states.  
- Viterbi algorithm finds the most probable hidden state sequence.  
- PageRank is a Markov chain steady‑state application.

---

# Part I — Markov Chains and MDPs

## 1. Markov Chains (Recap)

A **discrete‑time Markov chain (DTMC)** with finite state space $\mathcal{S}=\{1,\dots,S\}$ is defined by a **row‑stochastic** transition matrix $P \in \mathbb{R}^{S\times S}$ with $P_{ij}=p(X_{t+1}{=}j \mid X_t{=}i)$.
The **$t$‑step** transition matrix is $P^t$. The **stationary distribution** $\pi$ satisfies
$$
\pi^\top = \pi^\top P, \qquad \sum_i \pi_i = 1, \ \pi_i \ge 0.
$$

**Mixing** and **ergodicity** control convergence to stationarity.

---

## Code Cell — Power iteration for stationarity
```python
P = np.array([[0.9, 0.1],[0.3, 0.7]])
pi = np.array([0.5, 0.5])
for _ in range(20):
    pi = pi @ P
print("Approx stationary:", pi/pi.sum())
```

---

## 2. Markov Decision Processes (MDPs)

An **MDP** is a tuple $(\mathcal{S}, \mathcal{A}, P, R, \gamma)$ with discount $\gamma \in [0,1)$. For a policy $\pi(a\mid s)$, define the **value function**
$$
V^{\pi}(s) = \mathbb{E}_{\pi} \Bigg[ \sum_{t=0}^{\infty} \gamma^t R(S_t, A_t) \;\Big|\; S_0=s \Bigg].
$$
The **Bellman expectation** and **optimality** equations are
$$
V^{\pi}(s) = \sum_a \pi(a\mid s) \Big[ R(s,a) + \gamma \sum_{s'} P(s'\mid s,a) \, V^{\pi}(s') \Big],
$$
$$
V^*(s) = \max_a \Big[ R(s,a) + \gamma \sum_{s'} P(s'\mid s,a) V^*(s') \Big].
$$

Define $Q^\pi(s,a)$ analogously; optimal $\pi^*$ is **greedy** w.r.t. $V^*$ (or $Q^*$).

---

## 3. Algorithms: Policy/Value Iteration

**Policy Iteration**
1. Policy evaluation: solve linear system for $V^{\pi}$ or iterate to convergence.  
2. Policy improvement: $\pi \leftarrow \text{Greedy}(V^{\pi})$.  
Converges in finite steps for finite MDPs.

**Value Iteration**
$$
V_{k+1}(s) \leftarrow \max_a \Big[ R(s,a) + \gamma \sum_{s'} P(s'\mid s,a) V_k(s') \Big].
$$
Stop when $\lVert V_{k+1}-V_k \rVert_\infty \le \epsilon (1-\gamma)/(2\gamma)$.

---

## Code Cell — Value Iteration (Toy Grid)
```python
S, A = 4, [0,1]  # toy; two actions
P = {
    (0,0): [(1,1.0)], (0,1): [(0,1.0)],
    (1,0): [(2,1.0)], (1,1): [(1,1.0)],
    (2,0): [(3,1.0)], (2,1): [(2,1.0)],
    (3,0): [(3,1.0)], (3,1): [(3,1.0)],
}
R = {(s,a): (1.0 if s==2 and a==0 else 0.0) for s in range(S) for a in A}
gamma = 0.95
V = np.zeros(S)
for _ in range(100):
    V_new = np.zeros_like(V)
    for s in range(S):
        Qs = []
        for a in A:
            Qs.append(R[(s,a)] + gamma*sum(p*V[s2] for s2,p in [(s2,prob) for (s2,prob) in P[(s,a)]]))
        V_new[s] = max(Qs)
    if np.max(np.abs(V_new - V)) < 1e-6: break
    V = V_new
print("V* ~", V)
```

---

## 4. Optimality, Contraction, and Complexity

The Bellman optimality operator $T[V](s) = \max_a \{ R(s,a) + \gamma \sum_{s'} P(s'\mid s,a) V(s') \}$ is a **$\gamma$‑contraction** under $\lVert\cdot\rVert_\infty$:
$$
\lVert T[V]-T[W]\rVert_\infty \le \gamma \lVert V-W\rVert_\infty.
$$
Hence value iteration converges to $V^*$ from any initial $V_0$. Complexity per sweep is $O(|\mathcal{S}|^2 |\mathcal{A}|)$ for dense $P$.


---

# Part II — Hidden Markov Models (HMMs)

## 5. Generative Structure

An HMM has latent states $Z_t \in \{1,\dots,K\}$ and observations $X_t$ with parameters $\{\pi, A, \{\theta_k\}\}$:
- Initial $\pi_k = p(Z_1{=}k)$,
- Transition $A_{ij}=p(Z_{t+1}{=}j \mid Z_t{=}i)$,
- Emission $p(X_t \mid Z_t=k; \theta_k)$.

**Independence:**
$$
p(Z_{1:T}, X_{1:T}) = p(Z_1) \prod_{t=1}^{T-1} p(Z_{t+1}\mid Z_t) \prod_{t=1}^{T} p(X_t\mid Z_t).
$$

---

## 6. Inference: Forward, Backward, and Viterbi

**Forward (filtering):** $\alpha_t(k) = p(X_{1:t}, Z_t{k}\!=\!k)$ with
$$
\alpha_{t+1}(j) = \Big[ \sum_i \alpha_t(i) A_{ij} \Big] \, b_j(X_{t+1}), \quad b_j(x)=p(x\mid Z{=}j).
$$

**Backward:** $\beta_t(i) = p(X_{t+1:T} \mid Z_t{i}\!=\!i)$ with
$$
\beta_t(i) = \sum_j A_{ij} \, b_j(X_{t+1}) \, \beta_{t+1}(j).
$$

**Smoothing:** $p(Z_t=k \mid X_{1:T}) \propto \alpha_t(k)\,\beta_t(k)$.

**Viterbi (decoding):**
$$
\delta_{t+1}(j) = \max_i \big[ \delta_t(i) + \log A_{ij} \big] + \log b_j(X_{t+1}), \quad
\psi_{t+1}(j) = \arg\max_i \big[ \delta_t(i) + \log A_{ij} \big].
$$

Use **log‑space** to avoid underflow.

---

## 7. Learning: EM/Baum–Welch (Sketch)

Maximize $\log p(X_{1:T}\mid \Theta)$ via EM:
- **E‑step:** compute $\gamma_t(k)=p(Z_t{=}k\mid X)$ and $\xi_t(i,j)=p(Z_t{=}i,Z_{t+1}{=}j\mid X)$ using forward–backward.
- **M‑step:** update
$$
\hat{\pi}_k = \gamma_1(k), \quad
\hat{A}_{ij} = \frac{\sum_{t=1}^{T-1} \xi_t(i,j)}{\sum_{t=1}^{T-1} \gamma_t(i)}, \quad
\hat{\theta}_k \leftarrow \arg\max_{\theta} \sum_{t=1}^T \gamma_t(k) \log p(X_t\mid \theta).
$$

---

## Code Cell — Log‑space Forward (Toy)
```python
def logsumexp(arr):
    m = np.max(arr)
    return m + np.log(np.sum(np.exp(arr - m)))

def forward_log(pi, A, logB):
    T = logB.shape[1]; K = A.shape[0]
    alpha = np.full((K, T), -np.inf)
    alpha[:,0] = np.log(pi) + logB[:,0]
    for t in range(1, T):
        for j in range(K):
            alpha[j,t] = logsumexp(alpha[:,t-1] + np.log(A[:,j])) + logB[j,t]
    return alpha

pi = np.array([0.6, 0.4]); A = np.array([[0.7,0.3],[0.2,0.8]])
# Emissions for a binary observation sequence, e.g., Bernoulli params per state
B = np.array([[0.9,0.1,0.9],[0.2,0.8,0.2]]); x = np.array([1,0,1])
logB = np.log(B*(x) + (1-B)*(1-x))
alpha = forward_log(pi, A, logB)
print("log p(x):", logsumexp(alpha[:,-1]))
```

---

## 8. Domain‑Specific HMMs (with Colabs)

### 8.1 Robot Localization (Filtering)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HMM_Robot_Localization_Heatmaps_From_Scratch.ipynb)

- Motion and observation models on grids; visualize belief $b_t(x)$.
- Effects of sensor noise vs. motion noise on convergence.

[[MC]]
If observation reliability increases while motion is unchanged, the steady‑state belief tends to:
- (x) Concentrate more sharply around the true state.
- ( ) Spread out.
- ( ) Remain unchanged.

### 8.2 Speech Commands (Tiny/Fast)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HMM_Speech_Commands_Tiny_Fast.ipynb)

- Discrete HMM over quantized audio features; Viterbi decoding for keywords.

### 8.3 Image Recognition (From Scratch)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HMM_Image_Recognition_From_Scratch.ipynb)

- HMMs for sequential pixel/patch models; emission design and scalability considerations.

### 8.4 ECG Artifact Detector (From Scratch)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HMM_ECG_Artifact_Detector_From_Scratch.ipynb)

- Binary latent state (“clean vs. artifact”); thresholding vs. HMM smoothing comparison.

### 8.5 Bull/Bear Market Detection (From Scratch)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HMM_Bull_Bear_Detection_From_Scratch.ipynb)

- Regime switching via HMM; state persistence and heavy‑tail emissions discussion.

---

# Part III — Multi‑Armed Bandits (MABs)

## 9. Problem Setup

$K$ arms with unknown reward distributions. At time $t$, choose $A_t \in \{1,\dots,K\}$, observe $R_t \sim \nu_{A_t}$.
**Cumulative regret** versus the best fixed arm $a^*$ is
$$
\mathcal{R}_T = T \mu^* - \sum_{t=1}^T \mathbb{E}[R_t], \qquad \mu^* = \max_a \mu_a.
$$

---

## 10. Algorithms

### $\varepsilon$‑Greedy
With probability $\varepsilon_t$ explore; else exploit the empirical best. Typical $\varepsilon_t = c/t$.

### UCB1 (Hoeffding)
Select
$$
a_t = \arg\max_a \Big[ \hat{\mu}_a + \sqrt{\tfrac{2\ln t}{N_a(t)}} \Big],
$$
achieving $O(\sum_{a:\Delta_a>0} \tfrac{\ln T}{\Delta_a})$ regret where $\Delta_a=\mu^*-\mu_a$.

### Thompson Sampling (Beta–Bernoulli)
Sample $\theta_a \sim \text{Beta}(\alpha_a,\beta_a)$ per arm and pick $\arg\max_a \theta_a$. Performs competitively and is simple to implement.

---

## Code Cell — UCB1 (Toy Bernoulli Arms)
```python
rng = np.random.default_rng(0)
mus = np.array([0.2, 0.5, 0.6])
K = len(mus); T = 500
success = np.zeros(K); pulls = np.zeros(K)
est = np.zeros(K); reward_sum = 0.0
# pull each arm once
for a in range(K):
    r = (rng.random() < mus[a]).astype(float); reward_sum += r
    success[a] += r; pulls[a] += 1; est[a] = success[a]/pulls[a]
for t in range(K+1, T+1):
    ucb = est + np.sqrt(2*np.log(t)/pulls)
    a = int(np.argmax(ucb))
    r = (rng.random() < mus[a]).astype(float)
    reward_sum += r
    success[a] += r; pulls[a] += 1; est[a] = success[a]/pulls[a]
regret = T*mus.max() - reward_sum
print("Estimated means:", est, "Regret:", round(regret,2))
```

---

## 11. Project Funding Bandits — Colab Integration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/ProjectFunding_MultiArmedBandit_FromScratch.ipynb)

- Compare $\varepsilon$‑greedy, UCB1, and Thompson.  
- Plot cumulative regret and arm‑pull counts; discuss exploration budgets.

[[MC]]
UCB1 balances exploration and exploitation by:
- (x) Adding a confidence bonus that shrinks with $N_a(t)$ and grows with $\ln t$.
- ( ) Sampling from posterior Beta distributions.
- ( ) Cycling arms uniformly.

---

# Part IV — Synthesis and Extensions

## 12. POMDPs: Unifying MDPs and HMMs

An HMM is an MDP with **hidden state** and **no actions**. A **POMDP** generalizes MDPs with observation model $O(o\mid s,a)$. Planning often occurs in **belief space** with $b$ as state:
$$
b'(s') = \eta \, O(o\mid s',a) \sum_s P(s'\mid s,a) b(s).
$$
Approximate methods include **point‑based value iteration** and **Monte Carlo tree search** in belief space.

---

## 13. Regret vs. Discounted Return

Bandits optimize **regret** while MDPs optimize **discounted return**. In episodic settings, cumulative reward and regret are linked; in continuing tasks, consider **average reward** formulations.

---

## 14. Exercises

1. **Value iteration** on a grid MDP with terminal rewards; compare synchronous vs. asynchronous sweeping.  
2. **HMM smoothing:** implement forward–backward and verify that smoothed marginals dominate filtered ones in information (entropy reduction).  
3. **Bandit comparison:** simulate Bernoulli arms with gaps $\Delta \in \{0.05,0.1,0.2\}$; compare regret of $\varepsilon$‑decreasing, UCB1, and Thompson.  
4. **POMDP bridge:** implement a tiny POMDP as an HMM with actions by augmenting the transition matrix per action and performing belief updates.
