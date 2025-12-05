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

$$
\mathbf{x}_2 =
[0.6, 0.3, 0.1]
\begin{bmatrix}
0.6 & 0.3 & 0.1 \\\\
0.4 & 0.4 & 0.2 \\\\
0.2 & 0.6 & 0.2
\end{bmatrix}
$$

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

> It has a unique **steady‑state distribution** $ \pi $ such that  
> $ \pi P = \pi $.

We solve the equation:

$$
\pi P = \pi, \qquad \pi_1 + \pi_2 + \pi_3 = 1.
$$

This is equivalent to finding the eigenvector of $ P^T $ with eigenvalue 1.

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

with damping factor $ d \approx 0.85 $.

The steady state of this Markov chain gives the **PageRank** scores.

---

### What Is PageRank?

PageRank is Google’s original method for estimating how *important* a webpage is.

Think of a **random surfer**:

- They start on a random webpage.  
- They click links at random to move to new pages.  
- Sometimes they get bored and jump to a totally random page.  

Pages that the surfer visits *more often in the long run* get **higher PageRank**.

This is exactly how a **Markov chain** behaves:

- Pages = **states**
- Links = **transition probabilities**
- Teleportation = **damping factor** (usually $ d = 0.85 $)

---

### Conceptual Walkthrough

Suppose we have three pages:

- **A** links to B and C  
- **B** links to C  
- **C** links to A  

#### Build the link-following probability table

If a page has multiple outgoing links, we split probability equally.

| From → To | A | B | C |
|-----------|---|---|---|
| **A**     | 0 | 1/2 | 1/2 |
| **B**     | 0 | 0 | 1 |
| **C**     | 1 | 0 | 0 |

This gives the transition matrix:

$$
A =
\begin{bmatrix}
0 & \tfrac{1}{2} & \tfrac{1}{2} \\
0 & 0 & 1 \\
1 & 0 & 0
\end{bmatrix}
$$

---

### Adding Teleportation (Google Matrix)

Google fixes "dead ends" by allowing the surfer to jump to *any* page occasionally.

The PageRank matrix is:

$$
P = dA + (1-d)\frac{1}{n}\mathbf{1}\mathbf{1}^T,
\quad d=0.85,\ n=3
$$

Teleportation adds a small probability $ \frac{1-d}{3} = 0.05 $ to every entry.

So the Google matrix $ P $ becomes:

$$
P =
\begin{bmatrix}
0.05 & 0.475 & 0.475 \\
0.05 & 0.05  & 0.90 \\
0.90 & 0.05  & 0.05
\end{bmatrix}
$$

---

### Computing PageRank (Beginner Math)

Start with an equal guess:

$$
x^{(0)} = \left(\tfrac{1}{3},\tfrac{1}{3},\tfrac{1}{3}\right)
$$

Apply the update:

$$
x^{(k+1)} = x^{(k)} P
$$

After several rounds:

$$
x^{(\infty)} \approx (0.40,\ 0.15,\ 0.45)
$$

Meaning:

- **C** is the most visited page ≈ 45%
- **A** is second ≈ 40%
- **B** is least visited ≈ 15%

This ranking *is* the PageRank.

---

### Math Theory Connection to Markov Chains

A Markov chain describes how probabilities move between states.

In PageRank:

- States = webpages  
- Transition matrix = Google matrix $ P $  
- Teleportation ensures **irreducible + aperiodic** chain  
- Therefore it has a **unique steady-state distribution**

A steady state $ \pi $ satisfies:

$$
\pi = \pi P
$$

This eigenvector equation gives the PageRank.

---

### Code Example

```python
import numpy as np

# Transition matrix from the example
A = np.array([
    [0,   1/2, 1/2],
    [0,   0,   1  ],
    [1,   0,   0  ]
])

def pagerank(A, d=0.85, tol=1e-8, max_iter=100):
    n = A.shape[0]
    teleport = np.ones((n, n)) / n
    P = d * A + (1 - d) * teleport

    x = np.ones(n) / n  # start uniformly
    for _ in range(max_iter):
        x_new = x @ P
        if np.linalg.norm(x_new - x, 1) < tol:
            return x_new
        x = x_new
    return x

print("PageRank:", pagerank(A))
```

---

### Summary

- PageRank models a “random surfer” as a Markov chain.  
- We build a link-based probability matrix.  
- We fix problems with *teleportation*.  
- The steady state of the Google matrix is the PageRank ranking.  
- Power iteration is an easy way to compute it.

---

# Hidden Markov Models (HMMs): Overview

## Concept and Intuition

A Hidden Markov Model formalizes situations where:

- The system evolves through **unseen (hidden) states** over time.
- Each hidden state **probabilistically emits an observable symbol**.
- State changes follow a **Markov process**:  
  the next state depends *only* on the current state.
- Learning or inference requires working backward from observations to the most plausible hidden structure.

**Key components**

- Hidden states: $ S = \{s_1, \dots, s_N\} $  
- Observations: $ O = (o_1, \dots, o_T) $ from alphabet $ V $
- Transition probabilities:  
  $ A = [a_{ij}] = P(q_{t+1}=s_j \mid q_t=s_i) $
- Emission probabilities:  
  $ B = [b_j(k)] = P(\text{obs}=v_k \mid q_t=s_j) $
- Initial distribution:  
  $ \pi = [\pi_i] = P(q_1 = s_i) $

**Interpretive framing**

HMMs encode both **temporal structure** (via transitions) and **signal generation** (via emissions). Inference therefore answers: *which hidden path through time most plausibly produced what we observed?*

---

## The Viterbi Algorithm – A Conceptual Walkthrough

The **Viterbi algorithm** solves the problem:

$$
\arg\max_{q_1,\ldots,q_T} \Pr(q_1,\ldots,q_T,\ O_1,\ldots,O_T)
$$

> **Conceptual narrative:**  
> Imagine you see a sequence of *clues* (the observations $O_1, \dots, O_T$), and you believe those clues were produced by some hidden process (the hidden states $q_1, \dots, q_T$).  
>   
> The question is:  
> **“Among all possible hidden stories that could have produced these clues, which single story is most plausible?”**  
>   
> The expression above says: “Pick the sequence of hidden states $q_1, \dots, q_T$ that maximizes the joint probability of *both* the hidden states and the observed data.”  
> We are not just guessing each state separately; we are finding the *best overall path* through time.

## Why We Need Dynamic Programming

Naively, to solve

$$
\arg\max_{q_1,\ldots,q_T} \Pr(q_1,\ldots,q_T,\ O_1,\ldots,O_T),
$$

we would have to consider **every possible sequence** of hidden states of length $T$. If there are $N$ states, that is $N^T$ possible paths — exponential in $T$.

> **Conceptual narrative:**  
> Exponential search is like trying to read every book in a huge library just to answer a single question.  
>   
> The key idea of dynamic programming (and of Viterbi) is:  
> - **Do not re-solve the same subproblem repeatedly.**  
> - Instead, **store the “best so far” partial answers** and extend them step by step.  
>   
> We want a method that says: “At time $t$, what is the best way to have arrived at each possible state, given what we have seen so far?”

## The Core Quantity $\delta_t(j)$

We define:

$$
\delta_t(j) = \max_{q_1,\dots,q_{t-1}}
\Pr(q_1,\dots,q_t=s_j,\ O_1,\dots,O_t)
$$

> **Conceptual narrative:**  
> Think of $\delta_t(j)$ as the **score of the best story so far** that:
> 1. Ends in state $s_j$ at time $t$, and  
> 2. Is consistent with the first $t$ observations $O_1, \dots, O_t$.  
>   
> Intuitively: for each time step $t$ and each state $s_j$, $\delta_t(j)$ tells you **how likely the most plausible story is if we *force* the story to currently be in state $s_j$**.

## The Backpointer $\psi_t(j)$

We define:

$$
\psi_t(j) = \arg\max_i \ \delta_{t-1}(i)\, a_{ij}.
$$

> **Conceptual narrative:**  
> The backpointer answers:  
> **“If the best story at time $t$ ends in state $s_j$, which previous state $s_i$ did we come from?”**  
>   
> Later, we reconstruct the most likely full path by following these backpointers backward like arrows left on a trail.

## Initialization Step

$$
\delta_1(j) = \pi_j \, b_j(O_1)
$$

> **Conceptual narrative:**  
> At $t=1$, there is no past.  
> $\pi_j$ is the probability of starting in $s_j$, and $b_j(O_1)$ is the probability of producing the first observation.  
> This forms the **initial scores**.

## Recursion: Extending the Best Paths

$$
\delta_t(j) = \Big( \max_i \delta_{t-1}(i)\, a_{ij} \Big) b_j(O_t)
$$

> **Conceptual narrative:**  
> To end in $s_j$ at time $t$, we consider all possible predecessor states $s_i$.  
> We choose the best one, extend that story into $s_j$, then weight it by the probability of emitting $O_t$.

## Termination

$$
q_T^* = \arg\max_j \delta_T(j)
$$

> **Conceptual narrative:**  
> After processing all observations, pick the state with the highest final score as the endpoint of the best story.

## Backtrace: Recovering the Whole Path

$$
q_t^* = \psi_{t+1}(q_{t+1}^*) \quad t=T-1,\dots,1
$$

> **Conceptual narrative:**  
> Follow the backpointers backward to reconstruct the entire most-likely hidden sequence.

## Why This Is Efficient

> **Conceptual narrative:**  
> Instead of enumerating $N^T$ paths, we compute only $N$ scores per time step.  
> The algorithm runs in $O(N^2 T)$ time.

## Worked Example: Umbrella World

We have two weather states: Sunny (S) and Rainy (R).  
Observation: Umbrella (U).

Initial probabilities: $\pi_S = 0.5$, $\pi_R = 0.5$.  
Emissions: $b_S(U)=0.1$, $b_R(U)=0.8$.

## Step 1: Initialization

$$
\delta_1(S) = 0.5 \cdot 0.1 = 0.05
$$
$$
\delta_1(R) = 0.5 \cdot 0.8 = 0.40
$$

> **Conceptual narrative:**  
> Seeing an umbrella makes Rain more plausible at $t=1$.

## Step 2: Second Observation

Observation $O_2 = U$.  
Transitions: $a_{SS}=0.7$, $a_{SR}=0.3$, $a_{RS}=0.4$, $a_{RR}=0.6$.

## Computing $\delta_2(S)$

$$
\delta_2(S) = \max
\begin{cases}
0.05\cdot0.7 = 0.035 \\
0.40\cdot0.4 = 0.16
\end{cases}
\cdot 0.1
$$

Thus:

$$
\delta_2(S) = 0.016
$$

> **Conceptual narrative:**  
> The best way to end sunny at $t=2$ is to have been rainy at $t=1$.

## Computing $\delta_2(R)$

$$
\delta_2(R) = \max
\begin{cases}
0.05\cdot0.3 = 0.015 \\
0.40\cdot0.6 = 0.24
\end{cases}
\cdot 0.8
$$

Thus:

$$
\delta_2(R) = 0.192
$$

> **Conceptual narrative:**  
> Staying rainy yields a much stronger story.

## Interpretation After Two Steps

- $\delta_2(S) = 0.016$  
- $\delta_2(R) = 0.192$

> **Conceptual narrative:**  
> Two umbrellas make Rain the overwhelmingly likely hidden state.

---

## Numerical Example: Weather and Umbrellas

### Hidden states:
- $ s_1 = \text{Sunny} $
- $ s_2 = \text{Rainy} $

### Observations:
- `Umbrella`  
- `NoUmbrella`

### Transition matrix
$$
A = 
\begin{bmatrix}
0.7 & 0.3 \\
0.4 & 0.6
\end{bmatrix}
$$

### Emission matrix
$$
B = 
\begin{bmatrix}
0.1 & 0.9 \\
0.8 & 0.2
\end{bmatrix}
$$

Interpretation:

- When **Sunny**, umbrellas appear only 10% of the time.  
- When **Rainy**, umbrellas appear 80% of the time.  
- Rain persists with probability 0.6, while sun persists with probability 0.7.

### Observation sequence

$$
O = (\text{Umbrella},\ \text{Umbrella},\ \text{NoUmbrella})
$$

Goal: infer the most likely weather pattern behind these umbrella sightings.

---

## Mathematical Foundations: The Viterbi Algorithm

The **Viterbi algorithm** solves:

$$
\arg\max_{q_1,\ldots,q_T} \Pr(q_1,\ldots,q_T,\ O_1,\ldots,O_T)
$$

It does so by dynamic programming:

### Recurrence

Define  
$$
\delta_t(j) = \max_{q_1,\dots,q_{t-1}}
\Pr(q_1,\dots,q_t=s_j,\ O_1,\dots,O_t)
$$

and the backpointer  
$$
\psi_t(j) = \arg\max_i \ \delta_{t-1}(i)\, a_{ij}.
$$

Then:

- **Initialization**
  $$
  \delta_1(j) = \pi_j \, b_j(O_1)
  $$

- **Recursion**
  $$
  \delta_t(j) = \Big( \max_i \delta_{t-1}(i)\, a_{ij} \Big) b_j(O_t)
  $$

- **Termination**
  $$
  q_T^* = \arg\max_j \delta_T(j)
  $$

- **Backtrace**
  $$
  q_t^* = \psi_{t+1}(q_{t+1}^*) \quad \text{for} \ t=T-1,\dots,1
  $$

This converts an exponential search over all paths into a polynomial-time algorithm.

---

## Worked Viterbi Example (First Two Steps)

Let us compute the first steps explicitly for intuition.

### Step 1: Initialization

Observation $O_1 = \text{Umbrella}$.

$$
\delta_1(\text{Sunny}) = \pi_{\text{S}}\,b_{\text{S}}(U) = 0.5 \cdot 0.1 = 0.05
$$
$$
\delta_1(\text{Rainy}) = \pi_{\text{R}}\,b_{\text{R}}(U) = 0.5 \cdot 0.8 = 0.40
$$

Rain is already more plausible given an umbrella.

### Step 2: Second Observation $O_2=\text{Umbrella}$

For Sunny at $t=2$:

$$
\delta_2(S) = \max
\begin{cases}
\delta_1(S)\cdot a_{SS} = 0.05\cdot0.7 = 0.035 \\
\delta_1(R)\cdot a_{RS} = 0.40\cdot0.4 = 0.16
\end{cases}
\cdot\ b_S(U)=0.1
$$

Thus  
$$
\delta_2(S) = 0.16\cdot0.1=0.016
\quad(\text{backpointer} = R)
$$

For Rainy at $t=2$:

$$
\delta_2(R) = \max
\begin{cases}
0.05\cdot0.3 = 0.015 \\
0.40\cdot0.6 = 0.24
\end{cases}
\cdot b_R(U)=0.8
$$

Thus  
$$
\delta_2(R) = 0.24\cdot0.8 = 0.192
\quad(\text{backpointer}=R)
$$

Rain continues to dominate.

---

## Python Implementation

```python
import numpy as np

# Transition probabilities
A = np.array([[0.7, 0.3],
              [0.4, 0.6]])

# Emission probabilities
B = np.array([[0.1, 0.9],   # Sunny emits (Umbrella, NoUmbrella)
              [0.8, 0.2]])  # Rainy emits (Umbrella, NoUmbrella)

# Initial distribution
pi = np.array([0.5, 0.5])

obs_map = {"Umbrella": 0, "NoUmbrella": 1}
O = ["Umbrella", "Umbrella", "NoUmbrella"]
Oidx = [obs_map[o] for o in O]
```

---

## Viterbi Function

```python
def viterbi(A, B, pi, O):
    T = len(O)
    N = A.shape[0]

    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)

    # Initialization
    delta[0] = pi * B[:, O[0]]

    # Dynamic programming recursion
    for t in range(1, T):
        for j in range(N):
            probs = delta[t-1] * A[:, j]
            psi[t, j] = np.argmax(probs)
            delta[t, j] = probs[psi[t, j]] * B[j, O[t]]

    # Backtrace
    path = np.zeros(T, dtype=int)
    path[-1] = np.argmax(delta[-1])
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    return path, delta

path, delta = viterbi(A, B, pi, Oidx)
print("Most likely state sequence indices:", path)
```

---

# Summary

- HMMs model sequential processes with hidden structure and observable outputs.  
- The Viterbi algorithm provides an efficient method for decoding the most likely hidden trajectory.  
- The weather/umbrella example illustrates how HMMs combine temporal dynamics with probabilistic emissions to infer underlying causes.  

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
