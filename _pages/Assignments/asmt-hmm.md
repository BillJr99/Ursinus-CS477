---
layout: assignment
permalink: Assignments/HMMModeling
title: "Assignment: Modeling Sequential Data with Hidden Markov Models"

info:
  points: 100
  goals:
    - Gain intuition for probabilistic modeling of sequences using Hidden Markov Models (HMMs).
    - Implement a toy-sized HMM from scratch for a real-world inspired sequential dataset.
    - Compare results with scikit-learn's hmmlearn implementation.
    - Interpret learned parameters (transition, emission, and initial probabilities).
  purpose: "This assignment builds practical understanding of sequential probabilistic models by constructing and interpreting Hidden Markov Models (HMMs). Students will implement a toy HMM by hand, then use standard libraries to verify results. The purpose is to deeply connect mathematical formalism (transition/emission probabilities, likelihoods, decoding) with concrete data-driven examples."
  concepts:
    - Markov assumption and sequential dependencies
    - Hidden vs. observed variables
    - Forward-Backward algorithm and sequence likelihood
    - Viterbi decoding (most likely state path)
    - Baum-Welch parameter reestimation (EM algorithm)
    - Model selection and interpretation
    - Application of scikit-learn’s hmmlearn to a small dataset
  tasks:
    - Implement a simple HMM class with forward, backward, and Viterbi algorithms.
    - Generate synthetic sequences and visualize state emissions.
    - Apply scikit-learn’s hmmlearn to a small real dataset (e.g., UCI weather or POS tagging subset).
    - Compare parameters learned by EM with your manual estimates.
    - Discuss interpretability and potential biases.
  rubric:
    - weight: 30
      description: Implementation
      preemerging: Produces a basic HMM implementation with partial functionality.
      beginning: Implements forward and backward correctly for toy data.
      progressing: Implements Viterbi decoding and validates against known examples.
      proficient: Delivers a complete HMM class with solid structure, correctness, and documentation.
    - weight: 30
      description: Algorithmic Correctness and Reasoning
      preemerging: Provides code that executes without major errors.
      beginning: Explains algorithmic choices with test examples.
      progressing: Validates correctness with consistency checks and detailed reasoning.
      proficient: Provides thorough justification, comparison with library methods, and edge case analysis.
    - weight: 20
      description: Code Quality and Documentation
      preemerging: Code runs with basic comments.
      beginning: Uses structured code and brief docstrings.
      progressing: Clear naming, modular structure, and good commenting.
      proficient: High-quality, well-tested, and well-documented code with interpretable outputs.
    - weight: 10
      description: Analysis and Interpretation
      preemerging: Describes model behavior superficially.
      beginning: Explains transition and emission patterns.
      progressing: Connects parameters to sequence behavior and interpretability.
      proficient: Provides insightful discussion of results, biases, and improvements.
    - weight: 10
      description: Submission Completeness
      preemerging: Provides partial artifacts.
      beginning: Includes required files and brief report.
      progressing: Includes runnable code, report, and clear plots.
      proficient: Fully reproducible with documentation and reflection.

tags:
  - ai
  - probabilistic-models
---

# Overview

This assignment focuses on modeling sequential data using **Hidden Markov Models (HMMs)**.  
You will first build an HMM from scratch to gain a clear understanding of its internal mechanics, then apply and compare it to a real dataset using scikit-learn’s `hmmlearn`.

We will model **weather-based sensor readings** as a meaningful but toy example.

---

## Stage 0 — Setup

You may use the following libraries:

```python
import numpy as np
import matplotlib.pyplot as plt
from hmmlearn import hmm
from sklearn.preprocessing import LabelEncoder
np.random.seed(42)
```

---

## Stage 1 — HMM Intuition and Synthetic Example

Imagine a simple world with two hidden states — **Sunny** and **Rainy** — that influence observable **Umbrella usage** (1 for umbrella, 0 for no umbrella).

The generative process assumes:
- Weather transitions daily according to a Markov chain.
- Observations depend probabilistically on the hidden state.

Define transition, emission, and initial probabilities.

```python
states = ['Sunny', 'Rainy']
obs = ['NoUmbrella', 'Umbrella']

# Hidden state transition probabilities
A = np.array([[0.8, 0.2],   # Sunny → Sunny/Rainy
              [0.4, 0.6]])  # Rainy → Sunny/Rainy

# Emission probabilities P(obs | state)
B = np.array([[0.9, 0.1],   # Sunny → NoUmbrella/Umbrella
              [0.3, 0.7]])  # Rainy → NoUmbrella/Umbrella

# Initial state probabilities
pi = np.array([0.6, 0.4])
```

**Task:**  
1. Simulate 100 days of weather and umbrella usage.  
2. Visualize the hidden vs. observed sequences.

```python
def sample_hmm(A, B, pi, n_steps):
    n_states = A.shape[0]
    n_obs = B.shape[1]
    states = np.zeros(n_steps, dtype=int)
    obs = np.zeros(n_steps, dtype=int)
    states[0] = np.random.choice(n_states, p=pi)
    obs[0] = np.random.choice(n_obs, p=B[states[0]])
    for t in range(1, n_steps):
        states[t] = np.random.choice(n_states, p=A[states[t-1]])
        obs[t] = np.random.choice(n_obs, p=B[states[t]])
    return states, obs

states_seq, obs_seq = sample_hmm(A, B, pi, 100)

plt.figure(figsize=(10,3))
plt.plot(states_seq, 'o-', label="Hidden state (0=Sunny,1=Rainy)")
plt.plot(obs_seq + 0.1, 'x-', label="Observation (0=NoUmbrella,1=Umbrella)")
plt.legend(); plt.show()
```

---

## Stage 2 — Implement Forward Algorithm

The **Forward Algorithm** computes the likelihood of an observation sequence given the model parameters:

$$
P(O | \lambda) = \sum_i \alpha_T(i)
$$

where:

$$
\alpha_t(i) = P(O_1, \dots, O_t, X_t = i | \lambda)
$$

**Recurrence:**
$$
\alpha_t(j) = \left( \sum_i \alpha_{t-1}(i) a_{ij} \right) b_j(O_t)
$$

```python
def forward_algorithm(A, B, pi, O):
    N = A.shape[0]  # number of states
    T = len(O)
    alpha = np.zeros((T, N))
    alpha[0] = pi * B[:, O[0]]
    for t in range(1, T):
        for j in range(N):
            alpha[t, j] = B[j, O[t]] * np.sum(alpha[t-1] * A[:, j])
    return alpha, np.sum(alpha[-1])
```

**Checkpoint:** Verify `np.sum(alpha[-1])` ≈ 1 when normalized.

---

## Stage 3 — Implement Viterbi Decoding

Compute the most likely hidden-state sequence given observations.

```python
def viterbi_decode(A, B, pi, O):
    N, T = A.shape[0], len(O)
    delta = np.zeros((T, N))
    psi = np.zeros((T, N), dtype=int)
    delta[0] = np.log(pi * B[:, O[0]])
    for t in range(1, T):
        for j in range(N):
            seq_probs = delta[t-1] + np.log(A[:, j])
            psi[t, j] = np.argmax(seq_probs)
            delta[t, j] = np.max(seq_probs) + np.log(B[j, O[t]])
    states = np.zeros(T, dtype=int)
    states[-1] = np.argmax(delta[-1])
    for t in reversed(range(1, T)):
        states[t-1] = psi[t, states[t]]
    return states
```

**Checkpoint:** Compare Viterbi path vs. true simulated state path.

---

## Stage 4 — Use `hmmlearn` on Real Data

Now apply a trained HMM to a real sequential dataset. For example, we can use **weather patterns** from the UCI dataset or `sklearn.datasets.fetch_openml("daily-temperatures")`.

```python
from sklearn.datasets import fetch_openml
data = fetch_openml(name="daily-temperatures", version=1, as_frame=True)
temps = data.data['Temp'].values
obs_disc = np.digitize(temps, bins=np.percentile(temps, [33,66]))
obs_disc = obs_disc.reshape(-1,1)

model = hmm.MultinomialHMM(n_components=3, n_iter=50, random_state=42)
model.fit(obs_disc)
hidden_states = model.predict(obs_disc)

plt.figure(figsize=(10,3))
plt.plot(temps[:200], label="Temperature")
plt.plot(hidden_states[:200], label="HMM states")
plt.legend(); plt.show()
```

**Checkpoint:**  
- Interpret transition matrix: does it capture daily regime shifts?  
- What do emission probabilities correspond to? (e.g., low/mid/high temperature regimes)

---

## Stage 5 — Discussion

Answer the following questions in your report:

1. Compare your **hand-built HMM** and **hmmlearn** models. How do the transition/emission probabilities differ?
2. Discuss how the number of hidden states influences interpretability and likelihood.
3. Explain how Viterbi decoding relates to Bayesian inference and MAP estimation.
4. If your dataset had missing or noisy observations, how could the HMM handle this?
5. Reflect on how HMMs differ conceptually and practically from feedforward networks for sequence data.

---

---

## Stage 6 — Creative Mini-Project: Design & Solve a New HMM Problem

In this stage, you will **define and solve your own HMM problem**. Choose **one** of the options below (or propose your own), implement it end-to-end, and explain your modeling decisions. Your solution must include: (i) a clearly stated modeling goal, (ii) HMM specification (states, emissions, priors), (iii) training/decoding, and (iv) evaluation with thoughtful interpretation.

> **Deliverables:** A short write-up and runnable code cells. Use your Stage 2/3 **from-scratch** routines where appropriate (forward, Viterbi), and then **validate** using a library implementation (e.g., `hmmlearn`).

### Option A — Digit Stroke Patterns via Row-Scan HMM (no downloads)

Model **stroke regimes** within handwritten digits by scanning each image row-by-row as a time series. Use `sklearn.datasets.load_digits` (8×8 images).

**Idea.** Each image becomes a length-8 sequence where observation at time $t$ is the **row intensity** (sum or mean of row pixels). Hidden states capture regimes like “blank”, “light stroke”, “heavy stroke”.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from hmmlearn import hmm

digits = load_digits()
X_imgs = digits.images        # shape: (n_samples, 8, 8)
y_lbls = digits.target

def row_series(img):
    # length-8 sequence: mean intensity per row (normalize to [0,1])
    v = img.mean(axis=1)
    v = (v - v.min()) / (v.ptp() + 1e-8)
    return v

seqs = [row_series(img) for img in X_imgs]

# Build an HMM dataset for Gaussian emissions:
# Each sequence must be shape (T, 1) and lengths array tells hmmlearn the boundaries.
X_concat = np.concatenate([s.reshape(-1,1) for s in seqs], axis=0)
lengths = [len(s) for s in seqs]

# Choose K hidden states to reflect stroke regimes (try K ∈ {2,3,4}).
K = 3
model = hmm.GaussianHMM(n_components=K, covariance_type="diag", n_iter=200, random_state=0)
model.fit(X_concat, lengths)

# Decode (Viterbi) on a few sequences and visualize
def plot_decode(idx):
    s = seqs[idx].reshape(-1,1)
    z = model.predict(s)
    plt.figure(figsize=(6,2))
    plt.plot(s[:,0], marker="o", label="row intensity")
    plt.step(range(len(z)), z, where="mid", label="states (Viterbi)")
    plt.title(f"Digit={y_lbls[idx]}  (K={K})")
    plt.legend(); plt.tight_layout(); plt.show()

for idx in [0, 1, 2]:
    plot_decode(idx)
```

**Questions.**
1. What **semantics** do you assign to each state (e.g., “blank/light/heavy”)?  
2. Compare **$K=2$ vs. $K=4$**: which yields more interpretable segmentation?  
3. Compute sequence log-likelihoods and report **average per-time-step log-likelihood** by digit class; does the HMM capture class-specific structure?

---

### Option B — Character-Stream Topic HMM (self-contained text)

Treat a character stream as observations; hidden states represent **latent writing modes** (e.g., vowel-heavy vs. consonant-heavy). Create a **two-state** HMM with **categorical emissions** over characters.

**Mathematics.**
- Hidden states $Z_t \in \{1,2\}$ with transition matrix $A$ and initial distribution $\pi$.
- Emissions are categorical: $X_t \mid Z_t=k \sim \mathrm{Cat}(\boldsymbol{\phi}_k)$ over an alphabet $\mathcal{V}$.
- Train with Baum–Welch (EM) or compare to your from-scratch forward–backward + M-step updates.

```python
import numpy as np
from hmmlearn import hmm

text = ("in data we trust but we also verify "
        "models are wrong some are useful "
        "probabilistic thinking helps us reason under uncertainty")
alphabet = sorted({c for c in text if c.isalpha() or c == " "})
char2i = {c:i for i,c in enumerate(alphabet)}
seq = np.array([char2i[c] for c in text], dtype=int).reshape(-1, 1)

K = 2  # latent modes
model = hmm.CategoricalHMM(n_components=K, n_iter=200, random_state=0)
model.n_features = len(alphabet)
model.fit(seq)

z = model.predict(seq)  # Viterbi path
# Inspect emissions
phi = model.emissionprob_  # shape (K, |V|)
for k in range(K):
    top = np.argsort(-phi[k])[:8]
    print(f"State {k}: ", [(alphabet[j], round(phi[k,j],3)) for j in top][:8])
```

**Questions.**
1. Do the two states separate **space/vowels vs. consonants**, or capture another structure?  
2. If you **remove spaces** or **lowercase vowels**, how does $\boldsymbol{\phi}_k$ change?  
3. Try $K=3$ and interpret each state’s character distribution.

---

### Option C — (Design Your Own)

Propose a small sequence modeling task relevant to your interests (e.g., **gesture phases** from synthetic accelerometer magnitudes; **toy market regimes** from generated returns). Provide:
- Data-generation description (or data source, if allowed),
- HMM design (states, emissions),
- Training/decoding,
- Evaluation with a meaningful metric (e.g., segmentation purity, predictive log-likelihood).

---

### Required Evaluation & Discussion

1. **Model Selection.** Compare at least two values of $K$; choose $K^\*$ using **per-sequence average log-likelihood** and interpretability.  
2. **Decoding.** Show Viterbi state sequences for at least **three** examples with plots or printed segments.  
3. **Robustness.** Add small observation noise and re-evaluate; discuss changes in state assignments.  
4. **Library vs. Scratch.** Where feasible, replicate training or decoding with your **from-scratch** routines and compare to `hmmlearn` output (note: EM may converge to different local optima).  

**Rubric add-on (20 pts).**  
- (10) Technical completeness: correct data handling, training, decoding, and metrics.  
- (10) Insight: clear state semantics, thoughtful $K$ selection, and honest discussion of model–data fit.

---

# What to Submit

1. Python notebook or `.py` scripts implementing both custom and library-based HMMs.
2. Generated plots of:
   - Hidden vs. observed sequences (synthetic example)
   - Real dataset sequence with HMM state overlay
3. A concise report answering discussion questions and interpreting learned parameters.
4. (Optional) Explore one modification: Gaussian emissions or multi-sequence training.

---
