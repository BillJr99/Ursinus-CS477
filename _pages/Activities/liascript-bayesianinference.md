# Bayesian Inference and Probabilistic Reasoning
<!--
author:   William M. Mongan
language: en
narrator: US English Male

comment: Render with https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascriptbayesian-inference.md or locally if deployed via https://www.billmongan.com/LiaScript/?https://raw.githubusercontent.com/BillJr99/Ursinus-CS477/gh-pages/_pages/Activities/liascript-bayesianinference.md

import: https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md

link:   https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-4
        https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Bayesian Inference and Probabilistic Reasoning

This module provides a complete, first-principles introduction to **Bayesian inference**, from the algebra of probability to **belief updating**, **conjugate priors**, **posterior predictive inference**, and **sequential state estimation** (localization) with a light introduction to **graphical models**. We will blend formal derivations with small, runnable code cells and domain examples (diagnostic testing and robot localization).

---

## Open Colab: Diagnostic Testing & Belief Updating

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Bayesian_Diagnostic_Test_Belief_Updating.ipynb)

---

## Open Colab: Robot Localization Heatmaps (From Scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/gh-pages/files/notebooks/Bayesian_Robot_Localization_Heatmaps_From_Scratch.ipynb)

---

## Open Colab: Kalman Filtering — Robot Localization (Maze Heatmaps)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Kalman_Filter_Robot_Localization_Maze_Heatmaps.ipynb)

---


## Open Colab: Naïve Bayes (Step-by-Step)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/naive_bayes_step_by_step.ipynb)

---


## 0. Conventions & Utilities

We adopt the following notation:

- Random variables use uppercase (e.g., $X, Y$); realizations use lowercase ($x, y$).
- Densities/mass functions use $p(x)$; conditional probabilities use $p(x \mid y)$.
- Data $\mathcal{D}$; parameters $\theta$; models $\mathcal{M}$.

---

## Code Cell
```python
import math
import numpy as np
from typing import Dict, Callable

np.set_printoptions(precision=4, suppress=True)
print("NumPy ready for small demonstrations.")
```

---

# Part I — Foundations of Probability

## 1. Probability Axioms and Bayes’ Theorem

**Bayes’ rule** for events $A, B$ with $p(B) > 0$:
$$
p(A \mid B) = \frac{p(B \mid A)\, p(A)}{p(B)}.
$$

**Law of total probability**:
$$
p(B) = \sum_i p(B \mid A_i)\, p(A_i),
$$
for a partition $\{A_i\}$.

**Posterior, likelihood, prior, evidence** (for parameters $\theta$ and data $\mathcal{D}$):
$$
p(\theta \mid \mathcal{D}) = \frac{p(\mathcal{D} \mid \theta)\, p(\theta)}{p(\mathcal{D})}, \qquad
p(\mathcal{D}) = \int p(\mathcal{D} \mid \theta)\, p(\theta)\, d\theta.
$$

---

## 2. Updating Beliefs: A Discrete Worked Example

We consider a binary hypothesis $H \in \{0,1\}$ and a test result $T \in \{+, -\}$. Let the **prior** be $p(H=1)=\pi$ and the test’s characteristics be **sensitivity** $\text{Se}=p(T{=}+ \mid H{=}1)$ and **specificity** $\text{Sp}=p(T{=}- \mid H{=}0)$. The posterior after observing $T=+$ is
$$
p(H{=}1 \mid T{=}+) = \frac{\text{Se}\, \pi}{\text{Se}\,\pi + (1-\text{Sp})(1-\pi)}.
$$

---

## Code Cell — Positive Predictive Value (PPV) and Negative Predictive Value (NPV)

```python
def ppv(pi: float, Se: float, Sp: float) -> float:
    return (Se*pi) / (Se*pi + (1-Sp)*(1-pi))

def npv(pi: float, Se: float, Sp: float) -> float:
    return (Sp*(1-pi)) / (Sp*(1-pi) + (1-Se)*pi)

priors = [0.01, 0.1, 0.5]
Se, Sp = 0.95, 0.95
for p0 in priors:
    print(f"Prior={p0:.2f}  PPV={ppv(p0,Se,Sp):.3f}  NPV={npv(p0,Se,Sp):.3f}")
```

*Interpretation.* PPV is strongly affected by the **base rate** $\pi$; even excellent tests can have modest PPV in low-prevalence populations (the base-rate effect).

---

# Part II — Likelihoods, Priors, and Conjugacy

## 3. Bernoulli/Binomial with Beta Prior

Model: i.i.d. $X_i \sim \text{Bernoulli}(\theta)$, with $\theta \in (0,1)$. Prior: $\theta \sim \text{Beta}(\alpha, \beta)$. Observing $k$ successes in $n$ trials gives
$$
p(\theta \mid k,n) = \text{Beta}(\alpha + k, \beta + n - k).
$$

**Posterior mean and MAP:**
$$
\mathbb{E}[\theta \mid k,n] = \frac{\alpha + k}{\alpha + \beta + n}, \qquad
\theta_{\text{MAP}} = \frac{\alpha + k - 1}{\alpha + \beta + n - 2} \quad (\alpha,\beta>1).
$$

---

## Code Cell — Beta–Binomial Updates and Posterior Predictive

```python
import scipy.stats as st

def beta_posterior(alpha, beta, k, n):
    return alpha + k, beta + (n - k)

def posterior_predictive(alpha, beta, m=1):
    # Predictive for m future Bernoulli trials: mean of Bernoulli is E[theta]
    return (alpha) / (alpha + beta)

alpha, beta = 2, 2
k, n = 12, 20
a_post, b_post = beta_posterior(alpha, beta, k, n)
print("Posterior Beta(alpha, beta):", a_post, b_post)
print("Posterior mean:", a_post/(a_post+b_post))
print("Posterior predictive mean for next trial:", posterior_predictive(a_post, b_post))
```

---

## 4. Gaussian Likelihoods and Conjugate Priors

- **Known variance $\sigma^2$** and unknown mean $\mu$ with a **Gaussian prior** $\mu \sim \mathcal{N}(\mu_0, \tau_0^2)$ leads to a **Gaussian posterior**:
$$
\mu \mid \mathcal{D} \sim \mathcal{N}\!\left(\frac{\mu_0/\tau_0^2 + n\bar{x}/\sigma^2}{1/\tau_0^2 + n/\sigma^2}, \; \frac{1}{1/\tau_0^2 + n/\sigma^2}\right).
$$

- **Unknown mean and variance** uses a **Normal–Inverse-Gamma** prior, yielding a Normal–Inverse-Gamma posterior.

---

# Part III — Bayesian Decision Theory

## 5. Loss, Risk, and Bayes Estimators

Given a loss $L(\theta, a)$ and posterior $p(\theta \mid \mathcal{D})$, the **Bayes action** minimizes the **posterior risk**:
$$
a^* \in \arg\min_a \int L(\theta, a) \, p(\theta \mid \mathcal{D})\, d\theta.
$$

- With **squared-error loss** $L(\theta,a)=(\theta-a)^2$, the Bayes estimator is the **posterior mean** $\mathbb{E}[\theta \mid \mathcal{D}]$.
- With **absolute loss**, the Bayes estimator is the **posterior median**.
- With **0–1 loss**, the Bayes estimator is the **MAP**.

---

# Part IV — Conditional Independence & Graphical Models


## 6. Naïve Bayes and Feature Independence (Comprehensive)

For class variable $Y$ and features $X_1,\dots,X_d$, the Naïve Bayes assumption is
$$
p(X_1,\dots,X_d \mid Y) = \prod_{j=1}^d p(X_j \mid Y).
$$
Classification uses
$$
\hat{y} = \arg\max_y \; p(y) \prod_{j=1}^d p(x_j \mid y).
$$

### 6.1 Variants

- **Bernoulli NB** (binary features): $X_j \in \{0,1\}$ with class-conditional Bernoulli parameters $\theta_{jy}$.  
- **Multinomial NB** (counts/words): feature vector $\boldsymbol{x}$ are counts; likelihood
  $$
  p(\boldsymbol{x} \mid y) \propto \prod_{j=1}^d \phi_{jy}^{\,x_j},
  $$
  where $\sum_j \phi_{jy}=1$ for each class $y$.
- **Gaussian NB** (real-valued): each $X_j \mid Y=y \sim \mathcal{N}(\mu_{jy}, \sigma_{jy}^2)$ independently.

### 6.2 Smoothing (Derivation)

For Multinomial NB with a **Dirichlet prior** $\boldsymbol{\phi}_y \sim \text{Dir}(\alpha_1,\dots,\alpha_d)$, the posterior mode/mean yields **add-$\alpha$** (Laplace for $\alpha=1$) estimates:
$$
\hat{\phi}_{jy} = \frac{N_{jy} + \alpha_j}{\sum_{k=1}^d (N_{ky} + \alpha_k)}.
$$
For Bernoulli NB with Beta$(a,b)$ prior,
$$
\hat{\theta}_{jy} = \frac{N^{(1)}_{jy} + a}{N^{(1)}_{jy} + N^{(0)}_{jy} + a + b}.
$$

### 6.3 Log-space Computation

To avoid underflow, compute class scores in log-space:
$$
\log p(y \mid \boldsymbol{x}) \propto \log p(y) + \sum_{j=1}^d x_j \log \phi_{jy} \quad \text{(multinomial)}.
$$

### 6.4 Calibration and Decision Theory

Posterior scores from Naïve Bayes can be **miscalibrated**; apply **Platt scaling** or **isotonic regression** on a validation set. For unequal misclassification costs $C_{\text{FN}}, C_{\text{FP}}$, predict class 1 when
$$
p(Y{=}1 \mid \boldsymbol{x}) \ge \frac{C_{\text{FP}}}{C_{\text{FP}} + C_{\text{FN}}}.
$$

### 6.5 Code Cell — Multinomial Naïve Bayes (Toy Example)

```python
from collections import Counter
import math

docs = [
    ("sports win team game", 1),
    ("team scores goal win", 1),
    ("election debate policy", 0),
    ("policy vote election", 0),
]
# Build vocabulary and counts
vocab = sorted({w for s,_ in docs for w in s.split()})
V = len(vocab)
cls_counts = Counter(y for _,y in docs)
word_counts = {0: Counter(), 1: Counter()}
for s,y in docs:
    for w in s.split():
        word_counts[y][w] += 1

alpha = 1.0  # Laplace smoothing
phi = {y: {} for y in cls_counts}
for y in cls_counts:
    total = sum(word_counts[y].values()) + alpha*V
    for w in vocab:
        phi[y][w] = (word_counts[y][w] + alpha) / total

def log_score(text):
    toks = text.split()
    scores = {}
    for y in cls_counts:
        logp = math.log(cls_counts[y]/sum(cls_counts.values()))
        for w in toks:
            logp += math.log(phi[y].get(w, 1.0/(sum(word_counts[y].values())+alpha*V)))
        scores[y] = logp
    return max(scores, key=scores.get), scores

print(log_score("team win goal"))
print(log_score("policy debate"))
```

### 6.6 Connections to Bayesian Inference

Naïve Bayes is Bayesian when priors over likelihood parameters (Dirichlet/Beta/Gaussian–Inverse-Gamma) are used and predictions marginalize parameters. The common MLE-with-smoothing view corresponds to **MAP** estimation under conjugate priors.


---

## 7. Bayesian Networks (Brief Overview)

A **Bayesian network** is a DAG where nodes are variables and edges encode conditional dependence. The joint factorizes as
$$
p(x_1,\dots,x_n) = \prod_{i=1}^n p(x_i \mid \text{Pa}(X_i)).
$$
Conditional independence can be read via **d-separation**. Exact inference can be done via **variable elimination**; approximate inference via **sampling** (e.g., Gibbs).

---

# Part V — Sequential Bayesian Inference: Localization

## 8. Hidden Markov Models (HMMs) and Belief Recursion

Let latent state $X_t$ and observation $Z_t$. The **Bayesian filter** updates a belief $b_{t-1}(x)=p(x \mid z_{1:t-1})$ to $b_t(x)=p(x \mid z_{1:t})$ via

**Predict:**
$$
\tilde{b}_t(x) = \sum_{x'} p(x \mid x')\, b_{t-1}(x').
$$

**Update:**
$$
b_t(x) \propto p(z_t \mid x)\, \tilde{b}_t(x), \qquad
b_t(x) = \frac{p(z_t \mid x)\, \tilde{b}_t(x)}{\sum_{x''} p(z_t \mid x'')\, \tilde{b}_t(x'')}.
$$

This underlies **robot localization** with grid maps and sensor models, as demonstrated in the Colab notebook (heatmap visualizations of $b_t$).

---

## Code Cell — Discrete Bayes Filter (Gridworld)

```python
# 1D ring grid with wrap-around motion and noisy sensor
N = 10
belief = np.ones(N) / N

# Transition: move +1 with prob 0.8, stay with 0.2
def predict(b):
    p = np.zeros_like(b)
    for i in range(N):
        p[(i+1) % N] += 0.8 * b[i]
        p[i] += 0.2 * b[i]
    return p

# Observation: sensor reports landmark at positions {3,7}; correct with 0.7
landmarks = {3,7}
def likelihood(z, i):
    has = (i in landmarks)
    return 0.7 if (z and has) or ((not z) and (not has)) else 0.3

def update(prior, z):
    unnorm = np.array([likelihood(z, i)*prior[i] for i in range(N)])
    return unnorm / unnorm.sum()

# Perform one predict-update step
belief = predict(belief)
belief = update(belief, z=True)
print("Belief sums to:", belief.sum())
print("Argmax state:", np.argmax(belief))
```

---

## 9. Kalman Filtering (Linear–Gaussian State-Space Models)

When dynamics and observations are **linear** with **Gaussian** noise, the Bayes filter admits a **closed-form** recursion — the **Kalman filter**.

**Model (discrete time):**
$$
x_t = A x_{t-1} + B u_t + w_t, \quad w_t \sim \mathcal{N}(0, Q), \\
z_t = H x_t + v_t, \quad v_t \sim \mathcal{N}(0, R).
$$

We maintain a Gaussian belief $p(x_t \mid z_{1:t}) = \mathcal{N}(\hat{x}_{t\mid t}, P_{t\mid t})$.

### 9.1 Prediction (Time Update)
$$
\hat{x}_{t\mid t-1} = A \hat{x}_{t-1\mid t-1} + B u_t, \qquad
P_{t\mid t-1} = A P_{t-1\mid t-1} A^\top + Q.
$$

### 9.2 Update (Measurement Update)
$$
K_t = P_{t\mid t-1} H^\top (H P_{t\mid t-1} H^\top + R)^{-1},
$$
$$
\hat{x}_{t\mid t} = \hat{x}_{t\mid t-1} + K_t \big(z_t - H \hat{x}_{t\mid t-1}\big), \qquad
P_{t\mid t} = (I - K_t H) P_{t\mid t-1}.
$$

**Derivation sketch.** Combine the Gaussian prior $\mathcal{N}(\hat{x}_{t\mid t-1}, P_{t\mid t-1})$ with the Gaussian likelihood $\mathcal{N}(H x_t, R)$ using the **Gaussian conditioning identity**; the posterior mean and covariance yield the above formulas.

---

## Code Cell — Minimal 1D Kalman Filter

```python
import numpy as np

A, H = 1.0, 1.0
Q, R = 0.01, 0.25
xhat, P = 0.0, 1.0  # prior mean and variance
u = 0.0             # no control

zs = [1.2, 0.9, 1.1, 1.0, 0.95]
for z in zs:
    # Predict
    xhat = A*xhat + 0*u
    P = A*P*A + Q
    # Update
    S = H*P*H + R
    K = P*H / S
    xhat = xhat + K*(z - H*xhat)
    P = (1 - K*H)*P
    print(f"z={z:.2f}  xhat={xhat:.3f}  P={P:.3f}")
```

---

## 9.3 Relation to the Bayes Filter and HMMs

- The Kalman filter is a **special case** of the Bayes filter where beliefs remain Gaussian under linear dynamics/observations.
- An HMM with **continuous Gaussian emissions** and **linear–Gaussian transitions** corresponds to a **Linear Dynamical System (LDS)**; filtering coincides with the Kalman recursion.

**Extensions.**
- **Extended Kalman Filter (EKF):** linearize nonlinear $f,g$ about the current estimate.
- **Unscented Kalman Filter (UKF):** propagate a **sigma-point** set through nonlinearities to capture mean/covariance accurately.

---

## Notebook Integration: Kalman Filtering in Maze Localization

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Kalman_Filter_Robot_Localization_Maze_Heatmaps.ipynb)

This notebook demonstrates:
- Designing $A, H, Q, R$ for a grid/maze with approximate linearized dynamics.
- Visualizing $\hat{x}_{t\mid t}$ and uncertainty $P_{t\mid t}$ as **heatmaps** alongside discrete Bayes filters.
- Comparing **sensor noise** ($R$) vs. **process noise** ($Q$) on convergence and uncertainty.

[[MC]]
Increasing the **measurement noise** $R$ will generally make the Kalman gain $K_t$:
- ( ) Larger, trusting the measurement more.
- (x) Smaller, trusting the prediction more.
- ( ) Unchanged.

---

## 9.4 Practice & Discussion

- **Tuning:** Start with sensor specs for $R$ and motion variance for $Q$; refine via residual analysis.
- **Consistency check:** Innovation $z_t - H \hat{x}_{t\mid t-1}$ should be zero-mean with covariance $S_t$.

> How would you adapt EKF/UKF to handle **range–bearing sensors** with nonlinear observation models?

# Part VI — End-to-End Diagnostic Inference

## 9. From Likelihood Ratios to Posteriors

The **likelihood ratio** is
$$
\Lambda(x) = \frac{p(x \mid H{=}1)}{p(x \mid H{=}0)}.
$$
Posterior **odds** update multiplicatively:
$$
\frac{p(H{=}1 \mid x)}{p(H{=}0 \mid x)} = \Lambda(x) \cdot \frac{p(H{=}1)}{p(H{=}0)}.
$$
Equivalently, in log-odds:
$$
\log \frac{p(H{=}1 \mid x)}{p(H{=}0 \mid x)} = \log \Lambda(x) + \log \frac{\pi}{1-\pi}.
$$

---

## Code Cell — Sequential Evidence Accumulation (Log-Odds)

```python
def update_log_odds(lg_prior, log_lr):
    return lg_prior + log_lr

# Example: three independent pieces of evidence with LRs 3, 0.5, 2
pi = 0.1
lg_prior = math.log(pi/(1-pi))
log_lrs = [math.log(3), math.log(0.5), math.log(2)]
lg_post = lg_prior
for r in log_lrs:
    lg_post = update_log_odds(lg_post, r)
post = math.exp(lg_post) / (1 + math.exp(lg_post))
print("Posterior probability:", round(post,4))
```

---

# Part VII — Model Checking and Predictive Performance

## 10. Posterior Predictive Checks

Given posterior $p(\theta \mid \mathcal{D})$, the **posterior predictive** is
$$
p(x_{\text{new}} \mid \mathcal{D}) = \int p(x_{\text{new}} \mid \theta)\, p(\theta \mid \mathcal{D})\, d\theta.
$$
We compare simulated $x_{\text{new}}$ to observed statistics (means, variances) for **calibration**. Overconfident models are revealed by systematic discrepancies.

---

## 11. Credible Intervals vs. Confidence Intervals

A $100(1-\alpha)\%$ **credible interval** $[a,b]$ satisfies
$$
p(a \le \theta \le b \mid \mathcal{D}) = 1-\alpha.
$$
This is a **probability statement about $\theta$ given data**, unlike frequentist confidence intervals, which concern **procedures** over repeated samples.

---

# Part VIII — Putting It Together

## 12. Workflow Summary

1. **Specify** likelihood $p(\mathcal{D} \mid \theta)$ and prior $p(\theta)$.
2. **Update** to $p(\theta \mid \mathcal{D})$ via Bayes’ rule.
3. **Decide** with a loss function, or **predict** via posterior predictive.
4. **Diagnose** with checks and sensitivity analyses (vary priors, examine robustness).

---

## 13. Sensitivity to Priors

Compare posteriors under multiple plausible priors $p_1(\theta), p_2(\theta)$. If conclusions persist across choices, they are **robust**; if not, report the dependence explicitly.

---

# Part IX — Exercises

1. **Base-rate intuition.** Fix $\text{Se}=0.95,\ \text{Sp}=0.95$ and vary $\pi\in\{0.005,0.05,0.5\}$; compute PPV/NPV and discuss base-rate effects.
2. **Beta–Binomial.** With prior $\text{Beta}(1,1)$ and $k=30$ successes in $n=50$, compute posterior mean, MAP (if defined), and a $95\%$ central credible interval.
3. **Gaussian–Gaussian.** Derive the posterior for $\mu$ with known variance and verify numerically that the posterior mean is a precision-weighted average of $\mu_0$ and $\bar{x}$.
4. **Naïve Bayes.** Implement a multinomial Naïve Bayes text classifier with Laplace smoothing; evaluate on a toy dataset.
5. **Localization.** Extend the Bayes filter to 2D with four-connected motion and a simple range sensor; visualize heatmaps over time.
6. **Posterior predictive check.** For the Beta–Binomial model, generate replicate datasets from the posterior predictive and compare the distribution of totals to the observed count.

---

## 14. Further Reading

- Gelman et al., *Bayesian Data Analysis* (BDA3).
- Murphy, *Machine Learning: A Probabilistic Perspective*.
- Bishop, *Pattern Recognition and Machine Learning* (PRML).
- Thrun, Burgard, and Fox, *Probabilistic Robotics* (Bayes filters, localization).

---

## Open Colab Links 

- Diagnostic Testing & Belief Updating: [Open in Colab](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Bayesian_Diagnostic_Test_Belief_Updating.ipynb)
- Robot Localization Heatmaps (From Scratch): [Open in Colab](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/gh-pages/files/notebooks/Bayesian_Robot_Localization_Heatmaps_From_Scratch.ipynb)
- Naive Bayes: [Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/naive_bayes_step_by_step.ipynb)

---

# Part I Supplement — Bayesian Diagnostic Reasoning (Detailed)

## Bayesian Diagnostics: Reasoning Under Uncertainty

A **diagnostic test** provides uncertain evidence about a hidden condition.
The Bayesian formulation combines *base rates* (priors) and *test characteristics* (likelihoods) to form a posterior belief.

$$
p(H{=}1 \mid T{=}+) = \frac{p(T{=}+ \mid H{=}1) \, p(H{=}1)}{p(T{=}+ \mid H{=}1)\,p(H{=}1) + p(T{=}+ \mid H{=}0)\,p(H{=}0)}.
$$

---

## Visualization and Colab Walkthrough

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Bayesian_Diagnostic_Test_Belief_Updating.ipynb)

This notebook explores:
- Positive and negative predictive values
- Posterior probabilities after single and multiple tests
- Base-rate effects in population screening

---

## Code Snippet Example

```python
pi = 0.01  # prior disease probability
Se, Sp = 0.95, 0.95
post = (Se*pi) / (Se*pi + (1-Sp)*(1-pi))
print(f"Posterior probability given positive test: {post:.3f}")
```

---

## Concept Check

What happens to the posterior probability if prevalence ($\pi$) decreases?

[[MC]]
- ( ) It increases.
- (x) It decreases, even with the same sensitivity and specificity.
- ( ) It remains constant.
- ( ) It becomes 0.5.

---

## Think-Pair-Share

> Why might a positive test in a **low-prevalence** population still not indicate disease presence with high confidence?

---

# Part IV Supplement — Naïve Bayes Detailed Slides

## Naïve Bayes in Depth

We assume conditional independence among features:
$$
p(x_1, \dots, x_d \mid y) = \prod_{j=1}^d p(x_j \mid y).
$$

This assumption simplifies the computation of posteriors and allows tractable parameter estimation.

---

## Notebook Integration

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/naive_bayes_step_by_step.ipynb)

In this Colab:
- Each step of training and prediction for Multinomial Naïve Bayes is derived.
- Visual examples demonstrate classification boundaries and independence assumptions.

---

## Implementation Highlights

```python
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import CountVectorizer

docs = ["team wins match", "player scores goal", "debate election policy"]
y = [1, 1, 0]
vec = CountVectorizer()
X = vec.fit_transform(docs)
nb = MultinomialNB()
nb.fit(X, y)
print(nb.predict(vec.transform(["goal win team"])))
```

---

## Comprehension Quiz

[[MC]]
Which assumption enables Naïve Bayes to simplify the joint likelihood?
- (x) Conditional independence given the class.
- ( ) Mutual exclusivity of features.
- ( ) Equal feature variances.
- ( ) Perfect correlation.

---

## Discussion Prompt

> How can Naïve Bayes perform well in practice even if the independence assumption is violated?

---

# Part V Supplement — Bayesian Robot Localization (Detailed)

## Introduction to Probabilistic Localization

A robot maintains a **belief distribution** $b_t(x)$ over positions $x$ at time $t$.

At each time step:
- **Prediction:** Apply motion model $p(x_t \mid x_{t-1})$.
- **Update:** Incorporate sensor observation $p(z_t \mid x_t)$.

---

## Mathematical Recursion

$$
b_t(x) = \eta \; p(z_t \mid x) \sum_{x'} p(x \mid x') b_{t-1}(x')
$$

where $\eta$ normalizes the belief to sum to one.

---

## Colab Integration: Heatmaps from Scratch

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Bayesian_Robot_Localization_Heatmaps_From_Scratch.ipynb)

The notebook simulates:
- A 1D or 2D grid world
- Motion noise and sensor noise
- Evolving belief maps visualized as heatmaps

---

## Code Example (1D Localization)

```python
import numpy as np
belief = np.ones(10) / 10
motion = np.roll(belief, 1) * 0.8 + belief * 0.2  # prediction
sensor = np.array([0.9 if i==3 else 0.1 for i in range(10)])
belief = sensor * motion
belief /= belief.sum()
print("Most probable location:", np.argmax(belief))
```

---

## Comprehension Quiz

[[MC]]
If the robot’s motion model becomes noisier (higher variance), what happens to the belief distribution?
- (x) It becomes wider and more uncertain.
- ( ) It collapses to a single point.
- ( ) It stays the same.
- ( ) It becomes uniform instantly.

---

## Reflective Prompt

> Explain why localization requires **continuous correction** even with a perfect map.

---

# Summary

These integrated slides tie each conceptual block directly to its corresponding Colab notebook, offering:
- Mathematical rigor
- Algorithmic intuition
- Code-level insight
- Interactive learning checks

