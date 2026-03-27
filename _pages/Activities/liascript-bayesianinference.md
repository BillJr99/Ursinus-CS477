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

# Part VI Supplement - Bayesian Code Examples

---

## Example 1 — Beta–Binomial Coin Updating (Functional Style)

In this example we encode the classic **coin-flip** Bayesian update:

- The unknown coin bias is a parameter $\theta = P(\text{Heads})$.
- Our **prior belief** about $\theta$ is modeled as a $\text{Beta}(\alpha, \beta)$ distribution.
- Each flip is a Bernoulli observation $X_i \in \{\text{H}, \text{T}\}$.
- After seeing more flips, we **update** $(\alpha, \beta)$ and track the posterior mean
  $$
  \mathbb{E}[\theta \mid \text{data}] = \frac{\alpha}{\alpha + \beta}.
  $$

This code organizes the computations into small, reusable functions and then runs a
short sequence of flips.

---

### Code: `bayesian_coin.py`

```python
# Bayesian updating for a coin's bias using a Beta prior

from typing import List, Tuple

def beta_posterior_mean(alpha: float, beta: float) -> float:
    """Return the mean of a Beta(alpha, beta) distribution."""
    return alpha / (alpha + beta)


def update_beta_after_flip(
    alpha: float,
    beta: float,
    outcome: str
) -> Tuple[float, float]:
    """
    Update Beta(alpha, beta) parameters after a single coin flip.

    outcome: 'H' for heads, 'T' for tails.
    """
    if outcome.upper() == 'H':
        alpha += 1
    elif outcome.upper() == 'T':
        beta += 1
    else:
        raise ValueError(f"Invalid outcome '{outcome}'. Use 'H' or 'T'.")

    return alpha, beta


def run_bayesian_coin_example():
    # Initial prior: Beta(2, 2)
    alpha, beta = 2.0, 2.0

    # Sequence of observed flips
    flips: List[str] = ['H', 'T', 'H', 'H', 'H']

    print("Bayesian updating for a coin with Beta prior")
    print("Initial prior: Beta(alpha=2, beta=2)")
    print()

    # Header for the table
    print(f"{'Step':<5} {'Flip':<6} {'Alpha':<8} {'Beta':<8} {'Mean p(H)':<10}")
    print("-" * 40)

    # Step 0: before seeing any data
    step = 0
    mean_p = beta_posterior_mean(alpha, beta)
    print(f"{step:<5} {'-':<6} {alpha:<8.2f} {beta:<8.2f} {mean_p:<10.3f}")

    # Sequentially update after each flip
    for flip in flips:
        step += 1
        alpha, beta = update_beta_after_flip(alpha, beta, flip)
        mean_p = beta_posterior_mean(alpha, beta)
        print(f"{step:<5} {flip:<6} {alpha:<8.2f} {beta:<8.2f} {mean_p:<10.3f}")


if __name__ == "__main__":
    run_bayesian_coin_example()
```

---

### Conceptual Walkthrough (Beginner-Friendly)

1. **State of belief as $(\alpha, \beta)$**

   - We never store the full continuous posterior distribution explicitly.
   - Instead, we keep track of the **shape parameters** $(\alpha, \beta)$ of the $\text{Beta}$ distribution.
   - These two numbers summarize *all* past flips under the Beta–Binomial conjugate model.

2. **Likelihood and conjugate prior**

   - Each flip $X_i$ is modeled as:
     $$
     X_i \sim \text{Bernoulli}(\theta), \quad \theta \in (0,1).
     $$
   - The prior is $\theta \sim \text{Beta}(\alpha, \beta)$.
   - After observing data with $k$ heads and $n-k$ tails, the posterior is
     $$
     \theta \mid \text{data} \sim \text{Beta}(\alpha + k, \beta + (n-k)).
     $$
   - In this script we integrate observations **one at a time**, updating the parameters sequentially:
     - On heads: $\alpha \leftarrow \alpha + 1$.
     - On tails: $\beta \leftarrow \beta + 1$.

3. **Posterior mean as a point estimate**

   - The function `beta_posterior_mean` returns
     $$
     \mathbb{E}[\theta] = \frac{\alpha}{\alpha + \beta}.
     $$
   - This is a **Bayes estimator** under squared-error loss: it gives the
     action (here, an estimate of $P(\text{Heads})$) that minimizes expected
     squared error with respect to the posterior.

4. **Sequential updating as data arrive**

   - `run_bayesian_coin_example` prints a table with one row per flip:
     - `Step` number.
     - The observed `Flip` (`H` or `T`).
     - The updated $(\alpha, \beta)$.
     - The **current mean** belief $P(\text{Heads})$.
   - You can imagine this as an *online learning* process where each new
     observation refines your belief about $\theta$.

5. **Interpreting the numbers**

   - Starting from $\text{Beta}(2,2)$ (a slightly “U-shaped” but fairly diffuse prior),
     each head increases $\alpha$ and pushes the mean upward.
   - Each tail increases $\beta$ and drags the mean downward.
   - Over many flips, the mean stabilizes near the true bias (if the model is well-specified).

---

## Example 2 — Inferring Coin Bias from Simulated Data

This example uses the same Beta–Binomial ideas but now:

- There is a **hidden true bias** $\theta^\star = P(\text{Heads})$ that generates data.
- We start with an uninformative prior $\text{Beta}(1,1)$, i.e., a uniform prior over $[0,1]$.
- We simulate many flips from the *true* coin and watch our posterior mean converge.

Mathematically:

- Prior: $\theta \sim \text{Beta}(\alpha_0, \beta_0)$ with $(\alpha_0,\beta_0)=(1,1)$.
- Data: $X_i \mid \theta \sim \text{Bernoulli}(\theta)$, independent.
- Posterior after $n$ flips with $k$ heads:
  $$
  \theta \mid X_{1:n} \sim \text{Beta}(\alpha_0 + k, \beta_0 + n - k).
  $$

---

### Code: `bayesian_coin_determine.py`

```python
import random

# True bias of the coin (unknown to the Bayesian learner)
TRUE_P_HEADS = 0.7

# Prior: Beta(α, β)
alpha = 1.0
beta  = 1.0

# Number of flips to simulate
num_flips = 70

print(f"{'Flip #':<7} {'Outcome':<8} {'α':<8} {'β':<8} {'Mean belief P(H)':<18}")
print("-" * 60)

for flip_num in range(1, num_flips + 1):
    # Simulated outcome from the real (biased) coin
    outcome = 'H' if random.random() < TRUE_P_HEADS else 'T'

    # Bayesian update
    if outcome == 'H':
        alpha += 1
    else:
        beta += 1

    mean_belief = alpha / (alpha + beta)
    print(f"{flip_num:<7} {outcome:<8} {alpha:<8.2f} {beta:<8.2f} {mean_belief:<18.3f}")

print("\nFinal inference:")
print(f"Posterior Beta(α={alpha:.2f}, β={beta:.2f})")
print(f"Estimated probability of heads: {alpha/(alpha+beta):.3f}")
print(f"True probability of heads:      {TRUE_P_HEADS:.3f}")
```

---

### Conceptual Walkthrough

1. **True generative process**

   - The line `TRUE_P_HEADS = 0.7` encodes the *ground truth*:
     $$
     P_{\text{true}}(\text{Heads}) = 0.7.
     $$
   - The learner does **not** know this number; it will try to infer it from data.

2. **Prior and posterior parameters**

   - We begin with $\alpha = 1,\ \beta = 1$, i.e., $\text{Beta}(1,1)$.
   - This prior is uniform: it does not favor any value of $\theta$ over another.
   - Each time we see a head, we do $\alpha \leftarrow \alpha + 1$.
   - Each time we see a tail, we do $\beta \leftarrow \beta + 1$.

3. **Posterior mean as running estimate**

   - After each flip, the mean belief is
     $$
     \hat{\theta}_{\text{mean}} = \frac{\alpha}{\alpha + \beta}.
     $$
   - This quantity is printed in the table as `"Mean belief P(H)"`.

4. **Law of large numbers meets Bayesian updating**

   - As `num_flips` grows large, the posterior distribution $\text{Beta}(\alpha,\beta)$
     concentrates around $\theta^\star = 0.7$.
   - The final lines print:
     - The posterior parameters: $\text{Beta}(\alpha_{\text{final}}, \beta_{\text{final}})$.
     - The estimated probability of heads (posterior mean).
     - The true probability of heads used to generate data.

5. **What to look for when running the script**

   - Early on, the mean belief will move around considerably (because the data
     are small and the prior is diffuse).
   - Later, the updates become **incremental**: each new flip only slightly
     nudges the mean.
   - This captures a general Bayesian phenomenon: early data have more influence,
     and as data accumulate, the posterior becomes more concentrated.

---

## Example 3 — Bayesian Disease Testing (Sensitivity, Specificity, and Priors)

This example applies Bayes’ rule to **diagnostic testing**:

- Hidden variable $D \in \{0,1\}$: disease absent/present.
- Observable test result $T \in \{+,-\}$.
- Known test characteristics:
  - Sensitivity: $\text{Se} = P(T{=}+ \mid D{=}1)$.
  - Specificity: $\text{Sp} = P(T{=}- \mid D{=}0)$.
- Prior probability of disease: $\pi = P(D{=}1)$.

Bayes’ rule for a **positive** test result is:
$$
P(D{=}1 \mid T{=}+) = \frac{\text{Se}\,\pi}{\text{Se}\,\pi + (1-\text{Sp})(1-\pi)}.
$$

This code wraps those formulas in a clear API and demonstrates **sequential**
updates across multiple test results.

---

### Code: `bayesian_disease.py`

```python
from dataclasses import dataclass
from typing import List


@dataclass
class TestCharacteristics:
    """
    Characteristics of a diagnostic test.

    sensitivity = P(test is positive | disease is present)
    specificity = P(test is negative | disease is absent)
    """
    sensitivity: float  # True positive rate
    specificity: float  # True negative rate


def update_disease_probability(
    prior_disease: float,
    result: str,
    test: TestCharacteristics
) -> float:
    """
    Update the probability of disease given a new test result.

    prior_disease: P(D) before seeing this test
    result: 'P' or '+' for positive, 'N' or '-' for negative
    test: TestCharacteristics(sensitivity, specificity)

    Returns:
        posterior_disease: P(D | result)
    """
    sens = test.sensitivity
    spec = test.specificity

    if not (0.0 <= prior_disease <= 1.0):
        raise ValueError(f"Invalid prior_disease={prior_disease}. Must be in [0, 1].")

    if result.upper() in ['P', '+']:
        # Positive test
        # P(+ | D) = sensitivity
        # P(+ | ¬D) = 1 - specificity
        p_pos_given_d = sens
        p_pos_given_not_d = 1.0 - spec

        # Total probability of a positive test:
        # P(+) = P(+|D)P(D) + P(+|¬D)P(¬D)
        p_pos = p_pos_given_d * prior_disease + p_pos_given_not_d * (1.0 - prior_disease)

        if p_pos == 0.0:
            raise ZeroDivisionError("P(positive) = 0; cannot update.")

        # Bayes:
        # P(D | +) = P(+|D)P(D) / P(+)
        posterior = (p_pos_given_d * prior_disease) / p_pos

    elif result.upper() in ['N', '-']:
        # Negative test
        # P(- | D) = 1 - sensitivity  (false negative rate)
        # P(- | ¬D) = specificity
        p_neg_given_d = 1.0 - sens
        p_neg_given_not_d = spec

        # Total probability of a negative test:
        # P(-) = P(-|D)P(D) + P(-|¬D)P(¬D)
        p_neg = p_neg_given_d * prior_disease + p_neg_given_not_d * (1.0 - prior_disease)

        if p_neg == 0.0:
            raise ZeroDivisionError("P(negative) = 0; cannot update.")

        # Bayes:
        # P(D | -) = P(-|D)P(D) / P(-)
        posterior = (p_neg_given_d * prior_disease) / p_neg

    else:
        raise ValueError(f"Invalid test result '{result}'. Use 'P', '+', 'N', or '-'.")

    return posterior


def run_disease_testing_example():
    """
    Demonstrate Bayesian updating for disease testing over multiple test results.
    """

    # --- Model setup ---

    # Suppose the disease is relatively rare in the population:
    # prior P(D) = 1%
    prior_disease = 0.01

    # A single type of test with given sensitivity and specificity.
    # Example: sensitivity = 95%, specificity = 98%.
    test = TestCharacteristics(
        sensitivity=0.95,
        specificity=0.98
    )

    # Sequence of test results over time: positive, positive, negative, ...
    # (You can modify this sequence to explore different patterns.)
    test_results: List[str] = ['+', '+', '-']

    # --- Print header and initial state ---
    print("Bayesian Updating for Disease Testing")
    print("------------------------------------")
    print(f"Initial prior probability of disease: {prior_disease:.4f}")
    print(f"Test sensitivity: {test.sensitivity:.2f}")
    print(f"Test specificity: {test.specificity:.2f}")
    print()

    print(f"{'Step':<5} {'Result':<8} {'Posterior P(Disease)':<22}")
    print("-" * 40)

    # Step 0: before any tests
    step = 0
    print(f"{step:<5} {'(prior)':<8} {prior_disease:<22.6f}")

    # --- Sequentially update as each new test result arrives ---
    current_prob = prior_disease
    for result in test_results:
        step += 1
        current_prob = update_disease_probability(current_prob, result, test)
        print(f"{step:<5} {result:<8} {current_prob:<22.6f}")


if __name__ == "__main__":
    run_disease_testing_example()
```

---

### Conceptual Walkthrough

1. **Encoding test characteristics**

   - The `TestCharacteristics` dataclass simply bundles:
     - `sensitivity = P(T{=}+ \mid D{=}1)$
     - `specificity = P(T{=}- \mid D{=}0)$
   - These numbers are **properties of the test**, not of the patient.

2. **Positive vs. negative test updates**

   - For a positive result:
     $$
     P(D{=}1 \mid T{=}+) = \frac{P(T{=}+ \mid D{=}1) P(D{=}1)}
                                {P(T{=}+ \mid D{=}1) P(D{=}1) + P(T{=}+ \mid D{=}0) P(D{=}0)}.
     $$
   - For a negative result:
     $$
     P(D{=}1 \mid T{=}-) = \frac{P(T{=}- \mid D{=}1) P(D{=}1)}
                                {P(T{=}- \mid D{=}1) P(D{=}1) + P(T{=}- \mid D{=}0) P(D{=}0)}.
     $$
   - The function `update_disease_probability` implements these formulas case-by-case.

3. **Base-rate (prior) matters**

   - Even with high sensitivity and specificity, a **rare** disease (small prior $\pi$)
     will usually have a relatively small posterior probability after *one* positive test.
   - Two positive tests in a row may push the posterior quite high.
   - The example sequence `['+', '+', '-']` shows how the posterior climbs and then
     drops after a negative test.

4. **Sequential reasoning**

   - After processing the first test, the resulting posterior becomes the new **prior**
     for the second test, and so on.
   - This mirrors realistic use: each new test refines our belief, rather than “starting over”.

5. **Practical interpretation**

   - The printed table lets you see the **posterior probability of disease** after each step.
   - You can experiment by changing:
     - The prior `prior_disease`.
     - The test accuracy `(sensitivity, specificity)`.
     - The order and type of test results.

---

## Example 4 — Bayesian Pursuit / Stealth Game (Gridworld Belief Tracking)

This larger example (`bayesian_chase.py`) implements an interactive **gridworld game**:

- A **player** moves around a 2D grid using WASD keys.
- An **enemy** tries to find the player but cannot directly observe their position
  except when the player is within a certain **vision radius**.
- Otherwise, the enemy gets **noisy sound cues** about the *direction* of the player.
- The enemy maintains a **belief distribution** over all grid cells:
  $$
  b_t(x) = P(X_t = x \mid \text{observations up to time } t),
  $$
  and updates this belief using a discrete **Bayes filter**.

This is a concrete, game-like illustration of the **Hidden Markov Model / Bayesian filter**
recursion:

- **Predict (motion model)**:
  $$
  \tilde{b}_t(x) = \sum_{x'} P(x \mid x')\, b_{t-1}(x').
  $$
- **Update (observation model)**:
  $$
  b_t(x) \propto P(z_t \mid x)\, \tilde{b}_t(x).
  $$

---

### Code: Core of `bayesian_chase.py` (Bayesian Filter + Game Loop)

For slides, you may wish to show only selected pieces in class; here is a cohesive,
runnable version of the core script that emphasizes the **Bayesian state estimation**
and the enemy’s decision logic.

```python
"""
Real-time grid-based stealth / pursuit demo with Bayesian state estimation.

- Player moves with WASD.
- Enemy maintains a belief distribution over player positions.
- Enemy updates belief using a motion model plus noisy directional sound
  and occasional perfect vision within a radius.
"""

import curses
import random
import time
import math

# ----------------- CONFIGURATION -----------------

WIDTH = 10
HEIGHT = 10

HEARING_RADIUS = 5     # Manhattan distance for hearing
VISION_RADIUS = 4      # Manhattan distance for vision
P_CORRECT_DIR = 0.75   # P(sound direction matches true direction)
RANDOM_MOVE_PROB = 0.1 # Chance enemy moves randomly instead of greedily

TICK_SECONDS = 0.35    # Simulation time step

DIRECTIONS = {
    'w': (-1, 0),  # up
    's': (1, 0),   # down
    'a': (0, -1),  # left
    'd': (0, 1),   # right
}

# ----------------- UTILITY FUNCTIONS -----------------

def in_bounds(r, c):
    """Check whether (r,c) lies inside the grid."""
    return 0 <= r < HEIGHT and 0 <= c < WIDTH


def manhattan(p1, p2):
    """Manhattan (L1) distance between two grid cells."""
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


def normalize(belief):
    """
    Normalize a 2D belief grid so entries sum to 1.
    If the total is 0, fall back to uniform.
    """
    total = sum(sum(row) for row in belief)
    if total == 0:
        val = 1.0 / (WIDTH * HEIGHT)
        return [[val for _ in range(WIDTH)] for _ in range(HEIGHT)]
    return [[cell / total for cell in row] for row in belief]


def make_uniform_belief():
    """Return a uniform prior over all grid cells."""
    val = 1.0 / (WIDTH * HEIGHT)
    return [[val for _ in range(WIDTH)] for _ in range(HEIGHT)]


def motion_update(belief):
    """
    Prediction step: from each cell, the player can
    stay put or move N/S/E/W with equal probability,
    respecting boundaries.
    """
    new_belief = [[0.0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for r in range(HEIGHT):
        for c in range(WIDTH):
            moves = [(r, c)]
            for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
                nr, nc = r + dr, c + dc
                if in_bounds(nr, nc):
                    moves.append((nr, nc))
            p_move = belief[r][c] / len(moves)
            for nr, nc in moves:
                new_belief[nr][nc] += p_move
    return new_belief


def direction_bucket(from_pos, to_pos):
    """
    Map the vector from 'from_pos' to 'to_pos' into
    a coarse direction: 'N', 'S', 'E', or 'W'.
    Return None if positions coincide.
    """
    fr, fc = from_pos
    tr, tc = to_pos
    dr = tr - fr
    dc = tc - fc
    if dr == 0 and dc == 0:
        return None
    if abs(dr) >= abs(dc):
        return 'S' if dr > 0 else 'N'
    else:
        return 'E' if dc > 0 else 'W'


def sample_observed_dir(true_dir):
    """
    Sample a noisy observed direction given the true
    coarse direction. With probability P_CORRECT_DIR
    we return true_dir; otherwise pick a random
    different direction.
    """
    dirs = ['N', 'S', 'E', 'W']
    if true_dir is None:
        return random.choice(dirs)
    if random.random() < P_CORRECT_DIR:
        return true_dir
    others = [d for d in dirs if d != true_dir]
    return random.choice(others)


def sound_update(belief, enemy_pos, observed_dir):
    """
    Correction step: given a directional sound observation,
    update the belief via Bayes:
        posterior(x) ∝ P(observed_dir | x) * prior(x).
    """
    if observed_dir is None:
        return belief

    dirs = ['N', 'S', 'E', 'W']
    n_dirs = len(dirs)
    p_correct = P_CORRECT_DIR
    p_incorrect = (1.0 - P_CORRECT_DIR) / (n_dirs - 1)

    new_belief = [[0.0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
    for r in range(HEIGHT):
        for c in range(WIDTH):
            dir_to_cell = direction_bucket(enemy_pos, (r, c))
            if dir_to_cell is None:
                likelihood = 0.0
            elif dir_to_cell == observed_dir:
                likelihood = p_correct
            else:
                likelihood = p_incorrect
            new_belief[r][c] = belief[r][c] * likelihood

    return normalize(new_belief)


def choose_enemy_move(enemy_pos, target_pos):
    """
    Choose a one-step move for the enemy toward a target cell.
    With probability RANDOM_MOVE_PROB, move randomly instead.
    """
    er, ec = enemy_pos
    if enemy_pos == target_pos:
        return enemy_pos

    # Occasional random move
    if random.random() < RANDOM_MOVE_PROB:
        candidates = []
        for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
            nr, nc = er + dr, ec + dc
            if in_bounds(nr, nc):
                candidates.append((nr, nc))
        return random.choice(candidates) if candidates else enemy_pos

    # Greedy move to reduce Manhattan distance
    best_pos = enemy_pos
    best_dist = manhattan(enemy_pos, target_pos)
    for dr, dc in [(-1,0), (1,0), (0,-1), (0,1)]:
        nr, nc = er + dr, ec + dc
        if in_bounds(nr, nc):
            d = manhattan((nr, nc), target_pos)
            if d < best_dist:
                best_dist = d
                best_pos = (nr, nc)
    return best_pos


def find_belief_peak(belief):
    """Return (cell, probability) of the maximum-belief cell."""
    best_cell = (0, 0)
    best_prob = -1.0
    for r in range(HEIGHT):
        for c in range(WIDTH):
            if belief[r][c] > best_prob:
                best_prob = belief[r][c]
                best_cell = (r, c)
    return best_cell, best_prob

# ----------------- RENDERING WITH CURSES -----------------

def draw_state(stdscr, player_pos, enemy_pos, belief, step, paused):
    """Draw the actual grid and the belief heatmap side-by-side."""
    stdscr.clear()

    peak_cell, peak_prob = find_belief_peak(belief)

    status_line = f"Step: {step}   PAUSED: {'YES' if paused else 'NO'}   q=quit, p=pause"
    stdscr.addstr(0, 0, status_line)

    # Actual world on the left
    start_row = 2
    start_col_left = 0
    stdscr.addstr(start_row - 1, start_col_left,
                  "Actual world (P=player, E=enemy, X=both):")
    for r in range(HEIGHT):
        row_chars = []
        for c in range(WIDTH):
            if (r, c) == player_pos and (r, c) == enemy_pos:
                ch = 'X'
            elif (r, c) == player_pos:
                ch = 'P'
            elif (r, c) == enemy_pos:
                ch = 'E'
            else:
                ch = '.'
            row_chars.append(ch)
        stdscr.addstr(start_row + r, start_col_left, " ".join(row_chars))

    # Belief heatmap on the right
    shades = " .:-=+*#%@"  # low -> high
    start_col_right = 3 + 2 * WIDTH
    stdscr.addstr(start_row - 1, start_col_right,
                  "Enemy belief (heatmap, @ = highest):")

    peak_r, peak_c = peak_cell
    max_prob = peak_prob if peak_prob > 0 else 1e-9

    for r in range(HEIGHT):
        row_chars = []
        for c in range(WIDTH):
            p = belief[r][c]
            level = int((p / max_prob) * (len(shades) - 1))
            level = max(0, min(level, len(shades) - 1))
            ch = shades[level]
            if (r, c) == peak_cell:
                ch = '@'
            row_chars.append(ch)
        stdscr.addstr(start_row + r, start_col_right, " ".join(row_chars))

    bottom_row = start_row + HEIGHT + 1
    stdscr.addstr(bottom_row, 0,
                  f"Player at {player_pos}, Enemy at {enemy_pos}, "
                  f"Belief peak at {peak_cell} (prob ≈ {peak_prob:.3f})")
    stdscr.addstr(bottom_row + 1, 0,
                  "Use WASD to move. Enemy moves each tick based on belief or vision.")
    stdscr.addstr(bottom_row + 2, 0,
                  "If enemy sees you (within vision radius), it chases directly.")

    stdscr.refresh()

# ----------------- MAIN GAME LOOP -----------------

def game_loop(stdscr):
    """Main real-time control loop for the game."""
    curses.curs_set(0)
    stdscr.nodelay(True)
    stdscr.timeout(50)

    # Random initial positions
    player_pos = (random.randrange(HEIGHT), random.randrange(WIDTH))
    enemy_pos = (random.randrange(HEIGHT), random.randrange(WIDTH))
    while enemy_pos == player_pos:
        enemy_pos = (random.randrange(HEIGHT), random.randrange(WIDTH))

    belief = make_uniform_belief()

    step = 0
    paused = False
    caught = False
    last_time = time.time()
    game_loop.prev_player_pos = player_pos

    while True:
        # Handle keyboard input (player movement and commands)
        ch = stdscr.getch()
        if ch != -1:
            key = chr(ch)
            if key in DIRECTIONS:
                dr, dc = DIRECTIONS[key]
                nr, nc = player_pos[0] + dr, player_pos[1] + dc
                if in_bounds(nr, nc):
                    player_pos = (nr, nc)
            elif key == 'p':
                paused = not paused
            elif key == 'q':
                break

        now = time.time()

        # Simulation tick
        if not paused and not caught and (now - last_time) >= TICK_SECONDS:
            last_time = now
            step += 1

            old_player_pos = game_loop.prev_player_pos
            player_moved = (player_pos != old_player_pos)
            game_loop.prev_player_pos = player_pos

            # 1. Predict: motion model
            belief = motion_update(belief)

            # 2. Observation: vision or sound
            visible = manhattan(enemy_pos, player_pos) <= VISION_RADIUS

            if visible:
                # Perfect observation: collapse belief to delta at player_pos
                belief = [[0.0 for _ in range(WIDTH)] for _ in range(HEIGHT)]
                pr, pc = player_pos
                belief[pr][pc] = 1.0
                enemy_pos = choose_enemy_move(enemy_pos, player_pos)
            else:
                # No vision: maybe receive directional sound
                if player_moved and manhattan(enemy_pos, player_pos) <= HEARING_RADIUS:
                    true_dir = direction_bucket(enemy_pos, player_pos)
                    observed_dir = sample_observed_dir(true_dir)
                    belief = sound_update(belief, enemy_pos, observed_dir)
                else:
                    belief = normalize(belief)

                # Move towards MAP estimate (belief peak)
                peak_cell, _ = find_belief_peak(belief)
                enemy_pos = choose_enemy_move(enemy_pos, peak_cell)

            # Capture check
            if enemy_pos == player_pos:
                caught = True

        draw_state(stdscr, player_pos, enemy_pos, belief, step, paused)

        # Small sleep to avoid a busy loop
        time.sleep(0.01)


def main():
    curses.wrapper(game_loop)


if __name__ == "__main__":
    main()
```

---

### Conceptual Walkthrough

1. **Hidden state and belief**

   - Hidden state: $X_t = (r_t, c_t)$, the player’s grid position at time $t$.
   - Belief grid: `belief[r][c]` represents $P(X_t = (r,c) \mid \text{observations up to } t)$.
   - At the start, we use a **uniform prior**:
     $$
     b_0(x) = \frac{1}{\text{WIDTH} \times \text{HEIGHT}}.
     $$

2. **Prediction step (motion model)**

   - The function `motion_update` implements
     $$
     \tilde{b}_t(x) = \sum_{x'} P(x \mid x')\, b_{t-1}(x').
     $$
   - From each cell $x'$, the player can:
     - Stay put, or
     - Move N/S/E/W (subject to bounds),
     all with equal probability.
   - Probabilities are redistributed accordingly.

3. **Observation step: vision vs. sound**

   - **Vision**:
     - If `manhattan(enemy_pos, player_pos) <= VISION_RADIUS`, the enemy “sees” the player.
     - Then $P(z_t \mid x)$ is $1$ if $x$ is the true player position and $0$ otherwise.
     - The belief collapses to a **delta** at the true location.

   - **Sound**:
     - If the player moves and is within `HEARING_RADIUS`, we compute the **true direction**
       from enemy to player (`direction_bucket`).
     - We then sample a noisy observation `observed_dir` from this direction.
     - The likelihood model:
       $$
       P(z_t = d_{\text{obs}} \mid X_t = x) =
       \begin{cases}
       P_{\text{correct}} & \text{if direction\_bucket(enemy, x) = d_{\text{obs}},} \\
       \dfrac{1 - P_{\text{correct}}}{3} & \text{if direction\_bucket(enemy, x) \neq d_{\text{obs}}}, \\
       0 & \text{if positions coincide and no direction is defined.}
       \end{cases}
       $$
     - The function `sound_update` implements:
       $$
       b_t(x) \propto P(z_t \mid x) \tilde{b}_t(x).
       $$

4. **Decision-making from belief**

   - When no vision is available, the enemy chooses an action by:
     - Finding the **belief peak** (MAP estimate): `peak_cell = argmax_x b_t(x)`.
     - Moving one step greedily toward `peak_cell`.
   - When vision is available, the enemy simply moves toward the **true** player position.
   - This illustrates a key idea: actions can be based on **beliefs** over hidden states,
     not only on fully observed states.

5. **Visualization**

   - The left grid shows the **true** positions:
     - `P` for player, `E` for enemy, `X` if both occupy the same cell.
   - The right grid shows a coarse **heatmap** of the belief:
     - Darker characters encode higher probabilities.
     - The cell with the highest probability is marked `@`.

6. **Connections to theory**

   - This is an instance of a **discrete Bayes filter** or **Hidden Markov Model**:
     - State transition model: $P(X_t \mid X_{t-1})$ encoded in `motion_update`.
     - Observation model: $P(Z_t \mid X_t)$ encoded in `sound_update` and the vision check.
   - The game loop continuously applies:
     - Predict $\to$ Update $\to$ Act,
     providing a rich, hands-on example of probabilistic reasoning over time.

---
