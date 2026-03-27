---
layout: assignment
permalink: Assignments/BayesianInference
title: "Assignment: Bayesian Inference & Naïve Bayes — From Scratch and with Libraries"

info:
  points: 100
  goals:
    - Develop conceptual and computational fluency with Bayesian inference for simple models.
    - Implement Bayesian updating and posterior prediction from scratch with clear math.
    - Apply Naïve Bayes for classification on a small dataset; compare MAP/MLE estimates.
    - Perform posterior predictive checks and discuss calibration, robustness, and priors.
  purpose: "This assignment connects the mathematics of Bayes' rule to practical modeling. You will derive conjugate posteriors, implement Bayesian updates and predictions, and build a Naïve Bayes classifier from scratch, then compare to scikit-learn."
  concepts:
    - Bayes' rule, prior/likelihood/posterior, evidence
    - Conjugacy (Beta–Binomial, Gaussian–Gaussian)
    - Posterior predictive distributions
    - Naïve Bayes (Bernoulli/Multinomial) with smoothing as MAP estimation
    - Calibration and posterior predictive checks
  tasks:
    - Implement Beta–Binomial and Gaussian–Gaussian updating and prediction.
    - Build a Naïve Bayes classifier (from scratch) for a toy text dataset.
    - Use scikit-learn to reproduce and compare results; analyze differences.
    - Conduct sensitivity analyses over priors and smoothing.
  rubric:
    - weight: 30
      description: Implementation
      preemerging: Partial functions for updates or prediction; fragile I/O.
      beginning: Correct posterior updates for at least one conjugate pair.
      progressing: Complete Beta–Binomial and Gaussian–Gaussian with tests and plots/tables.
      proficient: Clean, reusable code with checks, docstrings, and insightful visual summaries.
    - weight: 30
      description: Mathematical Correctness and Reasoning
      preemerging: Basic formulas without derivations.
      beginning: Correct derivations for one model; limited commentary.
      progressing: Clear derivations for both models; interprets parameters and credible intervals.
      proficient: Thorough reasoning, edge cases, prior sensitivity, and predictive interpretation.
    - weight: 20
      description: Naïve Bayes Modeling & Analysis
      preemerging: Minimal working NB; default settings.
      beginning: Implements smoothing and compares to scikit-learn.
      progressing: Performs calibration or threshold analysis; explains independence assumption.
      proficient: Deep dive on features/priors, error analysis, and robustness to shift.
    - weight: 10
      description: Code Quality & Documentation
      preemerging: Sparse comments.
      beginning: Basic docstrings and structure.
      progressing: Modular functions, type hints, and neat reporting.
      proficient: Excellent organization, reproducibility, and clarity.
    - weight: 10
      description: Submission Completeness
      preemerging: Missing artifacts or instructions.
      beginning: Includes code and brief report.
      progressing: Includes code, results, and discussion.
      proficient: Fully reproducible with seeds, configs, and instructions.

tags:
  - ai
  - bayesian
  - probabilistic-models
  - classification
---

# Overview

You will implement **Bayesian inference** for two classic conjugate models and build a **Naïve Bayes** classifier for a small text dataset. First, you will perform derivations and compute posteriors and posterior predictives **from scratch**. Then, you will reproduce results using **libraries** (e.g., `scipy`, `scikit-learn`) and analyze differences.

---

## Stage 0 — Setup

```python
import numpy as np
from dataclasses import dataclass
from typing import Tuple

# Libraries for later stages
import scipy.stats as st
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB, BernoulliNB
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
np.set_printoptions(precision=4, suppress=True)
```

---

## Stage 1 — Beta–Binomial: Derivation and Implementation (From Scratch)

We model binary outcomes <span>\\(X_i \sim \mathrm{Bernoulli}(\theta)\\)</span> with prior <span>\\(\theta \sim \mathrm{Beta}(\alpha, \beta)\\)</span>. Given <span>\\(k\\)</span> successes in <span>\\(n\\)</span> trials, the posterior is
$$
\theta \mid k,n \sim \mathrm{Beta}(\alpha + k,\; \beta + n - k).
$$

**Posterior mean and MAP:**
$$
\mathbb{E}[\theta \mid k,n] = \frac{\alpha + k}{\alpha + \beta + n},\quad
\theta_{\mathrm{MAP}} = \frac{\alpha + k - 1}{\alpha + \beta + n - 2}\; (\alpha,\beta>1).
$$

**Posterior predictive** for the next Bernoulli trial:
$$
p(X_{n+1}=1 \mid k,n) = \mathbb{E}[\theta \mid k,n] = \frac{\alpha + k}{\alpha + \beta + n}.
$$

```python
@dataclass
class BetaBinomial:
    alpha: float
    beta: float
    k: int
    n: int

    def posterior_params(self) -> Tuple[float,float]:
        return self.alpha + self.k, self.beta + (self.n - self.k)

    def posterior_mean(self) -> float:
        a,b = self.posterior_params()
        return a / (a + b)

    def map_estimate(self) -> float:
        a,b = self.posterior_params()
        if a > 1 and b > 1:
            return (a - 1) / (a + b - 2)
        return np.nan  # undefined for a<=1 or b<=1

    def posterior_predictive_next(self) -> float:
        return self.posterior_mean()

bb = BetaBinomial(alpha=2.0, beta=2.0, k=30, n=50)
print("Posterior (a,b):", bb.posterior_params())
print("Posterior mean:", bb.posterior_mean())
print("MAP:", bb.map_estimate())
print("p(X_{n+1}=1 | data):", bb.posterior_predictive_next())
```

**Checkpoint:** Vary the prior <span>\\((\alpha,\beta)\\)</span> to see how posteriors shift with small <span>\\(n\\)</span> (prior sensitivity).

---

## Stage 2 — Gaussian Mean with Known Variance (From Scratch)

Assume <span>\\(X_i \sim \mathcal{N}(\mu, \sigma^2)\\)</span> with known <span>\\(\sigma^2\\)</span>, and prior <span>\\(\mu \sim \mathcal{N}(\mu_0, \tau_0^2)\\)</span>. For <span>\\(n\\)</span> observations with mean <span>\\(\bar{x}\\)</span>, the posterior is
$$
\mu \mid \mathbf{x} \sim \mathcal{N}\!\left(\frac{\mu_0/\tau_0^2 + n\bar{x}/\sigma^2}{1/\tau_0^2 + n/\sigma^2},\; \frac{1}{1/\tau_0^2 + n/\sigma^2}\right).
$$

**Posterior predictive** for a new draw is
$$
X_{\mathrm{new}} \mid \mathbf{x} \sim \mathcal{N}\!\left(\mu_{\text{post}},\; \sigma^2 + \tau_{\text{post}}^2\right).
$$

```python
def gaussian_posterior(mu0, tau0_sq, sigma_sq, x):
    n = len(x); xbar = np.mean(x)
    prec0 = 1.0/tau0_sq; prec = 1.0/sigma_sq
    post_var = 1.0 / (prec0 + n*prec)
    post_mean = post_var * (prec0*mu0 + n*prec*xbar)
    return post_mean, post_var

rng = np.random.default_rng(0)
true_mu, sigma = 1.5, 1.0
x = rng.normal(true_mu, sigma, size=30)
mu_post, var_post = gaussian_posterior(mu0=0.0, tau0_sq=4.0, sigma_sq=sigma**2, x=x)
print("Posterior mean:", mu_post, "Posterior var:", var_post)
# Posterior predictive variance = sigma^2 + var_post
```

**Checkpoint:** Show how the posterior mean moves from <span>\\(\mu_0\\)</span> toward <span>\\(\bar{x}\\)</span> as <span>\\(n\\)</span> grows (precision-weighted average).

---

## Stage 3 — Posterior Predictive Checks (PPC)

Simulate replicated datasets from the posterior predictive and compare summary statistics to the observed data.

```python
def ppc_gaussian(mu_post, var_post, sigma_sq, T=500, n=30, rng=0):
    rng = np.random.default_rng(rng)
    sims = []
    for _ in range(T):
        mu_sim = rng.normal(mu_post, np.sqrt(var_post))
        x_new = rng.normal(mu_sim, np.sqrt(sigma_sq), size=n)
        sims.append([np.mean(x_new), np.var(x_new)])
    return np.array(sims)

sims = ppc_gaussian(mu_post, var_post, sigma_sq=sigma**2, n=len(x))
obs_stats = np.array([np.mean(x), np.var(x)])
pval_mean = (sims[:,0] <= obs_stats[0]).mean()
pval_var  = (sims[:,1] <= obs_stats[1]).mean()
print("PPC p-values (mean, var):", (pval_mean, pval_var))
```

**Interpretation:** Extreme p-values may indicate model–data mismatch (e.g., heavy tails).

---

## Stage 4 — Naïve Bayes (From Scratch and scikit-learn)

We study **Multinomial Naïve Bayes** for text classification. Under class <span>\\(y \in \{0,1\}\\)</span> and word counts <span>\\(\mathbf{x}\\)</span>, the class-conditional is
$$
p(\mathbf{x}\mid y) \propto \prod_{j=1}^d \phi_{jy}^{\,x_j}, \quad \sum_j \phi_{jy}=1.
$$

With a Dirichlet prior <span>\\(\boldsymbol{\phi}_y \sim \mathrm{Dir}(\alpha,\dots,\alpha)\\)</span>, the MAP estimate yields **add-<span>\\(\alpha\\)</span> smoothing**:
$$
\hat{\phi}_{jy} = \frac{N_{jy} + \alpha}{\sum_{k=1}^d (N_{ky} + \alpha)}.
$$

### 4.1 Toy Corpus and From-Scratch NB

```python
docs = [
    ("team wins match", 1), ("player scores goal", 1), ("election debate policy", 0),
    ("policy vote election", 0), ("team scores again", 1), ("debate team policy", 0)
]
corpus, y = zip(*docs); y = np.array(y)
vec = CountVectorizer()
X = vec.fit_transform(corpus).toarray()
V = X.shape[1]; alpha = 1.0

# Estimate class priors
pi = np.bincount(y) / len(y)

# Estimate class-conditional word probs with add-alpha
phi = np.zeros((2, V))
for c in [0,1]:
    counts = X[y==c].sum(axis=0)
    phi[c] = (counts + alpha) / (counts.sum() + alpha*V)

# Predict in log-space
def predict_nb(Xrow):
    logp = np.log(pi.copy())
    for c in [0,1]:
        logp[c] += (Xrow * np.log(phi[c] + 1e-12)).sum()
    return int(np.argmax(logp))

yhat = np.array([predict_nb(x) for x in X])
print("From-scratch accuracy:", (yhat==y).mean())
```



#### 4.1b From-Scratch NB on Tiny **Spam vs. Ham**

We construct a miniature SMS-like corpus to evaluate Naïve Bayes on a **spam/ham** task. This emphasizes tokenization, smoothing, and independence assumptions.

```python
sms_docs = [
    ("win cash now free entry", 1),
    ("urgent claim prize call now", 1),
    ("free tickets win now", 1),
    ("lets meet for lunch", 0),
    ("are you coming to class", 0),
    ("see you at the game", 0),
    ("free lunch offer today", 1),
    ("can we call later", 0)
]
corpus_sms, y_sms = zip(*sms_docs)
y_sms = np.array(y_sms)

vec_sms = CountVectorizer()
X_sms = vec_sms.fit_transform(corpus_sms).toarray()
V_sms = X_sms.shape[1]
alpha_sms = 1.0

# Class priors
pi_sms = np.bincount(y_sms) / len(y_sms)

# Class-conditional word probabilities (add-alpha smoothing)
phi_sms = np.zeros((2, V_sms))
for c in [0, 1]:
    counts = X_sms[y_sms == c].sum(axis=0)
    phi_sms[c] = (counts + alpha_sms) / (counts.sum() + alpha_sms * V_sms)

def predict_nb_sms(xrow):
    logp = np.log(pi_sms.copy())
    for c in [0, 1]:
        logp[c] += (xrow * np.log(phi_sms[c] + 1e-12)).sum()
    return int(np.argmax(logp))

yhat_sms = np.array([predict_nb_sms(x) for x in X_sms])
print("From-scratch SMS spam/ham accuracy:", (yhat_sms == y_sms).mean())
```

**Checkpoint:** Vary the smoothing parameter <span>\\(\\alpha \\in \\{0.1, 1.0, 2.0\\}\\)</span> and identify which class benefits most (spam often hinges on rare tokens like “win”, “free”, “prize”). Discuss the impact on **precision/recall** for the spam class.
### 4.2 scikit-learn NB and Comparison

```python
Xtr, Xte, ytr, yte = train_test_split(corpus, y, test_size=0.4, random_state=0, stratify=y)
vec = CountVectorizer()
XtrB = vec.fit_transform(Xtr)
XteB = vec.transform(Xte)

clf = MultinomialNB(alpha=1.0)
clf.fit(XtrB, ytr)
yhat_sk = clf.predict(XteB)
print(classification_report(yte, yhat_sk, digits=3))

# Confusion matrix
cm = confusion_matrix(yte, yhat_sk)
ConfusionMatrixDisplay(cm, display_labels=["class 0","class 1"]).plot()
plt.title("Naïve Bayes — Confusion Matrix")
plt.show()
```

**Checkpoint:** Vary <span>\\(\alpha \in \{0.1, 1.0, 2.0\}\\)</span> and discuss the bias–variance trade-off and rare-word handling.


#### 4.2b Medical Test Scenario with scikit-learn (**Breast Cancer**)

We illustrate **base-rate effects** and calibration on a medical dataset using `load_breast_cancer` from scikit-learn.

Let the positive class be <span>\\(Y{=}1\\)</span> (malignant). The **base rate** <span>\\(\\pi = \\Pr(Y{=}1)\\)</span> influences post-test probabilities even with strong likelihood evidence.

```python
from sklearn.datasets import load_breast_cancer
from sklearn.calibration import CalibratedClassifierCV
import numpy as np

data = load_breast_cancer()
X_med = data.data
y_med = data.target  # 0=malignant, 1=benign in sklearn; flip to make 1=malignant
y_med = 1 - y_med

Xtr_m, Xte_m, ytr_m, yte_m = train_test_split(X_med, y_med, test_size=0.3, random_state=0, stratify=y_med)

# BernoulliNB assumes binary features; MultinomialNB assumes counts.
# For continuous features, GaussianNB is more appropriate:
from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(Xtr_m, ytr_m)
yhat_m = gnb.predict(Xte_m)
print(classification_report(yte_m, yhat_m, digits=3))

# Calibration (Platt scaling via sigmoid) for better probability estimates
cal_gnb = CalibratedClassifierCV(GaussianNB(), method="sigmoid", cv=5)
cal_gnb.fit(Xtr_m, ytr_m)
probs = cal_gnb.predict_proba(Xte_m)[:, 1]  # P(Y=1 | X)
print("Base rate (train) π:", ytr_m.mean())
print("Predicted P(Y=1) mean:", probs.mean())
```

**Interpretation:** Even with strong evidence (high or low scores), the **posterior** depends on the **prior/base rate** <span>\\(\\pi = \\Pr(Y{=}1)\\)</span>. Plotting a **reliability curve** or comparing predicted probabilities by deciles checks calibration.

**Optional:** Compute **PPV/NPV** at different thresholds <span>\\(\\tau\\)</span> and show how changing <span>\\(\\pi\\)</span> (via class weighting or sample rebalancing) shifts operating points.

---

## Stage 5 — Calibration and Decision Thresholds

For decision support, scores should be calibrated. While MultinomialNB outputs log-probabilities under the model assumptions, real-world data may violate independence and affect calibration. Consider isotonic/Platt scaling on a validation set (optional). Discuss how thresholding changes **precision/recall**.

---

## Stage 6 — Sensitivity & Robustness

1. **Prior sensitivity (Beta–Binomial):** Compare posteriors for <span>\\((\alpha,\beta) \in \{(1,1), (2,2), (5,5)\}\\)</span> when <span>\\(n\\)</span> is small (e.g., <span>\\(n=10\\)</span>).  
2. **Model misspecification (Gaussian):** Re-simulate from a heavy-tailed distribution and repeat PPCs. What changes?  
3. **NB smoothing:** Grid over <span>\\(\alpha\\)</span> and evaluate accuracy; report the best value and discuss overfitting vs. underfitting.

---

---

## Stage 7 — Creative Mini-Project: Design & Solve a New Bayesian Problem

In this stage, you will **define and solve your own Bayesian inference problem**. Choose **one** of the options below (or propose your own), implement it end-to-end, and explain your modeling decisions. Your solution must include: (i) a clearly stated modeling goal, (ii) model specification (likelihood, prior, and any hierarchy), (iii) posterior computation (analytical or approximate), (iv) posterior predictive checks, and (v) a decision or interpretation grounded in the posterior.

> **Deliverables:** A short write-up and runnable code cells. Reuse your earlier functions where appropriate (e.g., Beta–Binomial updates), and then **validate** or extend using a library (e.g., `scipy`, or a probabilistic library such as `pymc` if available).

---

### Option A — Bayesian A/B Testing with Sequential Stopping

We compare two conversion rates <span>\\(\\theta_A, \\theta_B\\)</span> with independent Beta priors and Bernoulli likelihoods.

**Model.**
- Prior: <span>\\(\\theta_A, \\theta_B \\sim \\mathrm{Beta}(\\alpha_0, \\beta_0)\\)</span>  
- Data: <span>\\(X_{i}^{(g)} \\sim \\mathrm{Bernoulli}(\\theta_g)\\)</span> for group <span>\\(g\\in\\{A,B\\}\\)</span>

**Posterior.**
- <span>\\(\\theta_g \\mid \\text{data} \\sim \\mathrm{Beta}(\\alpha_0 + k_g,\\; \\beta_0 + n_g - k_g)\\)</span>

**Decision question.** Compute <span>\\(\\Pr(\\theta_A > \\theta_B \\mid \\text{data})\\)</span> and recommend the better variant if the probability exceeds a threshold (e.g., <span>\\(0.95\\)</span>). Include **sequential stopping** logic.

```python
import numpy as np
from numpy.random import default_rng
rng = default_rng(0)

def post_params(alpha0, beta0, k, n):
    return alpha0 + k, beta0 + (n - k)

def prob_A_better(alphaA, betaA, alphaB, betaB, draws=200_000, seed=0):
    rng = default_rng(seed)
    thA = rng.beta(alphaA, betaA, size=draws)
    thB = rng.beta(alphaB, betaB, size=draws)
    return float((thA > thB).mean())

# Example scaffold
alpha0, beta0 = 1.0, 1.0
kA, nA = 45, 400
kB, nB = 62, 420
aA, bA = post_params(alpha0, beta0, kA, nA)
aB, bB = post_params(alpha0, beta0, kB, nB)
p = prob_A_better(aA, bA, aB, bB)
print("Pr(theta_A > theta_B | data) =", round(p, 4))
# Add: sequential data accumulation and stopping rule at p > 0.95.
```

**PPC.** Simulate posterior predictive conversions and verify observed lifts are plausible under the joint posterior.

**Questions.**
1. How sensitive is the recommendation to the prior <span>\\((\\alpha_0,\\beta_0)\\)</span>?  
2. Under asymmetric loss (e.g., false-win cost <span>\\(\\lambda\\)</span>), how would you change the stopping threshold?

---

### Option B — Gaussian Mean **Change-Point** Detection (Conjugate Inference)

Detect a single change point <span>\\(\\tau\\)</span> in a sequence of Gaussian observations with known variance <span>\\(\\sigma^2\\)</span>:
$$
X_t \sim \mathcal{N}(\mu_1, \sigma^2) \quad \text{for } t \le \tau,
\qquad
X_t \sim \mathcal{N}(\mu_2, \sigma^2) \quad \text{for } t > \tau,
\qquad
\mu_1,\mu_2 \sim \mathcal{N}(\mu_0, \tau_0^2),\quad
\tau \sim \text{Uniform}\{1,\ldots,T-1\}.
$$

**Scaffold.** Use conjugate Gaussian updates to compute the **marginal likelihood** for each proposed <span>\\(\\tau\\)</span> by integrating out <span>\\(\\mu_1,\\mu_2\\)</span>, then evaluate the posterior over <span>\\(\\tau\\)</span>.

```python
import numpy as np

def log_marginal_mean_known_var(x, mu0, tau0_sq, sigma_sq):
    n = len(x); xbar = x.mean()
    prec0 = 1.0/tau0_sq; prec = 1.0/sigma_sq
    post_var = 1.0 / (prec0 + n*prec)
    # log evidence up to additive constants:
    return -0.5*np.log(prec0 + n*prec) - 0.5*prec*n*(xbar - mu0)**2 + 0.5*(mu0**2*prec0 + n*prec*xbar**2)*post_var

def posterior_tau(x, mu0=0.0, tau0_sq=10.0, sigma_sq=1.0):
    T = len(x)
    logs = []
    for tau in range(1, T):  # split at tau
        l1 = log_marginal_mean_known_var(x[:tau], mu0, tau0_sq, sigma_sq)
        l2 = log_marginal_mean_known_var(x[tau:], mu0, tau0_sq, sigma_sq)
        logs.append(l1 + l2)
    logs = np.array(logs)
    logs -= logs.max()
    p = np.exp(logs); p /= p.sum()
    return p  # length T-1

# Example: synthesize a change
rng = np.random.default_rng(1)
x = np.r_[rng.normal(0.0, 1.0, 60), rng.normal(1.0, 1.0, 40)]
p_tau = posterior_tau(x, mu0=0.0, tau0_sq=10.0, sigma_sq=1.0)
print("MAP tau (1..T-1) =", int(np.argmax(p_tau)+1))
```

**PPC.** Sample <span>\\(\\tau, \\mu_1, \\mu_2\\)</span> from their posteriors and simulate sequences; compare mean shifts to observed.

**Questions.**
1. How does uncertainty in <span>\\(\\sigma^2\\)</span> affect results? Extend with an **Inverse-Gamma** prior if time permits.  
2. Compare MAP <span>\\(\\tau\\)</span> to a frequentist CUSUM or two-sample <span>\\(t\\)</span> test baseline.

---

### Option C — Hierarchical Beta–Binomial **Partial Pooling** (Multi-Group)

Pool information across groups (e.g., small clinics’ success rates) using a hierarchical prior:

<span>\\(\\theta_i \\mid \\alpha,\\beta \\sim \\mathrm{Beta}(\\alpha,\\beta),\\quad k_i \\mid \\theta_i \\sim \\mathrm{Binomial}(n_i, \\theta_i)\\)</span>,

with hyperprior on <span>\\((\\alpha,\\beta)\\)</span> (or empirical Bayes).

**Scaffold (Empirical Bayes).** Estimate <span>\\((\\alpha,\\beta)\\)</span> by matching moments across groups, then compute group posteriors.

```python
import numpy as np

# toy data: successes k_i out of n_i
k = np.array([3, 15, 2, 20, 7, 1])
n = np.array([10, 30, 5, 40, 15, 4])

p_hat = k / n
m = p_hat.mean(); v = p_hat.var()
# moment matching for Beta: alpha,beta > 0, v = ab / ((a+b)^2 (a+b+1))
A = m*(1-m)/v - 1
alpha_hat = m*A; beta_hat = (1-m)*A
print("alpha_hat, beta_hat =", round(alpha_hat,2), round(beta_hat,2))

# group-level posteriors
alpha_post = alpha_hat + k
beta_post  = beta_hat + (n - k)
theta_post_mean = alpha_post / (alpha_post + beta_post)
print("Posterior means:", np.round(theta_post_mean, 3))
```

**Questions.**
1. Compare **no pooling** (independent Beta priors with weak hyperparameters) vs. **partial pooling** above.  
2. Which groups shrink most toward the global mean, and why?

---

### Option D — Naïve Bayes with **Asymmetric Loss** and **Calibration**

Extend your Naïve Bayes to a deployment setting with asymmetric misclassification costs <span>\\((C_{\\text{FP}}, C_{\\text{FN}})\\)</span>. Choose a validation set and a decision threshold

<span>\\(\\tau^* = \\frac{C_{\\text{FP}}}{C_{\\text{FP}} + C_{\\text{FN}}}\\)</span>

for classifying <span>\\(\\Pr(Y{=}1\\mid x) \\ge \\tau^*\\)</span>. Calibrate scores (e.g., isotonic regression) and report **cost-sensitive metrics**.

**Scaffold.**
```python
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import confusion_matrix

# suppose y_score are uncalibrated NB scores for class 1
iso = IsotonicRegression(out_of_bounds="clip")
iso.fit(y_score_val, y_val)       # fit on validation
y_score_cal = iso.transform(y_score_test)

C_FP, C_FN = 1.0, 5.0
tau = C_FP / (C_FP + C_FN)
y_pred = (y_score_cal >= tau).astype(int)

cm = confusion_matrix(y_test, y_pred)
print("Threshold tau*", round(tau,3), "Confusion:\n", cm)
# Compute expected cost = C_FP * FP + C_FN * FN
```

**Questions.**
1. How does calibration change your **expected cost** under <span>\\((C_{\\text{FP}}, C_{\\text{FN}})\\)</span>?  
2. Report a **decision curve** by sweeping <span>\\(\\tau\\)</span>; where is your operating point?

---

### Required Evaluation & Discussion

1. **Model choice.** Justify your likelihood and prior(s) and any independence assumptions.  
2. **Posterior & PPC.** Report key posterior summaries (means/credible intervals) and at least one **posterior predictive** diagnostic.  
3. **Sensitivity.** Show a prior sensitivity analysis for at least one parameter (e.g., vary Beta hyperparameters or hierarchical prior strength).  
4. **Decision.** If applicable, translate posterior results into a decision under a specified loss or threshold.  

**Rubric add-on (20 pts).**  
- (10) Technical completeness: correct model statement, posterior computation, and PPC.  
- (10) Insight: defensible modeling choices, clear interpretation, and honest discussion of limitations.

---

# What to Submit

1. A notebook or scripts implementing:  
   - Beta–Binomial and Gaussian–Gaussian updates and predictions **from scratch**.  
   - Naïve Bayes **from scratch** and with **scikit-learn**.  
2. Plots/tables summarizing posterior parameters and predictive checks.  
3. A concise report (2–3 pages) with derivations, sensitivity analyses, calibration discussion, and conclusions.  
4. Reproducibility: fix random seeds and describe data preprocessing steps.

---
