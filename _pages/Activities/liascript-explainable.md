# Explainable and Responsible AI: Shapley Values, Fairness, and Governance
<!--
author:   William M. Mongan
language: en
narrator: US English Male
comment:  Render via https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-explainable.md
import:   https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md
link:     https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-6
           https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Explainable and Responsible AI: Shapley Values, Fairness, and Governance

This module develops the **mathematical foundations of feature attributions** via **Shapley values**, connects them to **model-agnostic explainability** in practice, and situates them within a broader **Responsible AI** program covering fairness, privacy, safety, and governance.

We proceed from **theory $\to$ algorithms $\to$ diagnostics $\to$ policy**, with runnable snippets and in-class checks.

---

## Open Colab: Shapley Values from Scratch on Credit Scoring NN

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/ExplainableAI_Shapley_From_Scratch_CreditNN.ipynb)

---

## 0. Setup & Conventions

- $f:\mathbb{R}^d \to \mathbb{R}$ denotes a prediction function (e.g., score or logit).  
- A data point is $x \in \mathbb{R}^d$ with features indexed by $N = \{1,\dots,d\}$.  
- For a subset $S \subseteq N$, write $x_S$ for the restriction to features in $S$.

---

## Code Cell
```python
import numpy as np
np.set_printoptions(precision=4, suppress=True)
print("Explainability utilities loaded.")
```

---

# Part I — Why Explainability?

## 1. Motivations

- **Trust & debugging:** Identify spurious correlations and data leakage.
- **Regulation:** Right to explanation, model risk governance, auditability.
- **Safety & ethics:** Detect harmful biases; enable human oversight.

**Model-agnostic** interpretability treats the model as a black box, perturbing inputs and observing outputs. **Local** explanations focus on a single instance; **global** explanations summarize across data.

---

# Part II — Shapley Values: Theory

## 2. Cooperative Game View

Construct a game where **players are features**; the **value** of a coalition $S$ is the model output with only features in $S$ present (others “missing”). Let $v(S)$ denote this coalition value.

The **Shapley value** for feature $i$ is
$$
\phi_i(v) = \sum_{S\subseteq N\setminus\{i\}} \frac{|S|!(d-|S|-1)!}{d!} \big[ v(S\cup\{i\}) - v(S) \big].
$$

**Axioms satisfied by Shapley values**
- **Efficiency:** $\sum_i \phi_i = v(N) - v(\emptyset)$
- **Symmetry:** Identical contributors receive equal credit
- **Dummy/Null player:** If $v(S\cup\{i\}) = v(S)$ for all $S$, then $\phi_i=0$
- **Additivity:** Explanations add over sums of games

These map to desirable properties for model explanations.

---

## 3. Choosing the Coalition Value $v(S)$

Common instantiations for a prediction function $f$ at instance $x$:
- **Conditional expectation (interventional):**  
  $$
  v(S) = \mathbb{E}[f(X) \mid X_S = x_S].
  $$
- **Marginal baseline (observational):**  
  $$
  v(S) = \mathbb{E}_{X_{\bar{S}}}[f(x_S, X_{\bar{S}})].
  $$

Choice affects faithfulness when features are **dependent**; conditional baselines preserve relationships but require a generative model.

---

## 4. Exact vs. Approximate Computation

Exact Shapley requires $O(2^d)$ evaluations. Approximations:
- **KernelSHAP:** weighted linear regression to match Shapley constraints.
- **Sampling-based SHAP:** average marginal contributions over random permutations.
- **TreeSHAP:** polynomial-time for tree ensembles via dynamic programming.
- **DeepSHAP/GradientSHAP:** backpropagate attributions using reference distributions.

---

## Code Cell — Monte Carlo Shapley (Didactic)

```python
import itertools, math, numpy as np

def monte_carlo_shapley(f, x, baseline, M=200, rng=None):
    rng = np.random.default_rng(rng)
    d = len(x)
    phi = np.zeros(d)
    for _ in range(M):
        pi = rng.permutation(d)
        x_curr = baseline.copy()
        prev = f(x_curr)
        for j in pi:
            x_next = x_curr.copy()
            x_next[j] = x[j]
            val = f(x_next)
            phi[j] += (val - prev)
            x_curr, prev = x_next, val
    return phi / M

# Example f: weighted sum with ReLU
w = np.array([0.7,-0.6,0.4,0.2])
def f_relu(x): return np.maximum(0, w @ x)
x = np.array([1.0, 0.0, 2.0, -1.0])
baseline = np.zeros_like(x)
phi = monte_carlo_shapley(f_relu, x, baseline, M=1000, rng=0)
print("Approx Shapley:", np.round(phi,3), "Sum:", np.round(phi.sum(),3), "Output shift:", f_relu(x)-f_relu(baseline))
```

---

# Part III — Practical SHAP with Credit Scoring NN

## 5. Notebook Integration: From Scratch & Model-Agnostic SHAP

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/ExplainableAI_Shapley_From_Scratch_CreditNN.ipynb)

This notebook builds a small neural network for credit scoring and implements **Shapley attributions from first principles**. It then compares with **KernelSHAP** or **GradientSHAP** to validate correctness and performance.

**Workflow**
1. Train NN on tabular credit features.  
2. Choose a **baseline/background** dataset (e.g., k‑medoids or training subset).  
3. Compute local Shapley vectors $\phi(x)$ for selected applicants.  
4. Aggregate to **global importance** by $\mathbb{E}[|\phi_i(x)|]$.  
5. Visualize beeswarm/force plots and check consistency with domain knowledge.

---

## Code Cell — KernelSHAP Skeleton (scikit-friendly)

```python
import shap
import numpy as np
# X_train, model fitted earlier; choose a background sample
background = shap.sample(X_train, 200, random_state=0)
explainer = shap.KernelExplainer(model.predict, background)
phi = explainer.shap_values(X_test[:10], nsamples=300)
# Visualization calls in Colab (force, beeswarm) can be added as needed.
print(np.array(phi).shape)
```

**Notes**
- Use **backgrounds** that reflect the population for stable attributions.
- For tree models, prefer **TreeExplainer** for speed/consistency.
- For deep nets, consider **DeepExplainer/GradientSHAP** with appropriate baselines.

---

# Part IV — Validating Explanations

## 6. Faithfulness & Robustness

- **Sanity checks:** randomize model weights; explanations should degrade.  
- **Ablation tests:** remove top‑$k$ features; performance should drop more than removing random features (monotonicity tests).  
- **Stability:** small input perturbations should not wildly change attributions unless the decision boundary is highly nonlinear.

[[MC]]
Which baseline choice can distort Shapley values under **strong feature dependence**?
- (x) Independent marginal baseline $\mathbb{E}_{X_{\bar{S}}}[f(x_S, X_{\bar{S}})]$
- ( ) Conditional $\mathbb{E}[f(X)\mid X_S=x_S]$
- ( ) Neither baseline has any effect

---

# Part V — Responsible AI: Fairness, Privacy, and Governance

## 7. Fairness Concepts & Metrics

- **Group fairness:** demographic parity, equalized odds, equal opportunity.  
- **Individual fairness:** similar individuals should receive similar predictions.  
- **Calibration:** predicted probabilities reflect observed frequencies.

**Metrics**
- **Demographic parity difference:**  
  $$
  \mathrm{DP} = |\Pr(\hat{Y}{=}1\mid A{=}0) - \Pr(\hat{Y}{=}1\mid A{=}1)|.
  $$
- **Equalized odds gap:** differences in TPR/FPR across groups.

**Interventions**
- Pre‑processing (reweighing), in‑processing (fairness constraints), post‑processing (threshold adjustments).

---

## 8. Privacy & Safety

- **Data governance:** lineage, retention, consent, access control.  
- **Differential Privacy (DP) basics:** $(\varepsilon,\delta)$‑DP bounds influence of any one record.  
- **Robustness & security:** distribution shift monitoring, adversarial example awareness, red‑teaming.

---

## 9. Documentation & Accountability

- **Model cards** and **data sheets** describing intended use, limitations, and performance slices.  
- **Human‑in‑the‑loop** review for high‑risk decisions.  
- **Change management**: versioning models and explanation pipelines; audit trails.

---

# Part VI — Case Study Workflow (Credit)

1. **Define** use case and fairness constraints (e.g., equal opportunity).  
2. **Measure** group metrics on validation slices; compare with business/ethical thresholds.  
3. **Explain** decisions locally with Shapley; identify top drivers per group.  
4. **Mitigate** with reweighing or constraint tuning; **re‑explain** to verify reduction in disparate impact.  
5. **Govern** with documentation, human review triggers, and ongoing monitoring.

---

# Part VII — Exercises

1. Implement Monte Carlo Shapley on a logistic model; compare with KernelSHAP.  
2. Evaluate fairness metrics across sensitive attributes; propose mitigations.  
3. Test explanation stability under small Gaussian noise on inputs.  
4. Create a **model card** summarizing use, risks, and monitoring plans for the credit model.

---

## Open Colab Links 

- Shapley Values — Credit Scoring NN (From Scratch): [Open in Colab](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/ExplainableAI_Shapley_From_Scratch_CreditNN.ipynb)
