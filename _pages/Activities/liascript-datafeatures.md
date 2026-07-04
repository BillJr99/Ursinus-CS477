# Thinking in Data and Features: A Machine Learning Overview
<!--
author:   William M. Mongan
language: en
narrator: US English Male
comment:  Render via https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-datafeatures.md
import:   https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md
link:     https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-5
           https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Thinking in Data and Features: A Machine Learning Overview

We now cross the bridge from *programmed* intelligence (search, logic) to *learned* intelligence. Before any algorithm, machine learning requires a way of **seeing the world as data**: examples, features, labels, and the discipline of honest evaluation. Master this vocabulary now and every model for the rest of the semester — regression, trees, SVMs, neural networks — becomes a variation on one theme.

---

## Open Colab: Feature Weights in a Credit Score Model

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/CreditScoreFeatureWeightEstimator.ipynb)

---

## 1. Intuition First: Programs That Write Themselves

Traditional programming:

```text
rules + data  →  [ program ]  →  answers
```

Machine learning flips the arrows:

```text
data + answers  →  [ learning ]  →  rules (a "model")
```

Nobody can hand-write rules that recognize a cat in pixels. But given 100,000 labeled photos, an algorithm can *find* rules — as parameters — that do. The catch: the learned rules are only as good, and only as fair, as the examples they came from.

---

## 2. The Data Matrix: Examples × Features

Almost every dataset this semester will be a table $X \in \mathbb{R}^{n \times d}$: $n$ **examples** (rows), $d$ **features** (columns), plus — for supervised learning — a **label** vector $y$.

| # | sq. ft. ($x_1$) | bedrooms ($x_2$) | age in yrs ($x_3$) | price $y$ |
|---|---|---|---|---|
| 1 | 1400 | 3 | 20 | \$245,000 |
| 2 | 2100 | 4 | 5  | \$389,000 |
| 3 | 950  | 2 | 45 | \$155,000 |

- A **feature** is a measurable property we *choose* to expose to the learner.
- Each example is then a **point in $d$-dimensional space** — geometry we exploit constantly (distances → k-NN and clustering; separating planes → perceptrons and SVMs; projections → PCA).

**Feature choice is modeling.** "Age of house" vs. "year built" carry the same information, but "distance to nearest school" vs. "ZIP code" do not — and a ZIP code can quietly proxy for race or income. Feature engineering is where domain knowledge *and* ethical judgment enter the pipeline.

---

## 3. The Three Learning Paradigms

| Paradigm | Given | Learns | Course examples |
|---|---|---|---|
| **Supervised** | features + labels $(x_i, y_i)$ | map $x \to y$ | regression, trees, SVMs, neural nets |
| **Unsupervised** | features only $(x_i)$ | structure in $X$ | k-means, density estimation, PCA |
| **Reinforcement** | states, actions, rewards | a decision **policy** | value iteration, Q-learning |

Within supervised learning: predict a **number** → *regression*; predict a **category** → *classification*.

---

## 4. From Features to Predictions: Two Contrasting Styles

To preview where we are headed, here are two ways a model can consume the same features:

1. **Weighted sums (linear models).** Score $= w_1 x_1 + w_2 x_2 + \dots + b$. Learning = choosing weights. *(Weeks 7–9: regression, perceptrons, SVMs — and stacked weighted sums make neural networks.)*
2. **Question trees (decision trees).** "Is sq. ft. > 1500? If yes, is age < 10? ..." Learning = choosing which questions to ask, in what order. *(This Friday's module!)*

Numbered micro-example of style 1 — a toy credit score:

1. Features: income (\$k) $x_1 = 60$, late payments last year $x_2 = 2$.
2. Learned weights: $w_1 = 1.5$ (income helps), $w_2 = -20$ (lateness hurts), bias $b = 10$.
3. Score $= 1.5(60) - 20(2) + 10 = 90 - 40 + 10 = 60$. *(One dot product — the same operation from the intro module.)*
4. Decision: approve if score $> 50$ → approve. *(A threshold turns regression into classification.)*

The Colab notebook above lets you *reverse-engineer* such weights from data — worth 10 minutes to see "learning = estimating weights" concretely.

---

## 5. Generalization: The Only Score That Counts

A model that memorizes its training data can score 100% — and be useless. What we care about is performance on **unseen** data: **generalization**.

The discipline:

1. **Split** the data — e.g., 80% training / 20% test — *before* doing anything else.
2. **Fit** the model on the training set only.
3. **Evaluate** on the untouched test set. This number is your honest estimate.
4. Never tune on the test set; if you must tune, carve out a **validation set** (or cross-validate — details in the Model Evaluation module, week 9).

Two failure modes to name now and diagnose all semester:

| | **Underfitting** | **Overfitting** |
|---|---|---|
| Model is... | too simple | too flexible |
| Training error | high | very low |
| Test error | high | high |
| Fix | richer features / model | more data, regularization, pruning |

---

## 6. Data Hygiene: Where Real Projects Live and Die

Practitioners report that most ML effort is data work, not modeling. A minimal checklist:

1. **Missing values** — drop, impute (mean/median), or add a "was-missing" flag; each choice changes the model.
2. **Scaling** — features on wild scales (sq. ft. in thousands vs. bedrooms in single digits) distort distance-based and gradient-based methods; standardize: $x' = (x - \mu)/\sigma$.
3. **Categorical encoding** — one-hot encode ("red/green/blue" → three 0/1 columns); beware inventing false orderings.
4. **Leakage** — no feature may contain information unavailable at prediction time (e.g., "was the loan repaid" hiding inside a feature). Leakage produces spectacular test scores and catastrophic deployments.
5. **Representativeness** — a model trained on one hospital's patients, one country's faces, or one decade's prices inherits those boundaries silently.

The *Machine Learning Systems* readings for this session expand each of these into full engineering practice (data pipelines, labeling, monitoring).

---

## 7. Runnable Code: The Whole Workflow in Miniature

```python
import numpy as np
rng = np.random.default_rng(477)

# 1. "Collect" data: y = 3*x1 + 20*x2 + noise  (n=200, d=2)
n = 200
X = np.column_stack([rng.uniform(0, 10, n), rng.integers(1, 5, n)])
y = 3 * X[:, 0] + 20 * X[:, 1] + rng.normal(0, 4, n)

# 2. Split BEFORE fitting (80/20)
idx = rng.permutation(n)
train, test = idx[:160], idx[160:]

# 3. Fit a weighted-sum model on TRAINING data only (least squares)
Xb = np.column_stack([X, np.ones(n)])                  # add bias column
w, *_ = np.linalg.lstsq(Xb[train], y[train], rcond=None)
print("learned weights:", np.round(w, 2), " (true: [3, 20, 0])")

# 4. Evaluate on the untouched TEST split
pred = Xb[test] @ w
rmse = np.sqrt(np.mean((pred - y[test]) ** 2))
print("test RMSE:", round(rmse, 2), " (noise floor was 4)")
```

**What to look for:** the learned weights land near the true $[3, 20]$, and test RMSE approaches the noise level $\sigma = 4$ — the best any model could do. When week 7 opens with regression, you will study *why* `lstsq` finds those weights (spoiler: calculus + linear algebra).

---

## Comprehension Quiz

[[MC]]
A colleague reports 99.8% accuracy... measured on the same data used to fit the model. What can you conclude?
- ( ) The model will generalize roughly that well.
- (x) Very little — training-set performance cannot distinguish learning from memorization.
- ( ) The model has underfit.
- ( ) The features must have been well chosen.

[[MC]]
Predicting tomorrow's electricity demand (a number) from weather features, using historical labeled records, is:
- ( ) unsupervised classification.
- ( ) reinforcement learning.
- (x) supervised regression.
- ( ) supervised classification.

[[MC]]
Which scenario describes data leakage?
- ( ) Standardizing features to zero mean.
- (x) Including "number of fraud investigations opened on this transaction" as a feature for predicting fraud.
- ( ) Using 5-fold cross-validation.
- ( ) One-hot encoding a color feature.

---

## Discussion Prompt

> Suppose you are building an admissions-support model for a college. Propose five candidate features; for each, state what it measures, what it might *proxy* for, and whether you would keep it. Where does feature engineering become a values decision?
