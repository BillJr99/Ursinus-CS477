# Model Evaluation for Linear & Logistic Regression (scikit-learn)
<!--
author:   William M. Mongan
language: en
narrator: US English Male

comment: Render with https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-perceptronsmodelevaluation.md or locally if deployed via https://www.billmongan.com/LiaScript/?https://raw.githubusercontent.com/BillJr99/Ursinus-CS477/gh-pages/_pages/Activities/liascript-perceptronsmodelevaluation.md

import: https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md

link:   https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-4
        https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap

-->

# Model Evaluation for Linear & Logistic Regression (scikit-learn)

- **Train/Test Splits & K-Fold Cross-Validation**
- **Overfitting vs. Generalization**
- **Linear Regression Metrics:** RMSE and $R^2$
- **Logistic Regression Metrics:** Precision, Recall, F1
- **Threshold Curves:** ROC & AUC
- **Scikit-learn Recipes** to compute each

---

## 0. Environment & Utilities

We will use only built-in datasets or synthetic data (no internet needed).

```python
# Core
import math
import numpy as np
import matplotlib.pyplot as plt

# scikit-learn
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import (
    mean_squared_error, r2_score,
    precision_recall_fscore_support, classification_report,
    confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc, RocCurveDisplay
)
from sklearn.datasets import make_regression, load_breast_cancer

np.random.seed(42)
plt.rcParams['figure.figsize'] = (6,4)
plt.rcParams['axes.grid'] = True

print('Environment ready.')
```

---

# Part I — Splits, Cross-Validation, and Overfitting

## 1. Train/Test Split

**Goal:** estimate generalization by holding out data the model has **never** seen.

```python
# Synthetic regression data
X, y = make_regression(n_samples=600, n_features=5, noise=15.0, random_state=0)

# 80/20 split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Simple pipeline: scale → linear regression
reg = Pipeline([('scaler', StandardScaler()), ('linreg', LinearRegression())])
reg.fit(X_train, y_train)

y_pred = reg.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)
print(f'Test RMSE: {rmse:.3f}, Test R^2: {r2:.3f}')
```

### Why split?

- Mimics **future, unseen** data.
- If you evaluate on training data, you will **overestimate** performance.

---

## 2. K-Fold Cross-Validation (CV)

**Idea:** repeat the “train on a subset, test on the rest” process **K** times to reduce variance of a single split.

- Split data into $K$ equal folds.
- For each fold $k$: train on the other $K-1$ folds and evaluate on fold $k$.
- Report the **mean ± std** across folds.

```python
kf = KFold(n_splits=5, shuffle=True, random_state=42)
scores_r2   = cross_val_score(reg, X, y, cv=kf, scoring='r2')
scores_rmse = -cross_val_score(reg, X, y, cv=kf, scoring='neg_root_mean_squared_error')

print('CV R^2 (5-fold):   ', f'{scores_r2.mean():.3f} ± {scores_r2.std():.3f}')
print('CV RMSE (5-fold):  ', f'{scores_rmse.mean():.3f} ± {scores_rmse.std():.3f}')
```

> **When to use StratifiedKFold?** For **classification** to preserve label proportions in each fold.

```python
# Example (for later, classification):
# skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
```

---

## 3. Overfitting vs. Generalization

**Overfitting**: model learns **noise** in training data.

- High training performance, **low test/CV** performance.
- Typically caused by excessive **capacity/complexity** or data leakage.

**Bias–Variance perspective**

- **High variance** models: low bias (fit training data very well) but large changes across splits.
- **High bias** models: underfit (both training and test errors are high).

**How to detect/mitigate**

- Compare **train vs. test/CV** metrics.
- Use **regularization**, fewer features, or **more data**.
- Tune hyperparameters using **CV**, not the test set.

[![Machine Learning Fundamentals: Bias and Variance (StatQuest)](https://img.youtube.com/vi/EuBBz3bI-aA/0.jpg)](https://www.youtube.com/watch?v=EuBBz3bI-aA)

*Video summary (text equivalent):* StatQuest illustrates bias (a model too rigid to capture the true pattern) and variance (a model that changes wildly across training sets) with a simple curve-fitting example, and shows why the best test-set performance lives at the sweet spot between the two.

---

# Part II — Linear Regression Metrics

## 4. RMSE (Root Mean Squared Error)

Measures average magnitude of error (same units as $y$):

$$
\text{RMSE}=\sqrt{\frac{1}{n}\sum_{i=1}^n (\hat y_i - y_i)^2}.
$$

```python
print('RMSE (again):', math.sqrt(mean_squared_error(y_test, y_pred)))
```

---

## 5. Coefficient of Determination $R^2$

Fraction of variance explained (with an intercept):

$$
R^2 = 1 - \frac{\sum_i (y_i - \hat y_i)^2}{\sum_i (y_i - \bar y)^2}.
$$

Properties:
- $R^2 \in (-\infty,1]$; $1$ is perfect, $0$ equals predicting $\bar y$, negative = worse than mean.
- For simple linear regression with intercept, $R^2 = \mathrm{corr}(y,\hat y)^2$.

```python
print('R^2 (again):', r2_score(y_test, y_pred))
```

---

# Part III — Logistic Regression Metrics

We now switch to a **binary** classification dataset.

```python
data = load_breast_cancer()
X_clf = data.data
y_clf = data.target  # 0 = malignant, 1 = benign

Xc_train, Xc_test, yc_train, yc_test = train_test_split(
    X_clf, y_clf, test_size=0.25, random_state=42, stratify=y_clf
)

clf = Pipeline([('scaler', StandardScaler()),
                ('logreg', LogisticRegression(max_iter=1000))])
clf.fit(Xc_train, yc_train)

y_pred = clf.predict(Xc_test)        # hard labels at threshold 0.5
y_prob = clf.predict_proba(Xc_test)[:, 1]  # P(y=1|x)
print(classification_report(yc_test, y_pred, target_names=data.target_names))
```

---

## 6. Confusion Matrix

Counts of True/False Positives/Negatives at a fixed threshold (default 0.5).

```python
cm = confusion_matrix(yc_test, y_pred, labels=[0,1])
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['malignant','benign'])
disp.plot()
plt.title('Confusion Matrix')
plt.show()
```

Let the positive class be **benign** ($y=1$). Then:

- **TP**: predicted 1 & true 1
- **FP**: predicted 1 but true 0
- **TN**: predicted 0 & true 0
- **FN**: predicted 0 but true 1

---

## 7. Precision, Recall, F1

- **Precision** (positive predictive value): among predicted positives, **how many are correct?**  
  $$ \text{Precision}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FP}}. $$

- **Recall** (sensitivity, TPR): among actual positives, **how many did we find?**  
  $$ \text{Recall}=\frac{\mathrm{TP}}{\mathrm{TP}+\mathrm{FN}}. $$

- **F1 score** (harmonic mean): balances precision and recall  
  $$ \text{F1} = 2 \cdot \frac{\text{Precision}\cdot \text{Recall}}{\text{Precision}+\text{Recall}}. $$

```python
prec, rec, f1, _ = precision_recall_fscore_support(yc_test, y_pred, average='binary')
print(f'Precision: {prec:.3f}, Recall: {rec:.3f}, F1: {f1:.3f}')
```

> **Imbalanced data tip:** accuracy can mislead; prefer **precision/recall/F1** and **AUC**.

---

## 8. ROC Curve & AUC

**ROC** plots Recall (TPR) vs. FPR as you vary the threshold on $p=\Pr(y=1\mid x)$.  
**AUC** is the area under ROC: probability that a randomly chosen positive ranks above a randomly chosen negative.

```python
fpr, tpr, thr = roc_curve(yc_test, y_prob)
roc_auc = auc(fpr, tpr)

RocCurveDisplay(fpr=fpr, tpr=tpr, roc_auc=roc_auc, estimator_name='LogisticRegression').plot()
plt.title(f'ROC Curve (AUC = {roc_auc:.3f})')
plt.show()
```

**Why AUC?**

- **Threshold‑independent** summary.
- More **robust to class imbalance** than accuracy.
- Measures **ranking quality** of the scores.

---

## 9. Cross‑Validation for Classification

Use **StratifiedKFold** to preserve label ratios.

```python
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# AUC via cross_val_score (uses predict_proba by default for 'roc_auc')
auc_cv = cross_val_score(clf, X_clf, y_clf, cv=skf, scoring='roc_auc')
f1_cv  = cross_val_score(clf, X_clf, y_clf, cv=skf, scoring='f1')

print('CV AUC (5-fold): ', f'{auc_cv.mean():.3f} ± {auc_cv.std():.3f}')
print('CV F1  (5-fold): ', f'{f1_cv.mean():.3f} ± {f1_cv.std():.3f}')
```

---

## 10. Threshold Tuning (Precision–Recall Tradeoff)

Shifting the decision threshold changes precision & recall.

```python
# Sweep thresholds and compute precision/recall pairs
thresholds = np.linspace(0.0, 1.0, 21)
pairs = []
for thr in thresholds:
    y_hat = (y_prob >= thr).astype(int)
    p, r, f1, _ = precision_recall_fscore_support(yc_test, y_hat, average='binary', zero_division=0)
    pairs.append((thr, p, r, f1))

for thr, p, r, f1 in pairs[:5]:
    print(f't={thr:.2f}: Precision={p:.3f}, Recall={r:.3f}, F1={f1:.3f}')

# Simple PR visualization
pr = np.array(pairs)
plt.plot(pr[:,1], pr[:,2], marker='o')  # Recall (x) vs Precision (y)
plt.xlabel('Precision'); plt.ylabel('Recall'); plt.title('Precision–Recall tradeoff (varying threshold)')
plt.show()
```

> In safety‑critical tasks (e.g., medical screening), prefer **high recall** (few false negatives) even if precision drops; in filtering spam, you may prioritize **precision**.

---

# Part IV — Putting It All Together

## 11. Checklist: Linear vs Logistic Evaluation

| Task | Typical Metric(s) | scikit‑learn function(s) |
|---|---|---|
| **Linear regression** | RMSE, $R^2$ | `mean_squared_error`, `r2_score` |
| **Binary classification** | Precision, Recall, F1 | `precision_recall_fscore_support`, `classification_report` |
| **Ranking/threshold‑free** | ROC AUC | `roc_curve`, `auc`, `RocCurveDisplay` |
| **Model selection** | CV mean ± std | `KFold`, `StratifiedKFold`, `cross_val_score` |
| **Diagnostics** | Confusion matrix | `confusion_matrix`, `ConfusionMatrixDisplay` |

---

## 12. Common Pitfalls & Best Practices

- **Leakage:** fit scalers/transformers **only on training** data (use `Pipeline`).
- **One split is noisy:** prefer **K‑fold CV** for model selection.
- **Imbalanced classes:** do not rely on accuracy; use **AUC**, **F1**, class weights (`class_weight='balanced'`).
- **Overfitting symptoms:** training ≫ test; fix with **regularization**, simpler models, or more data.
- **Final test:** after selecting hyperparameters via CV, evaluate **once** on the held‑out test set.

---

## 13. Exercises

1. **Vary CV folds:** Compare 3‑, 5‑, and 10‑fold CV for $R^2$ and AUC.
2. **Imbalance stress test:** Drop 60% of benign samples and compare accuracy vs AUC vs F1.
3. **Threshold policy:** Pick a threshold that achieves Recall ≥ 0.98, report resulting Precision and F1.
4. **Polynomial features:** Add `PolynomialFeatures` to the regression pipeline and compare CV RMSE vs $R^2$ as degree grows (watch for overfitting).
5. **Regularization sweep:** For logistic regression, vary `C` over `np.logspace(-3, 2, 10)` and plot AUC vs `C`.

---

## Appendix: Minimal APIs (Copy/Paste)

```python
# Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# CV
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
kf  = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=kf, scoring='r2')  # or 'roc_auc', 'f1', ...

# Linear metrics
rmse = math.sqrt(mean_squared_error(y_test, y_pred))
r2   = r2_score(y_test, y_pred)

# Classification metrics
prec, rec, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='binary')

# ROC/AUC
fpr, tpr, _ = roc_curve(y_test, y_prob)
roc_auc = auc(fpr, tpr)
```

# Support Vector Machines (SVM)

## Open Colab: SVM From Scratch (Linear & Kernel)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/SVM_From_Scratch_Linear_and_Kernel.ipynb)

---

## Warm-Up Demo: Logistic Regression as the SVM's Baseline

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/linear_logistic_regression_demo.ipynb)

Before studying max-margin classifiers, re-run the linear/logistic regression demo notebook with SVM questions in mind:

- **Logistic-regression cells:** the fitted boundary is a straight line — but *which* straight line? Logistic regression places it to maximize the likelihood of *all* points, so it can be swayed by points far from the boundary.
- **Boundary plots:** the SVM below chooses its line using **only the points nearest the boundary** (the support vectors) and pushes the margin as wide as possible. On well-separated data the two lines nearly coincide; on noisy, overlapping data they diverge — add noise in the demo's data cell and predict which model moves more.
- **Loss connection:** logistic regression's log-loss decays smoothly and never quite reaches zero, while the SVM's **hinge loss** (Section 2 below) is exactly zero beyond the margin — this is *why* distant points cannot influence an SVM.

**What to look for:** both models predict with $\mathrm{sign}(\mathbf{w}^\top \mathbf{x} + b)$. Everything that differs — margins, support vectors, kernels — comes from the *training objective*, not the decision rule.

---

## 1. Intuition & Goal

Support Vector Machines learn a **max-margin** decision boundary. For a binary task with labels $y_i \in \{-1, +1\}$ and feature vectors $\mathbf{x}_i$:

- The **linear** decision function is $f(\mathbf{x}) = \mathbf{w}^\top \mathbf{x} + b$ and the classifier is $\mathrm{sign}(f(\mathbf{x}))$.
- The goal is to maximize the **geometric margin** subject to correct classification (hard-margin) or limited violations (soft-margin).

---

## 2. Linear SVM: Hard-Margin and Soft-Margin

**Hard-margin (separable):**
$$
\min_{\mathbf{w}, b} \ \frac{1}{2}\lVert \mathbf{w} \rVert^2
\quad \text{s.t.} \quad y_i\,(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 \ \forall i.
$$

**Soft-margin (with slack $\xi_i \ge 0$ and penalty $C>0$):**
$$
\min_{\mathbf{w}, b, \{\xi_i\}} \ \frac{1}{2}\lVert \mathbf{w} \rVert^2 + C \sum_{i=1}^n \xi_i
\quad \text{s.t.} \quad y_i\,(\mathbf{w}^\top \mathbf{x}_i + b) \ge 1 - \xi_i.
$$

Equivalent **regularized hinge-loss** form:
$$
\min_{\mathbf{w}, b}\ \frac{1}{2}\lVert \mathbf{w} \rVert^2 + C\sum_{i=1}^n \max\!\big(0,\,1 - y_i(\mathbf{w}^\top \mathbf{x}_i + b)\big).
$$

- $C \uparrow$ $\Rightarrow$ **less** regularization (fit training errors more aggressively).
- $C \downarrow$ $\Rightarrow$ **more** regularization (wider margin, tolerate more violations).

---

## 3. Kernel SVM & Dual Formulation

The dual problem depends on inner products $\mathbf{x}_i^\top \mathbf{x}_j$. Replacing them with a **kernel** $K(\mathbf{x}_i,\mathbf{x}_j)$ maps implicitly to a high-dimensional feature space:

- **RBF (Gaussian) kernel:** $K(\mathbf{x},\mathbf{x}') = \exp(-\gamma \lVert \mathbf{x} - \mathbf{x}' \rVert^2)$  
  - $\gamma \uparrow$ $\Rightarrow$ narrower kernels (more complex boundaries).  
  - $\gamma \downarrow$ $\Rightarrow$ smoother boundaries.

- **Polynomial kernel:** $K(\mathbf{x},\mathbf{x}') = (\mathbf{x}^\top \mathbf{x}' + c)^d$

Support vectors are the training points with non-zero dual weights; they define the decision boundary.

---

## 4. scikit-learn Demo (Linear vs. RBF) — Classification Metrics

We will compare **linear** and **RBF** SVMs on a non-linear toy dataset and report evaluation metrics consistent with earlier sections.

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC, LinearSVC
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay,
    roc_curve, auc
)

X, y = make_circles(n_samples=800, factor=0.4, noise=0.25, random_state=42)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Linear SVM (hinge) with feature scaling
lin_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", LinearSVC(C=1.0, max_iter=5000))
])

# RBF SVM with probability estimates for ROC
rbf_clf = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(C=1.0, gamma="scale", kernel="rbf", probability=True))
])

lin_clf.fit(Xtr, ytr)
rbf_clf.fit(Xtr, ytr)

print("=== Linear SVM Report ===")
print(classification_report(yte, lin_clf.predict(Xte)))
print("=== RBF SVM Report ===")
print(classification_report(yte, rbf_clf.predict(Xte)))

# Confusion matrix (RBF)
cm = confusion_matrix(yte, rbf_clf.predict(Xte))
ConfusionMatrixDisplay(cm, display_labels=["class 0","class 1"]).plot()
plt.title("RBF SVM — Confusion Matrix")
plt.show()

# ROC AUC (RBF; use predict_proba for scores)
y_score = rbf_clf.predict_proba(Xte)[:,1]
fpr, tpr, thr = roc_curve(yte, y_score)
print("AUC:", auc(fpr, tpr))
```

**Notes**
- Always **scale features** for SVMs.  
- For ROC/AUC with SVMs, use `probability=True` (Platt scaling) or `decision_function` scores.

---

## 5. From Scratch: Linear SVM via Hinge-Loss SGD (Didactic)

The following illustrates a **didactic** (not production-grade) linear SVM using subgradient updates on the hinge loss.

```python
import numpy as np

def sgd_linear_svm(X, y, C=1.0, lr=1e-2, epochs=200):
    # y in {-1, +1}
    n, d = X.shape
    w = np.zeros(d); b = 0.0
    for _ in range(epochs):
        for i in np.random.permutation(n):
            margin = y[i]*(X[i].dot(w) + b)
            if margin >= 1:
                # Only regularization term
                w -= lr * w
            else:
                # Regularization + hinge penalty gradient
                w -= lr * (w - C * y[i] * X[i])
                b += lr * (C * y[i])
    return w, b

# Example on a linearly separable toy set
rng = np.random.default_rng(0)
X_pos = rng.normal(loc=[2,2], scale=0.5, size=(50,2))
X_neg = rng.normal(loc=[-2,-2], scale=0.5, size=(50,2))
X_toy = np.vstack([X_pos, X_neg])
y_toy = np.hstack([np.ones(50), -np.ones(50)])

w, b = sgd_linear_svm(X_toy, y_toy, C=1.0, lr=1e-2, epochs=20)
print("w:", w, "b:", b)
```

---

## 6. Hyperparameter Tuning (C, $\gamma$) with Cross-Validation

Use **StratifiedKFold** and grid search to optimize generalization performance.

```python
from sklearn.model_selection import GridSearchCV

pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(kernel="rbf", probability=True))
])

param_grid = {
    "svm__C": [0.1, 1, 10],
    "svm__gamma": ["scale", 0.1, 0.01]
}

grid = GridSearchCV(pipe, param_grid, scoring="roc_auc",
                    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42))
grid.fit(X, y)
print("Best AUC:", grid.best_score_)
print("Best params:", grid.best_params_)
```

**Interpretation**
- $C$ controls the **margin–violation trade-off** (regularization strength).
- $\gamma$ controls **kernel width**; too large can overfit (wiggly boundary), too small can underfit (overly smooth).

---

## 7. Connecting Back to Model Evaluation

- Use the **same metrics** as logistic regression: precision, recall, F1, ROC AUC.  
- For imbalanced data, prefer **AUC** and **PR curves** over accuracy.  
- **Calibrate** scores if probability estimates are needed for downstream decision thresholds.

---

## 8. Exercises

1. Re-run the RBF SVM with $C \in \{0.1, 1, 10\}$ and $\gamma \in \{\text{"scale"}, 0.1, 0.01\}$; report ROC AUC via 5-fold CV.  
2. Plot decision boundaries for linear vs. RBF SVM on `make_circles` and discuss **bias–variance** behavior.  
3. Replace `make_circles` with `sklearn.datasets.load_breast_cancer` and compare logistic regression vs. RBF SVM AUC.  
4. Extend the from-scratch SGD to include an **$L_1$ penalty** and examine sparsity in $\mathbf{w}$.

---
