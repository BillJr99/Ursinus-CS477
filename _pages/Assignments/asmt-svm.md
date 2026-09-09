---
layout: assignment
permalink: Assignments/SupportVectorMachines
title: "Assignment: Support Vector Machines (SVM) — From Scratch and with Libraries"

info:
  points: 100
  goals:
    - Understand the geometric and optimization foundations of linear SVMs.
    - Implement a linear SVM from scratch using gradient descent and hinge loss.
    - Use scikit-learn's SVM implementation for nonlinear kernels and comparison.
    - Analyze margins, support vectors, and kernel behavior visually.
    - Solve a scaffolded applied problem creatively with SVM concepts.
  purpose: "This assignment builds understanding of SVMs from geometric intuition to implementation. You will derive the primal objective, implement it, use libraries, and apply SVMs to a novel dataset."
  concepts:
    - Margin maximization and separating hyperplanes
    - Hinge loss and regularization
    - Kernel trick (polynomial, RBF)
    - Model complexity vs. generalization
  tasks:
    - Derive and implement linear SVM from scratch using hinge loss and gradient descent.
    - Apply scikit-learn SVMs with kernels to visualize decision boundaries.
    - Solve a new classification problem using SVMs (with scaffolding provided).
  rubric:
    - weight: 30
      description: Implementation
      preemerging: Partial hinge-loss or missing gradient steps.
      beginning: Working linear SVM with gradient updates.
      progressing: Clean modular code with visualizations.
      proficient: Optimized implementation, clear diagnostics, reproducibility.
    - weight: 30
      description: Mathematical Understanding and Derivation
      preemerging: Minimal formulas or unclear reasoning.
      beginning: Correct hinge loss and gradient formulas.
      progressing: Shows derivations of primal and dual, interprets margin.
      proficient: Deep understanding, edge-case discussion, and generalization theory links.
    - weight: 20
      description: Application and Analysis
      preemerging: Limited evaluation or explanation.
      beginning: Basic accuracy evaluation.
      progressing: Visual comparisons across kernels and hyperparameters.
      proficient: Insightful analysis of trade-offs, kernel choice, and regularization effects.
    - weight: 10
      description: Code Quality & Documentation
      preemerging: Sparse comments.
      beginning: Basic docstrings.
      progressing: Modular, readable code with visual plots.
      proficient: Well-structured, documented, and reproducible experiments.
    - weight: 10
      description: Submission Completeness
      preemerging: Incomplete files or results.
      beginning: Includes code and plots.
      progressing: Includes explanations and comparisons.
      proficient: Fully reproducible report and results.

tags:
  - ai
  - machine-learning
  - classification
  - optimization
---

# Overview

Support Vector Machines (SVMs) are a cornerstone of classical machine learning, balancing **margin maximization** with **empirical risk minimization**. In this assignment, you will first implement a linear SVM classifier **from scratch**, then apply **kernelized SVMs** using libraries, and finally solve a small creative application problem.

---

## Stage 0 — Setup

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs, make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
np.random.seed(42)
```

---

## Stage 1 — Linear SVM Theory and Derivation

We begin with the **hard-margin SVM** for separable data. The optimization problem is:

$$
\begin{aligned}
\min_{\mathbf{w}, b} &\quad \frac{1}{2}\lVert \mathbf{w} \rVert^2 \\
\text{s.t.} &\quad y_i(\mathbf{w}^T \mathbf{x}_i + b) \ge 1, \quad i = 1,\dots,n.
\end{aligned}
$$

For non-separable data, introduce **slack variables** $$\xi_i \ge 0$$ and penalty parameter $C$:

$$
\min_{\mathbf{w},b,\xi} \frac{1}{2}\lVert \mathbf{w} \rVert^2 + C\sum_i \xi_i
\quad \text{s.t. } y_i(\mathbf{w}^T\mathbf{x}_i + b) \ge 1 - \xi_i.
$$

In the **primal hinge-loss form**, we solve:

$$
\min_{\mathbf{w},b} \frac{1}{2}\lVert \mathbf{w} \rVert^2 + C \sum_i \max(0, 1 - y_i(\mathbf{w}^T\mathbf{x}_i + b)).
$$

---

## Stage 2 — Linear SVM from Scratch

We implement gradient descent on the primal hinge-loss objective.

```python
class LinearSVM:
    def __init__(self, C=1.0, lr=0.001, epochs=1000):
        self.C = C
        self.lr = lr
        self.epochs = epochs

    def fit(self, X, y):
        n, d = X.shape
        self.w = np.zeros(d)
        self.b = 0
        for _ in range(self.epochs):
            for i in range(n):
                margin = y[i] * (np.dot(self.w, X[i]) + self.b)
                if margin >= 1:
                    dw = self.w
                    db = 0
                else:
                    dw = self.w - self.C * y[i] * X[i]
                    db = -self.C * y[i]
                self.w -= self.lr * dw
                self.b -= self.lr * db

    def predict(self, X):
        return np.sign(np.dot(X, self.w) + self.b)

# Example on a toy dataset
X, y = make_blobs(n_samples=100, centers=2, random_state=0, cluster_std=1.2)
y = 2*(y - 0.5)  # {-1, +1}

svm = LinearSVM(C=1.0, lr=0.001, epochs=1000)
svm.fit(X, y)
y_pred = svm.predict(X)

acc = (y_pred == y).mean()
print("Training accuracy:", acc)

# Visualize decision boundary
xx, yy = np.meshgrid(np.linspace(X[:,0].min()-1, X[:,0].max()+1, 200),
                     np.linspace(X[:,1].min()-1, X[:,1].max()+1, 200))
Z = svm.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.3)
plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k')
plt.title("Linear SVM Decision Boundary (from scratch)")
plt.show()
```

**Checkpoint:** Test with different $C$ values and observe margin width and misclassifications.

---

## Stage 3 — scikit-learn SVMs with Kernels

We now use `sklearn.svm.SVC` for flexible kernels (linear, polynomial, RBF).

```python
from sklearn.svm import SVC

# Nonlinear dataset
X, y = make_moons(noise=0.2, random_state=0)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0)

models = {
    "Linear": SVC(kernel="linear", C=1.0),
    "Polynomial": SVC(kernel="poly", degree=3, C=1.0),
    "RBF": SVC(kernel="rbf", gamma=1.0, C=1.0)
}

for name, model in models.items():
    model.fit(Xtr, ytr)
    acc = model.score(Xte, yte)
    print(f"{name} kernel accuracy: {acc:.3f}")
    # Plot decision boundary
    xx, yy = np.meshgrid(np.linspace(-1.5, 2.5, 300), np.linspace(-1.0, 1.5, 300))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    plt.figure()
    plt.contourf(xx, yy, Z, alpha=0.3)
    plt.scatter(Xte[:,0], Xte[:,1], c=yte, cmap='bwr', edgecolor='k')
    plt.title(f"{name} Kernel Decision Boundary")
    plt.show()
```

**Checkpoint:** Compare decision boundaries for each kernel. Which one overfits or underfits?

---

## Stage 4 — Creative Problem: Classifying Handwritten Digits (Setup + Scaffold)

We now move beyond toy datasets to a more meaningful but tractable task. You will train SVMs to classify **handwritten digits** from `sklearn.datasets.load_digits`.

**Goal.**
1. Implement preprocessing (scaling, splitting, optionally PCA).  
2. Compare linear vs. RBF kernel SVMs.  
3. Evaluate confusion matrix and per-class precision/recall.  
4. Optimize hyperparameters ($C$, $$\gamma$$) using a small grid search.

```python
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV

digits = load_digits()
X, y = digits.data, digits.target

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
Xtr, Xte, ytr, yte = train_test_split(X_scaled, y, test_size=0.3, random_state=0, stratify=y)

param_grid = {"C": [0.1, 1, 10], "gamma": [0.001, 0.01, 0.1]}
grid = GridSearchCV(SVC(kernel="rbf"), param_grid, cv=3)
grid.fit(Xtr, ytr)

print("Best parameters:", grid.best_params_)
y_pred = grid.best_estimator_.predict(Xte)

acc = accuracy_score(yte, y_pred)
print("Test accuracy:", acc)
ConfusionMatrixDisplay(confusion_matrix(yte, y_pred)).plot(cmap="Blues")
plt.title("SVM Digit Classification Confusion Matrix")
plt.show()
```

**Checkpoint:** Report the number of support vectors for the best model and discuss class imbalance effects.

---

## Stage 5 — Scaffolded Challenge: SVMs for Environmental Sensing (Creative)

You will now solve a **new problem** using SVMs. The goal is to detect whether an environmental sensor reading indicates **normal** or **faulty** conditions.

### Dataset Setup (Provided)

Simulate a two-feature dataset representing two sensors: temperature and humidity. Label 0 = normal, 1 = faulty.

```python
rng = np.random.default_rng(1)
n = 200
temp = np.r_[rng.normal(20, 2, n//2), rng.normal(35, 3, n//2)]
humid = np.r_[rng.normal(50, 5, n//2), rng.normal(30, 4, n//2)]
y = np.r_[np.zeros(n//2), np.ones(n//2)]
X = np.c_[temp, humid]

plt.scatter(X[:,0], X[:,1], c=y, cmap='bwr', edgecolor='k')
plt.xlabel("Temperature (°C)"); plt.ylabel("Humidity (%)")
plt.title("Simulated Environmental Data")
plt.show()
```

### Your Tasks

1. Implement **from-scratch linear SVM** on this dataset (reuse Stage 2 class).  
2. Compare against **RBF kernel SVM** using scikit-learn.  
3. Visualize decision boundaries and support vectors.  
4. Discuss **margin width**, **support vector count**, and **generalization**.  
5. Propose an **additional feature** (derived or nonlinear) that could improve separability.

**Stretch goal:** Add Gaussian noise or shift sensor distributions and evaluate model robustness.

---

# What to Submit

1. Your notebook or scripts implementing **linear SVM (scratch)** and **kernel SVM (library)**.  
2. Plots of decision boundaries and confusion matrices for each section.  
3. A short (2–3 page) report describing derivations, parameter effects, and reflections on generalization.  
4. In the creative section, include your reasoning for feature engineering or robustness tests.  
5. Reproducibility: Fix random seeds and include code comments.

---
