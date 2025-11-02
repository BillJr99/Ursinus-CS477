
# Principal Component Analysis from Scratch, Fisher Separability, and Singular Value Decomposition
<!--
author:   William M. Mongan
language: en
narrator: US English Male

comment: Render with https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-pca.md or locally if deployed via https://www.billmongan.com/LiaScript/?https://raw.githubusercontent.com/BillJr99/Ursinus-CS477/gh-pages/_pages/Activities/liascript-pca.md

import: https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md

link:   https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-4
        https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap

-->

## Dimensionality Reduction
We start with the intuition, then the derivation of, **Principal Component Analysis (PCA)**, step-by-step from first principles using only geometry and averages.  
We connect PCA to **eigendecomposition** and then to the **Singular Value Decomposition (SVD)** — all explained for learners without a linear algebra background.  
Finally, we analyze the **Iris** dataset with a simple **linear regression One-vs-Rest** classifier and show how PCA helps with metrics and visualization, and how **Fisher's separability score** relates to PCA eigenvalues.

---

## Table of Contents

1. Motivation: Why PCA?
2. Data as Points in Space
3. Centering and Covariance
4. Variance, Directions, and the Rayleigh Quotient
5. **Eigendecomposition** for Beginners
6. Deriving PCA with Gentle Lagrange Multipliers
7. From PCA to **SVD** (and back)
8. Choosing the Number of Components ($k$) & Whitening
9. PCA as Low-Rank **Reconstruction** (Eckart–Young)
10. PCA From Scratch — Code Walkthrough (Deep Dive)
11. Sanity Check with scikit-learn
12. Fisher's Separability: Why Large Eigenvalues Often Help
13. Iris Classification Pipeline: Metrics & Validation (Deep Dive)
14. Decision Regions & Intuition
15. Practical Tips and Pitfalls
16. Summary & Further Exercises
17. Appendix A: Hand-Derivation Checklist
18. Appendix B: Code Map (Notebook ↔ Slides)

---

## 1. Motivation: Why PCA?

**Problem:** Many datasets have features that are redundant or noisy. Models trained on all features may:

- Overfit (pick up noise)
- Be slow (more dimensions → more cost)
- Be hard to visualize or interpret
- Suffer from the **curse of dimensionality** (distances become less informative as $d$ grows)

**Goal of PCA:** Find a small number of **new** features (called **principal components**) that capture most of the useful variation in the data.

**Key properties:**

- Components are **uncorrelated** (orthogonal directions).
- Ordered by **explained variance**: PC1 captures the most, then PC2, etc.
- Works **without labels** (unsupervised).

**Payoff:** Compress data, denoise, visualize, and often improve generalization when fed into a simple classifier.

**Intuition preview:** PCA finds the **axes** of an ellipse (in 2D) or ellipsoid (in 3D+). The **longest axis** is PC1, the next longest perpendicular axis is PC2, etc.

---

## 2. Data as Points in Space 

Imagine each example is a point with $d$ coordinates (one per feature).  
For 2 features $(x_1, x_2)$ you can draw points on a 2D plane. For 4 features, it’s like a 4D "cloud", but we reason with the same ideas.

**Patterns we care about:**
- Where is the cloud centered?
- In which direction is it most “stretched” (has most spread/variance)?
- Can we rotate our axes to align with those “stretch” directions?

PCA answers: *rotate the coordinate system so that the axes line up with the directions of **maximum variance***.

**Micro-Example (2D):** Suppose we have points near a slanted line $y \approx 2x$. The longest stretch is along that slant; PCA will set **PC1** along that slant, and **PC2** perpendicular to it. Projecting onto PC1 retains most information.

---

## 3. Centering and Covariance

**Centering:** Subtract the average of each feature so the cloud is centered at the origin.

- Data matrix: $X \in \mathbb{R}^{n \times d}$ (rows=samples, cols=features).  
- Column-wise mean: $\mu \in \mathbb{R}^d$, with $\mu_j = \frac{1}{n}\sum_{i=1}^n X_{ij}$.  
- Centered data: $\tilde{X} = X - \mathbf{1}\mu^\top$.

**Covariance matrix:** measures how features vary together.

$$
C \;=\; \frac{1}{n-1} \tilde{X}^\top \tilde{X}
\quad\in\mathbb{R}^{d\times d}.
$$

- $C_{jj}$ is the variance of feature $j$.
- $C_{jk}$ is the covariance between features $j$ and $k$.
- $C$ is **symmetric** (mirror across the diagonal).

**Goal restated:** Find directions (unit vectors) $w$ where the variance of projections $\tilde{X}w$ is **maximized**.

> Why divide by $(n-1)$? It makes $C$ an unbiased estimator of the population covariance when the mean is estimated from the data.

**Worked numeric example (2×2):**  
Let centered data be
$$
\tilde{X}=\begin{bmatrix}
-1 & -1 \\
0 & 0 \\
1 & 1
\end{bmatrix},\quad
C=\tfrac{1}{2}\tilde{X}^\top\tilde{X}
=\tfrac{1}{2}\begin{bmatrix}2 & 2\\2 & 2\end{bmatrix}
=\begin{bmatrix}1&1\\1&1\end{bmatrix}.
$$
You can already **see** the cloud lies along the line $x=y$ (variance only in that direction).

---

## 4. Variance, Directions, and the Rayleigh Quotient

For a unit direction $w$ (think of it as an arrow of length 1), the **projected data** is $z = \tilde{X}w$ (a 1D number per sample).  
Its variance is:

$$
\mathrm{Var}(z) \;=\; \frac{1}{n-1}\|\tilde{X}w\|^2 
\;=\; w^\top \left(\frac{1}{n-1}\tilde{X}^\top \tilde{X}\right) w
\;=\; w^\top C w.
$$

The function $w \mapsto w^\top C w$ with $\|w\|=1$ is the **Rayleigh quotient**.  
We want the $w$ that **maximizes** this value — the direction of **maximum variance**.

**Key theorem (informal):** For symmetric $C$, the Rayleigh quotient is maximized by the **top eigenvector** of $C$, and the maximum equals the **top eigenvalue**.

**Why this matters:** It links “find the most variable direction” to **eigenvectors**/**eigenvalues**.

---

## 5. Eigendecomposition for Beginners

**Definition:** An **eigenvector** $v$ of $C$ with **eigenvalue** $\lambda$ satisfies $Cv=\lambda v$.  
For covariance matrices, eigenvalues are $\lambda\ge 0$ and eigenvectors for distinct eigenvalues are **orthogonal**.

**Geometric picture:** $C$ turns circles into ellipses; eigenvectors give ellipse axes; axis lengths $\propto \sqrt{\lambda}$.

**2×2 example continued:** For $C=\begin{bmatrix}1&1\\1&1\end{bmatrix}$,
solve $\det(C-\lambda I)=0$:
$$
\det\begin{bmatrix}1-\lambda&1\\1&1-\lambda\end{bmatrix}
=(1-\lambda)^2-1=0
\Rightarrow \lambda\in\{0,2\}.
$$

- For $\lambda=2$, $(C-2I)v=0 \Rightarrow v\propto (1,1)$ (PC1).
- For $\lambda=0$, $v\propto (1,-1)$ (PC2).

**Explained Variance Ratio (EVR):**
$$
\text{EVR}_k=\frac{\lambda_k}{\sum_{j=1}^d \lambda_j}.
$$
Here, total variance $\sum\lambda_j=2$, so EVR$_1=1$, EVR$_2=0$. PC1 explains **all** variance.

**Why eigenvectors/eigenvalues correspond to meaningful features:**  

- Eigenvectors are **linear combinations** of original features that reveal **independent axes of variation**.  
- Eigenvalues measure **how much** variation lies along each axis. Large $\lambda$ = “lots of structure/info” along that axis.  
- In noisy, redundant data, top eigenvectors emphasize **shared, stable structure** and deemphasize random noise (often present in low-variance directions).

---

## 6. Deriving PCA with Lagrange Multipliers

We want to maximize $w^\top C w$ subject to $\|w\|=1$.  
Set
$$
\mathcal{L}(w,\alpha)=w^\top C w-\alpha(w^\top w-1).
$$
Stationary points satisfy
$$
\nabla_w\mathcal{L}=2Cw-2\alpha w=0 \Rightarrow Cw=\alpha w.
$$
So $w$ must be an **eigenvector** of $C$. The **largest** eigenvalue gives PC1; successive PCs are the next eigenvectors under orthogonality constraints (**Rayleigh–Ritz**).

**Intuition:** Subject to unit length, the best “aiming direction” is an axis of the covariance ellipsoid; the **longest** axis maximizes variance.

---

## 7. From PCA to Singular Value Decomposition (SVD)

The **SVD** of the centered data $\tilde{X}$ is
$$
\tilde{X}=U\Sigma V^\top,
$$
with orthonormal columns in $U,V$ and nonnegative singular values on $\Sigma$.

**Covariance via SVD:**
$$
C=\tfrac{1}{n-1}\tilde{X}^\top\tilde{X}
=V\left(\tfrac{\Sigma^2}{n-1}\right)V^\top.
$$

Thus:

- **Principal directions** = columns of **$V$** (right singular vectors).
- **Eigenvalues** $\lambda_k=\sigma_k^2/(n-1)$.

**Why SVD is preferred in practice:**

- Numerically stable; handles rank deficiency.
- Efficient when $n\ll d$ or $d\ll n$.

**Dual viewpoint (when $n<d$):** eigendecompose $G=\tfrac{1}{n-1}\tilde{X}\tilde{X}^\top=U(\Sigma^2/(n-1))U^\top$ then recover $V=\tilde{X}^\top U\Sigma^{-1}$.

---

## 8. Choosing the Number of Components ($k$) & Whitening

**Choosing $k$**

- **Scree plot**: look for elbow in $\lambda_k$ vs $k$.
- **EVR threshold**: smallest $k$ with $\sum_{j=1}^k \lambda_j / \sum_{j} \lambda_j \ge \tau$ (e.g., 0.95).
- **Validation-based**: pick $k$ maximizing downstream performance (e.g., macro F1).

**Whitening:**
After $Z=\tilde{X}V_k$, set $Z_\text{white}=Z\Lambda_k^{-1/2}$. Now $\operatorname{Cov}(Z_\text{white})\approx I$.  
**Pros**: decorrelated, unit variance features. **Cons**: can amplify noise when $\lambda_k$ is tiny.

---

## 9. PCA as Low-Rank **Reconstruction** (Eckart–Young)

Another view: PCA finds the **best rank-$k$ approximation** of $\tilde{X}$ (in least-squares sense).  
If $\tilde{X}=U\Sigma V^\top$, then the best rank-$k$ approximation is $U_k\Sigma_k V_k^\top$ (keep top $k$ singular values).

**Eckart–Young theorem:** Among all rank-$k$ matrices $\hat{X}$,
$$
\|\tilde{X}-\hat{X}\|_F \text{ is minimized by } \hat{X}=U_k\Sigma_k V_k^\top.
$$

**Connection to “explained variance”:**  
The squared Frobenius norm $\|\tilde{X}\|_F^2=\sum_i \sigma_i^2 = (n-1)\sum_j\lambda_j$.  
Keeping top $k$ singular values preserves the largest share of energy/variance.

---

## 10. PCA From Scratch 

```python
def pca_from_scratch(X, n_components=None):
    # Returns (mean, components, eigenvalues, explained_variance_ratio, Z, C)
    X = np.asarray(X)
    mu = X.mean(axis=0)               # column-wise means
    Xc = X - mu                       # center
    C = (Xc.T @ Xc) / (len(Xc) - 1)   # covariance
    eigvals, eigvecs = np.linalg.eigh(C)  # eigen-decomposition (symmetric)
    order = np.argsort(eigvals)[::-1]     # sort descending
    eigvals = eigvals[order]
    eigvecs = eigvecs[:, order]
    if n_components is not None:
        eigvecs = eigvecs[:, :n_components]
        eigvals = eigvals[:n_components]
    total_var = np.linalg.eigvalsh(C).sum()
    evr = eigvals / total_var         # explained variance ratio
    Z = (Xc @ eigvecs)                # projections
    return mu, eigvecs, eigvals, evr, Z, C
```

**Key details:**

- `eigh` exploits symmetry; **guarantees real** eigenpairs.  
- Sorting ensures components are ordered by **importance**.  
- `Z` are the **scores** — coordinates in PC space.  
- `evr` implements **EVR**: $\text{EVR}_k=\lambda_k/\sum_j\lambda_j$.

**Loadings:** The matrix `eigvecs` contains **loadings** — how each original feature contributes to a PC. Large magnitude $\Rightarrow$ stronger contribution.

### 10.1 Sanity check block (2D toy)
- Plot centered data, overlay PC axes (scaled by $\sqrt{\lambda}$).  
- Histograms of $Z_{:,0}$ vs $Z_{:,1}$ to confirm PC1 variance $\gg$ PC2.

### 10.2 Notes on numerical methods

- **Power iteration** finds top eigenvector by repeatedly applying $C$ to a vector and renormalizing.  
- **Deflation** removes the found component to get the next.  
- Practical PCA uses **SVD**, not explicit $C$ eigen-decomposition, for stability.

---

## 11. Sanity Check with scikit-learn

We use `sklearn.decomposition.PCA` and compare:

- `explained_variance_` (our $\lambda$)  
- `explained_variance_ratio_` (our EVR)  
- `components_` (our eigenvectors, up to sign)

**Sign flip note:** If $\hat{v}\approx -v$, it’s still the *same direction*.

---

## 12. Fisher's Separability: Why Large Eigenvalues Often Help

**Binary Fisher score (for 1D projections):**
$$
S_F = \frac{(\mu_1 - \mu_2)^2}{\sigma_1^2 + \sigma_2^2},
$$
where $\mu_k$ and $\sigma_k^2$ are the class means/variances in the **projected** 1D space.

**Multiclass:** average $S_F$ across unordered class pairs.

**Why correlate with eigenvalues?**  
Projecting onto PC1 maximizes overall variance. If between-class variance is a **large share** of total variance, then PC1 tends to separate classes, raising $S_F$. (Not guaranteed; verify on your data.)

**Relation to LDA:** Fisher’s criterion is the 1D version of **Linear Discriminant Analysis (LDA)**, which explicitly optimizes **between-class / within-class** scatter. PCA is unsupervised; LDA is supervised.

---

## 13. Iris Classification Pipeline: Metrics & Validation

### 13.1 OVR Linear Regression — Concept, Math, and Code

**OVR (One-vs-Rest) strategy:** Turn a $K$-class problem into $K$ binary regressions.  
For class $k$, define a target $y^{(k)}\in\{0,1\}$ (1 if sample is class $k$). Fit linear regression:
$$
\min_{w_k,b_k}\ \|y^{(k)} - (X w_k + b_k)\|_2^2.
$$
Prediction uses **scores** $s_k(x)=x^\top w_k+b_k$, choose $\hat{y}=\arg\max_k s_k(x)$.

**Pros:** Simple, interpretable, continuous scores (for ROC).  
**Cons:** Not probabilistic; can produce scores outside $[0,1]$; less robust than logistic loss.

**Code (excerpt):**

```python
class OVRLinearRegression:
    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.regressors_ = []
        for c in self.classes_:
            r = LinearRegression()
            r.fit(X, (y == c).astype(float))
            self.regressors_.append(r)
        return self

    def decision_function(self, X):
        return np.column_stack([r.predict(X) for r in self.regressors_])

    def predict(self, X):
        scores = self.decision_function(X)
        return self.classes_[np.argmax(scores, axis=1)]

    def score_individual_r2(self, X, y):
        r2s = []
        for c, r in zip(self.classes_, self.regressors_):
            y_bin = (y == c).astype(float)
            r2s.append(r2_score(y_bin, r.predict(X)))
        return r2s, float(np.mean(r2s))
```

**Reading coefficients:** Each $w_k$ is a **normal vector** to a separating hyperplane for class $k$ vs rest. The **decision boundary** between classes $i$ and $j$ occurs where $s_i(x)=s_j(x)$.

**Regularization (optional):** Ridge regression often improves stability:
$\min \|y-Xw\|^2+\alpha\|w\|^2$. Try `Ridge()` in place of `LinearRegression()`.

### 13.2 Pipelines — Scaling & PCA

- **Baseline:** `StandardScaler → OVR Linear Regression`  
- **With PCA:** `StandardScaler → PCA(2) → OVR Linear Regression`

**Why scale first?** PCA is not scale-invariant; features with large units would dominate otherwise.

**Why PCA here?** Reduce redundancy, potentially denoise, and visualize in 2D.

### 13.3 Metrics — Definitions with Formulas

- **Confusion Matrix** $[n_{ij}]$: true $i$, predicted $j$.
- **Per-class Precision/Recall/F1:**
  $$
  \text{Precision}_k=\frac{\text{TP}_k}{\text{TP}_k+\text{FP}_k},\quad
  \text{Recall}_k=\frac{\text{TP}_k}{\text{TP}_k+\text{FN}_k},\quad
  \text{F1}_k=\frac{2\cdot\text{Precision}_k\cdot\text{Recall}_k}{\text{Precision}_k+\text{Recall}_k}.
  $$
- **Macro / Weighted averages** (equal classes vs support-weighted).  
- **ROC AUC (OvR, macro):** AUC per class from continuous scores, averaged.  
- **$R^2$ (macro):** OVR regressions’ $R^2$ averaged across classes:
  $$
  R^2_k=1-\frac{\sum_i (y^{(k)}_i-\hat{y}^{(k)}_i)^2}{\sum_i (y^{(k)}_i-\bar{y}^{(k)})^2},\quad
  R^2_\text{macro}=\tfrac{1}{K}\sum_k R^2_k.
  $$

**Interpreting gaps:**
- Train $\gg$ Test → overfitting.  
- Macro $\ll$ Weighted → some minority classes perform worse.  
- High ROC AUC but lower F1 → rank quality is good, but thresholded decisions (argmax) still err; consider calibration.

---

## 13.3.1 Macro vs Weighted Averaging — Deep Dive

When we evaluate multiclass models, we compute metrics like **precision**, **recall**, and **F1** per class.  
But how we **average** across those classes matters — a *lot*.

---

### Why averaging matters

Imagine a dataset with **class imbalance**:

| Class | # Samples |
|--------|------------|
| Setosa | 50 |
| Versicolor | 50 |
| Virginica | 1000 |

If our model performs very well on the dominant class (Virginica) but poorly on the small ones,  
the overall “average F1” can look *artificially high* if we weight by class size.

That’s where **macro** and **weighted** averaging come in.

---

### Macro Average (Unweighted)

**Definition:**
Each class contributes **equally**, regardless of its frequency.

$$
\text{Macro-F1} = \frac{1}{K}\sum_{k=1}^{K} \text{F1}_k
$$

**Interpretation:**

- Treats all classes as equally important.
- Reflects *balanced* performance.
- Sensitive to small-class performance.

Yes Good when you care about fairness or minority class quality.  
No May underrepresent dominant classes if you truly care about aggregate performance.

---

### Weighted Average (Support-Weighted)

**Definition:**
Each class’s contribution is proportional to its **support** (number of true samples $n_k$):

$$
\text{Weighted-F1} = \sum_{k=1}^{K} \frac{n_k}{\sum_j n_j} \text{F1}_k
$$

**Interpretation:**

- Heavily influenced by large classes.
- Reflects “global” dataset performance.

Yes Good when overall accuracy matters.  
No Can mask poor performance on rare classes.

---

### Intuition Summary

| Metric Type | Weights by | Emphasizes | Good for |
|--------------|-------------|-------------|-----------|
| **Macro** | Equal across classes | Minority classes | Balanced performance, fairness |
| **Weighted** | Class frequency | Major classes | Overall performance, skewed data |

---

### Quick Example

| Class | Support ($n_k$) | F1$_k$ |
|--------|------------------|--------|
| Setosa | 50 | 0.95 |
| Versicolor | 50 | 0.90 |
| Virginica | 1000 | 0.50 |

Compute:

**Macro F1:**
$$
\frac{0.95 + 0.90 + 0.50}{3} = 0.78
$$

**Weighted F1:**
$$
\frac{50(0.95) + 50(0.90) + 1000(0.50)}{1100} = 0.53
$$

→ Weighted average drops sharply because the dominant class (Virginica) has low F1.

---

### Code Snippet: Computing Macro vs Weighted

```python
from sklearn.metrics import precision_recall_fscore_support

prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred)
macro_f1 = f1.mean()                           # Macro
weighted_f1 = (f1 * support).sum() / support.sum()  # Weighted

print(f"Macro F1: {macro_f1:.3f}, Weighted F1: {weighted_f1:.3f}")
```

### Practical Guidelines

| **Situation** | **Use Macro** | **Use Weighted** |
|----------------|---------------|------------------|
| Class imbalance present | Yes | (can hide poor minor class perf.) |
| Balanced dataset | Yes (similar to weighted) | Yes |
| You care about fairness | Yes | No |
| You care about total accuracy | No | Yes |

---

### Visual Comparison

Plot F1$_k$ for each class, plus macro and weighted averages:

```python
import matplotlib.pyplot as plt

classes = ["Setosa", "Versicolor", "Virginica"]
plt.bar(classes, f1, label="Per-class F1")
plt.axhline(macro_f1, color="red", linestyle="--", label="Macro F1")
plt.axhline(weighted_f1, color="green", linestyle="--", label="Weighted F1")
plt.legend()
plt.ylabel("F1 Score")
plt.title("Macro vs Weighted F1 Comparison")
plt.show()
```

You’ll clearly see how **weighted F1** tracks majority-class performance,  
while **macro F1** tracks average-class balance.

---

### Key Takeaways

- **Macro average** = “How good am I *on average* across classes?”  
- **Weighted average** = “How good am I *overall*, considering class frequency?”  
- Always report **both** when datasets are **imbalanced**.  
- Combine with the **confusion matrix** for detailed class-level insight.  


### 13.4 Stratified K-Fold Cross-Validation — No Leakage

- **StratifiedKFold** preserves class ratios in each fold.  
- Fit `StandardScaler` and `PCA` **inside** each fold on the **training** split only.  
- Report mean ± std across folds for each metric (No PCA vs PCA).

**Why it matters:** Reduces variance in estimates and prevents optimistic bias from leakage.

---

## 14. Decision Regions & Intuition

Train in PCA(2) space and plot decision regions (color-coded).  
**Simpler, smoother** regions suggest PCA found directions that align with class structure.  
If boundaries are jagged or tangled, consider more PCs or supervised reductions (LDA).

---

## 15. Practical Tips and Pitfalls

- **Scale then PCA.**
- **Pick $k$** via EVR or validation; avoid over-compressing.
- **Check outliers** — robust scaling can help.
- **Consider Ridge** for OVR stability.
- **Whitening** cautiously; tiny eigenvalues amplify noise.
- **Reproducibility:** fix random seeds; report mean ± std via CV.
- **Interpret loadings** to understand which original features drive PCs.

---

## 16. Summary & Further Exercises

**Summary:**

- PCA from centering → covariance → eigenvectors/eigenvalues.
- Rayleigh quotient & Lagrange multipliers derivation.
- PCA–SVD connection and low-rank reconstruction (Eckart–Young).
- Why eigenpairs correspond to meaningful directions (variance & structure).
- Fisher separability and when PCA aids discrimination.
- OVR linear regression: concept, math, pros/cons, and code.
- Full evaluation: $R^2$, Precision/Recall/F1, ROC AUC, Confusion Matrix, and stratified k-fold CV.

**Exercises:**

1. On Iris, vary `n_components` = 1, 2, 3 and compare metrics and Fisher scores.
2. Replace OVR Linear Regression with `LogisticRegression` or `Ridge` and compare ROC AUC and F1.
3. Plot **per-class ROC curves** from the decision scores.
4. Try PCA on a higher-dimensional dataset (Wine, Digits) and examine EVR and Fisher alignment.
5. Explore **whitening** and observe effects on classifiers.
6. Compute **loadings** and interpret which original features dominate PC1/PC2.
7. Implement **power iteration** to approximate PC1 and compare to SVD results.
