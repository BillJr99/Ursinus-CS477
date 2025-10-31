<!--
author:   William M. Mongan
language: en
narrator: US English Male

comment: Render with https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-unsupervised.md or locally if deployed via https://www.billmongan.com/LiaScript/?https://raw.githubusercontent.com/BillJr99/Ursinus-CS477/gh-pages/_pages/Activities/liascript-unsupervised.md

import: https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md

link:   https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-4
        https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Unsupervised Learning: GMM, PCA, K-Means, and GAN

**A comprehensive, example-driven walkthrough**  
For each algorithm we present **intuition**, **algorithm steps**, **full derivation**, **derivation explained (line by line)**, **from-scratch code**, **library code**, **line-by-line commentary**, **visualization interpretation**, and **design rationale**. Code & commentary are **interleaved** throughout (code → explanation → code → explanation).

1. **Gaussian Mixture Model (GMM)** — soft clustering via probabilistic mixtures and EM.
2. **Principal Component Analysis (PCA)** — dimensionality reduction via eigendecomposition/SVD.
3. **K-Means Clustering** — hard clustering via Lloyd’s algorithm.
4. **Generative Adversarial Network (GAN)** — generator vs discriminator in a minimax game.

---

## Open Colab: Companion Notebook

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Unsupervised_Learning_Colab.ipynb)

---

# Part I — Gaussian Mixture Models (GMM)

## 1. Concept & Dataset

**Intuition.** Many datasets contain hidden subpopulations. A **Gaussian Mixture Model** assumes data are generated from one of $K$ Gaussians with different means/covariances. Unlike K-Means, GMM yields **soft** cluster membership probabilities.

**Example dataset (used throughout GMM):** Two overlapping 2D Gaussian blobs.

---

## 2. Mathematical Formulation

Assume $x_i \in \mathbb{R}^d$, with latent component $z_i \in \{1,\dots,K\}$. The mixture density is
$$
p(x) \;=\; \sum_{k=1}^{K} \pi_k \, \mathcal{N}(x \mid \mu_k, \Sigma_k),\qquad \sum_k \pi_k=1,\ \pi_k\ge0.
$$

**Responsibilities (posterior component probabilities):**
$$
\gamma_{ik} \equiv p(z_i = k \mid x_i) \;=\;
\frac{\pi_k \, \mathcal{N}(x_i \mid \mu_k,\Sigma_k)}{\sum_{j=1}^{K}\pi_j \,\mathcal{N}(x_i \mid \mu_j,\Sigma_j)}.
$$

**EM updates:**
$$
N_k = \sum_{i=1}^n \gamma_{ik},\quad
\pi_k \leftarrow \frac{N_k}{n},\quad
\mu_k \leftarrow \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik}\,x_i,\quad
\Sigma_k \leftarrow \frac{1}{N_k}\sum_{i=1}^n \gamma_{ik}(x_i-\mu_k)(x_i-\mu_k)^\top.
$$

### Reader’s guide to the formulation (context)
- **Mixture idea.** We model the overall density as a **weighted sum** of simpler densities (Gaussians). Each weight $\pi_k$ is how common component $k$ is.  
- **What the normal pdf contributes.** $\mathcal{N}(x\mid \mu_k,\Sigma_k)$ is the **likelihood** of $x$ under bell $k$; elongated ellipses come from $\Sigma_k$.  
- **Soft membership.** $\gamma_{ik}$ is the **probability** that $x_i$ came from component $k$, not a hard yes/no.  
- **Sufficient statistics.** $N_k,\mu_k,\Sigma_k$ are exactly the weighted counts/means/covariances you would compute if labels were known; EM pretends they are known **in expectation**.  
- **Why sums and normalization?** The denominator in $\gamma_{ik}$ ensures posterior probabilities across components sum to 1 for each point.  

---

## 3. Derivation (Likelihood $\to$ EM)

We use one-hot indicators $z_{ik}\in\{0,1\}$. The **complete-data** log-likelihood is
$$
\ell_c(\Theta) \;=\; \sum_{i=1}^n \sum_{k=1}^K
z_{ik}\Big(\log \pi_k + \log \mathcal{N}(x_i \mid \mu_k,\Sigma_k)\Big).
$$

Define
$$
Q(\Theta \mid \Theta^{(t)}) \;\equiv\; \mathbb{E}_{z\mid x,\Theta^{(t)}}[\ell_c(\Theta)]
= \sum_{i,k} \gamma_{ik}^{(t)}\Big(\log \pi_k + \log \mathcal{N}(x_i \mid \mu_k,\Sigma_k)\Big).
$$

**E-step:** compute $\gamma_{ik}^{(t)} = p(z_i=k\mid x_i;\Theta^{(t)})$.  
**M-step:** maximize $Q$ w.r.t. $\pi_k,\mu_k,\Sigma_k$ under $\sum_k \pi_k=1$, yielding the updates above.  
Each iteration increases the observed log-likelihood
$$
\log p(X\mid\Theta) = \sum_{i=1}^n \log\left(\sum_{k=1}^{K} \pi_k \mathcal{N}(x_i\mid\mu_k,\Sigma_k)\right).
$$

### Reader’s guide to the derivation (context)
- **Why “complete-data”?** If we **knew** latent labels $z$, fitting each Gaussian would be trivial. EM takes the **expectation** over unknown $z$ instead.  
- **$Q$-function purpose.** $Q$ is the **expected complete-data log-likelihood** under the current parameters. Maximizing it moves parameters in a direction that improves the true log-likelihood.  
- **E-step intuition.** Compute soft labels $\gamma_{ik}$ via Bayes’ rule: prior $\pi_k$ times how well component $k$ explains $x_i$, renormalized.  
- **M-step as weighted MLE.** Updating $\pi_k,\mu_k,\Sigma_k$ is exactly maximum likelihood for Gaussians where each point contributes a **fractional count** $\gamma_{ik}$.  
- **Monotonic ascent.** EM guarantees $\log p(X\mid\Theta^{(t+1)})\ge \log p(X\mid\Theta^{(t)})$; in practice track it to diagnose convergence and singular covariances.

---

## 4. Derivation Explained (Line by Line, with Example)

**Line 1 — $\ell_c(\Theta)$ definition.** We pretend the latent labels $z_{ik}$ are known; then the likelihood splits by component, making the sum easy. In our 2-blob example, a point near the left ellipse contributes mostly to that component’s term.

**Line 2 — $Q(\Theta\mid\Theta^{(t)})$.** Since $z$ is unknown, take the expectation using current posteriors $\gamma_{ik}^{(t)}$. Intuitively, a point that is $70\%$ likely to be cluster 1 contributes $0.7$ of a “vote” to cluster 1’s statistics.

**Line 3 — E-step.** Compute $\gamma_{ik}^{(t)}$ by Bayes’ rule: prior $\pi_k$ times likelihood under $(\mu_k,\Sigma_k)$, normalized across $k$. In overlapping regions, $\gamma$ is soft ($\approx 0.5, 0.5$).

**Line 4 — M-step.** Maximize $Q$ under the mixing constraint. This yields weighted means/covariances: the left ellipse’s $\mu_1$ moves toward the weighted center of points with high $\gamma_{i1}$.

**Line 5 — Monotonicity.** Each EM step increases (or leaves unchanged) the observed log-likelihood until convergence for our two-blob data.

---

## 5. From-Scratch GMM — Code (A: Data & PDF)

```python
import numpy as np
import matplotlib.pyplot as plt

# Reproducibility
np.random.seed(42)

# --- Example dataset: two overlapping 2D Gaussians ---
n_per = 200
mean1, cov1 = np.array([0.0, 0.0]), np.array([[0.5, 0.0],[0.0, 0.5]])
mean2, cov2 = np.array([3.0, 3.0]), np.array([[0.6, 0.2],[0.2, 0.6]])
X = np.vstack([
    np.random.multivariate_normal(mean1, cov1, n_per),
    np.random.multivariate_normal(mean2, cov2, n_per)
])

# --- Multivariate Normal PDF (vectorized) ---
def mvn_pdf(X, mu, Sigma):
    d = X.shape[1]
    Xc = X - mu                        # center per row
    invS = np.linalg.inv(Sigma)        # inverse covariance
    expo = -0.5 * np.sum(Xc @ invS * Xc, axis=1)   # quadratic form
    norm = np.sqrt(((2*np.pi)**d) * np.linalg.det(Sigma))  # normalization
    return np.exp(expo) / norm
```

**Explanation (A).** We synthesize the two-blob dataset used for the EM walk-through. `mvn_pdf` evaluates $\mathcal{N}(x\mid\mu,\Sigma)$ row-wise using the quadratic form $(x-\mu)^\top\Sigma^{-1}(x-\mu)$ and the normalization constant.

---

## From-Scratch GMM — Code (B: EM Loop)

```python
def gmm_em(X, K=2, iters=60, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape

    # --- Initialization ---
    means = X[rng.choice(n, size=K, replace=False)]   # pick K data points
    covs = np.array([np.eye(d) for _ in range(K)])    # start with identity covariances
    weights = np.ones(K) / K                          # uniform mixture weights
    resp = np.zeros((n, K))                           # responsibilities (n x K)

    # --- EM iterations ---
    for _ in range(iters):
        # E-step: unnormalized responsibilities
        for k in range(K):
            resp[:, k] = weights[k] * mvn_pdf(X, means[k], covs[k])

        # Normalize per-row to get posteriors gamma_{ik}
        row_sum = resp.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1.0                   # avoid divide-by-zero
        resp /= row_sum

        # M-step: weighted re-estimation
        Nk = resp.sum(axis=0)                         # effective counts per component
        weights = Nk / n

        for k in range(K):
            # weighted mean
            means[k] = (resp[:, k, None] * X).sum(axis=0) / max(Nk[k], 1e-12)

            # weighted covariance
            Xc = X - means[k]
            covs[k] = (resp[:, k, None, None] *
                       np.einsum('ni,nj->nij', Xc, Xc)).sum(axis=0) / max(Nk[k], 1e-12)

            # numerical stability
            covs[k] += 1e-6 * np.eye(d)

    return weights, means, covs
```

**Explanation (B).**  
- **Init:** Random-point means; identity covariances; equal priors.  
- **E-step:** Compute $w_k \, \mathcal{N}(x\mid\mu_k,\Sigma_k)$, then row-normalize to get $\gamma_{ik}$.  
- **M-step:** Update $\pi_k=N_k/n$; $\mu_k$ as weighted mean; $\Sigma_k$ as weighted covariance; add tiny diagonal for conditioning.

---

## From-Scratch GMM — Code (C: Fit & Visualize)

```python
weights, means, covs = gmm_em(X, K=2, iters=60, seed=123)

# Posteriors for hard assignments (only for plotting convenience)
post = np.stack([weights[k] * mvn_pdf(X, means[k], covs[k]) for k in range(2)], axis=1)
post /= post.sum(axis=1, keepdims=True)
labels = np.argmax(post, axis=1)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=12)
plt.scatter(means[:, 0], means[:, 1], marker='x', s=100)
plt.title("GMM (From Scratch): Hard Assignments")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.show()
```

**Explanation (C).** We recompute posteriors with the learned parameters and use $\arg\max$ for a crisp plot. Crosses mark learned means near blob centers.

---

## 6. GMM with `scikit-learn`

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=42)
gmm.fit(X)                      # EM under the hood
labels_skl = gmm.predict(X)     # hard assignments
means_skl = gmm.means_
```

**Explanation.** `GaussianMixture` implements EM with robust options (`covariance_type`, `reg_covar`, `init_params`). `predict_proba(X)` returns responsibilities.

```python
plt.figure()
plt.scatter(X[:,0], X[:,1], c=labels_skl, s=12)
plt.scatter(means_skl[:,0], means_skl[:,1], marker='x', s=100)
plt.title("GMM (scikit-learn): Hard Assignments")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.show()
```

**Explanation.** Expect near-identical cluster structure to the from-scratch fit on this dataset.

---

# Part II — Principal Component Analysis (PCA)

## 1. Concept & Dataset

**Intuition.** PCA finds orthogonal directions that capture maximal variance and projects data onto the top directions.  
**Example dataset (PCA):** 3D correlated features collapsed to 2D for visualization.

---

## 2. Mathematical Formulation

Given centered $X\in\mathbb{R}^{n\times d}$, covariance
$$
\Sigma \;=\; \frac{1}{n-1}X^\top X.
$$
Eigendecompose $\Sigma = V\Lambda V^\top$ with eigenvalues $\lambda_1 \ge \lambda_2 \ge \dots$.  
Project onto first $m$ PCs:
$$
X_{\text{proj}} \;=\; X \, V_{[:,1:m]}.
$$
Explained variance ratio: $\lambda_j / \sum_i \lambda_i$.

### Reader’s guide to the formulation (context)
- **Centering matters.** Using $X$ **minus its column means** avoids spurious variance from offsets; otherwise PC1 might point toward the mean.  
- **Covariance as energy map.** $\Sigma$ records **pairwise co-variation**; large off-diagonals indicate strong linear relationships.  
- **Eigenpairs.** Eigenvectors in $V$ are directions; eigenvalues in $\Lambda$ quantify **how much variance** each direction carries.  
- **Projection.** Multiplying by $V_{[:,1:m]}$ keeps only the top-$m$ variance axes, reducing dimension while preserving most energy.  
- **Explained variance ratio.** The fraction $\lambda_j/\sum_i\lambda_i$ is a **budget** telling how much information each PC retains.

---

## 3. Derivation (Variance Maximization)

We seek the unit vector $v$ that maximizes projected variance:
$$
\max_{\lVert v\rVert=1}\ \operatorname{Var}(Xv) = \max_{\lVert v\rVert=1} v^\top \Sigma v.
$$
Lagrangian $L(v,\lambda)=v^\top \Sigma v - \lambda(v^\top v - 1)$ leads to the stationarity condition
$$
\Sigma v = \lambda v,
$$
so $v$ is an eigenvector of $\Sigma$ with eigenvalue $\lambda$. Subsequent PCs follow by orthogonality constraints.

### Reader’s guide to the derivation (context)
- **Optimization target.** We select the **direction** $v$ with maximum **spread** after projection; unit norm removes trivial scaling.  
- **Lagrange multiplier role.** $\lambda$ enforces $\lVert v\rVert=1$ while we optimize; it becomes the eigenvalue at optimum.  
- **Stationarity ⇒ eigenproblem.** Setting the gradient of $L$ to zero yields $\Sigma v=\lambda v$, revealing PCs as eigenvectors.  
- **Orthogonality.** Constraining later components to be orthogonal to earlier ones prevents re-capturing the same variance.  
- **SVD equivalence.** In practice we often compute via SVD of $X$; the right singular vectors equal $V$ for centered data.

---

## 4. Derivation Explained (Line by Line, with Example)

**Line 1 — Objective $v^\top\Sigma v$.** This equals the variance of $X$ after projection onto direction $v$. In our correlated 3D example, the $v$ that aligns with the longest elongation maximizes variance.

**Line 2 — Unit-norm constraint.** Without $\lVert v\rVert=1$, scaling $v$ would scale variance arbitrarily. The constraint ensures we compare *directions* not magnitudes.

**Line 3 — Lagrangian.** Introduce $\lambda$ to enforce the constraint during optimization.

**Line 4 — Stationarity $\Sigma v=\lambda v$.** Taking derivatives and setting to zero yields the eigenproblem. Eigenvectors give principal directions; eigenvalues measure captured variance.

**Line 5 — Subsequent PCs.** Enforce orthogonality to already selected PCs, capturing remaining variance in descending order.

---

## 5. From-Scratch PCA — Code (A: Data & Covariance)

```python
import numpy as np
import matplotlib.pyplot as plt

# Synthetic 3D data with strong correlation
np.random.seed(0)
n = 200
f1 = np.random.normal(0, 1, n)
f2 = 2 * f1 + np.random.normal(0, 0.5, n)
f3 = f1 - f2 + np.random.normal(0, 0.2, n)
X = np.column_stack([f1, f2, f3])

# Center columns (zero mean)
Xc = X - X.mean(axis=0)

# Sample covariance
Cov = np.cov(Xc, rowvar=False)
```

**Explanation (A).** We construct correlated features so PCA has a clear PC1. Centering is essential so covariance reflects variation around zero.

---

## From-Scratch PCA — Code (B: Eigen, Projection, Plot)

```python
# Eigen decomposition (symmetric positive semi-definite)
evals, evecs = np.linalg.eigh(Cov)

# Sort eigenvalues/vectors descending
idx = np.argsort(evals)[::-1]
evals, evecs = evals[idx], evecs[:, idx]

# Project onto first two principal components
X_proj = Xc @ evecs[:, :2]

# Plot
plt.figure()
plt.scatter(X_proj[:, 0], X_proj[:, 1], s=12)
plt.title("PCA (From Scratch): First 2 PCs")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()
```

**Explanation (B).** `eigh` exploits symmetry for numerical stability. Sorting ensures we take PCs in descending variance order. Projection multiplies by the first two eigenvectors.

---

## 6. PCA with `scikit-learn`

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
X_proj_skl = pca.fit_transform(X)       # centers internally
ratio = pca.explained_variance_ratio_
```

**Explanation.** `fit_transform` learns PCs and projects. `explained_variance_ratio_` guides dimension selection.

```python
plt.figure()
plt.scatter(X_proj_skl[:, 0], X_proj_skl[:, 1], s=12)
plt.title("PCA (scikit-learn): First 2 PCs")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()

plt.figure()
plt.bar([1, 2], ratio[:2])
plt.title("PCA: Explained Variance Ratio (First 2 PCs)")
plt.xlabel("Principal Component"); plt.ylabel("Explained Variance Ratio")
plt.show()
```

**Explanation.** Expect PC1 to capture most variance given how $f_2$ depends on $f_1$.

---

# Part III — K-Means Clustering

## 1. Concept & Dataset

**Intuition.** K-Means partitions data into $K$ clusters by minimizing within-cluster squared distances.  
**Example dataset:** Three compact 2D blobs.

---

## 2. Mathematical Formulation & Algorithm

Objective:
$$
\min_{\{C_k\},\{\mu_k\}}
\sum_{k=1}^{K} \sum_{x_i \in C_k} \lVert x_i - \mu_k \rVert^2.
$$

**Lloyd’s algorithm (iterate):**
1. **Assignment:** $c_i \leftarrow \arg\min_k \lVert x_i - \mu_k \rVert^2$  
2. **Update:** $\mu_k \leftarrow \frac{1}{|C_k|}\sum_{x_i \in C_k} x_i$

Converges to a local optimum; sensitive to initialization (use **k-means++**).

### Reader’s guide to the formulation (context)
- **What are we minimizing?** The **total squared reconstruction error** if every point is approximated by its cluster center.  
- **Two unknowns.** We jointly choose **labels** ($C_k$) and **centers** ($\mu_k$); the algorithm alternates because optimizing both at once is hard.  
- **Geometry.** Squared Euclidean distance yields **linear decision boundaries** (Voronoi cells) between centers.  
- **Assumptions.** Works best when clusters are **spherical**, similar size, and separable in Euclidean geometry.  
- **Initialization.** k-means++ spreads initial centers to reduce bad local minima.

---

## 3. Derivation Explained (Line by Line, with Example)

**Line 1 — Objective.** Sum of squared distances within clusters; promotes compact clusters. In our three-blob data, each blob becomes one cluster minimizing its internal spread.

**Line 2 — Assignment step.** With fixed centers, the best label for $x_i$ is the nearest center; this strictly decreases (or maintains) the objective.

**Line 3 — Update step.** With fixed labels, the center that minimizes within-cluster SSE is the mean of cluster points.

**Line 4 — Alternation.** Repeating 2–3 monotonically decreases the objective until convergence (local minimum).

### Reader’s guide to the derivation (context)
- **Assignment optimality.** Given fixed centers, the objective decouples by point; nearest-center labeling is optimal for squared distance.  
- **Mean as minimizer.** Taking the derivative of $\sum\lVert x_i-\mu\rVert^2$ and setting to zero gives $\mu=\text{average}$.  
- **Local vs global optimum.** Lloyd’s algorithm is **greedy**; multiple initializations help escape poor partitions.  
- **Empty clusters.** If a center gets no points, reinitialize it (or keep the previous center); practical guardrails matter.  
- **Complexity.** Each iteration is $O(nKd)$; mini-batch variants reduce cost for large $n$.

---

## 4. From-Scratch K-Means — Code (A: Data & Init)

```python
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

# Example dataset: three blobs
Ctrue = np.array([[0, 0], [5, 0], [2.5, 4]])
m = 120
A = np.random.randn(m, 2) + Ctrue[0]
B = np.random.randn(m, 2) + Ctrue[1]
C = np.random.randn(m, 2) + Ctrue[2]
X = np.vstack([A, B, C])

def kmeans(X, K=3, iters=100, seed=0):
    rng = np.random.default_rng(seed)
    n = X.shape[0]

    # Initialize centroids by sampling points
    cents = X[rng.choice(n, size=K, replace=False)]
```

**Explanation (A).** Build a simple 2D dataset; pick initial centers from points (basic init).

---

## From-Scratch K-Means — Code (B: Assign & Update)

```python
    for _ in range(iters):
        # Compute squared distances to each centroid (n x K matrix)
        d = ((X[:, None, :] - cents[None, :, :]) ** 2).sum(axis=2)

        # Assign each point to its nearest centroid
        labels = np.argmin(d, axis=1)

        # Update centroids as means of assigned points
        new_cents = np.array([
            X[labels == k].mean(axis=0) if np.any(labels == k) else cents[k]
            for k in range(K)
        ])

        # Convergence check: no significant change
        if np.allclose(cents, new_cents):
            cents = new_cents
            break

        cents = new_cents

    return cents, labels
```

**Explanation (B).** Vectorized distances; nearest-centroid assignment; average to update; guard empty clusters; stop when stable.

---

## From-Scratch K-Means — Code (C: Fit & Plot)

```python
centroids, labels = kmeans(X, K=3, iters=100, seed=123)

plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=labels, s=12)
plt.scatter(centroids[:, 0], centroids[:, 1], marker='x', s=120)
plt.title("K-Means (From Scratch): Cluster Assignments")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.show()
```

**Explanation (C).** Colors show final assignments; crosses mark learned centroids. Boundaries are piecewise linear (Voronoi).

---

## 5. K-Means with `scikit-learn`

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, n_init=10, random_state=42)
km.fit(X)                     # runs Lloyd with multiple random inits
labs = km.labels_
centers = km.cluster_centers_
```

**Explanation.** `n_init` restarts reduce sensitivity to initialization; sklearn uses k-means++ by default in recent versions.

```python
plt.figure()
plt.scatter(X[:,0], X[:,1], c=labs, s=12)
plt.scatter(centers[:,0], centers[:,1], marker='x', s=120)
plt.title("K-Means (scikit-learn): Cluster Assignments")
plt.xlabel("Feature 1"); plt.ylabel("Feature 2")
plt.show()
```

**Explanation.** Expect centroids near true blob centers.

---

# Part IV — Generative Adversarial Networks (GANs)

## 1. Concept & Dataset

**Intuition.** A **generator** $G(z)$ maps simple noise $z$ to realistic samples; a **discriminator** $D(x)$ distinguishes real vs generated. They co-train in a **minimax game**.  
**Example dataset (GAN):** 1D standard normal target distribution.

---

## 2. Mathematical Formulation

Minimax objective:
$$
\min_G \max_D\;\; \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)]
+ \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))].
$$

Non-saturating generator objective (commonly used):
$$
\max_G\;\; \mathbb{E}_{z\sim p_z}[\log D(G(z))].
$$

For fixed $G$, the optimal discriminator is
$$
D^*(x) = \frac{p_{\text{data}}(x)}{p_{\text{data}}(x) + p_G(x)}.
$$

### Reader’s guide to the formulation (context)
- **Two-player game.** $D$ is a classifier; $G$ is a generator whose outputs are judged by $D$. The objectives reflect opposing goals.  
- **Why logs?** Using $\log$ turns products into sums and corresponds to **cross-entropy** classification, stabilizing gradients.  
- **Non-saturating trick.** $\max_G \log D(G(z))$ avoids vanishing gradients when $D$ is initially strong.  
- **Optimal $D$.** $D^*$ expresses the **posterior probability** of “real” vs “fake” under densities $p_{\text{data}}$ and $p_G$.  
- **Goal.** When $p_G=p_{\text{data}}$, $D^*(x)=1/2$ everywhere — the equilibrium.

---

## 3. Derivation Explained (Line by Line, with Example)

**Line 1 — $\max_D$.** With $G$ fixed, the best $D$ pushes real examples to probability near 1 and fakes near 0; for our 1D Gaussian target, $D$ learns where the real histogram has more mass.

**Line 2 — $\min_G$.** $G$ improves by producing samples where $D$ outputs higher realness; in the 1D example, it shifts its mean/variance to match real data.

**Line 3 — Non-saturating loss.** Replacing $\min_G \log(1-D(G(z)))$ with $\max_G \log D(G(z))$ maintains strong gradients when $D$ is competent early in training.

**Line 4 — $D^*(x)$.** The Bayes-optimal discriminator returns the posterior of “real” vs “fake” given densities; when $p_G=p_{\text{data}}$, $D^*=1/2$ everywhere.

### Reader’s guide to the derivation (context)
- **Solving for $D$ given $G$.** For each $x$, maximize $\log D(x)$ for reals and $\log(1-D(x))$ for fakes; calculus yields $D^*(x)$.  
- **Plug back to see the divergence.** Substituting $D^*$ into the objective shows the game minimizes a **Jensen–Shannon divergence** between $p_{\text{data}}$ and $p_G$ (at a high level).  
- **Why alternating updates?** We cannot optimize both simultaneously easily; we **alternate** steps to track the moving target.  
- **Pathologies.** Mode collapse occurs when $G$ discovers a few modes that fool $D$; remedies include architectural/regularization tweaks (e.g., WGAN-GP).  
- **Evaluation.** In images, use FID or visual inspection; for 1D toy, compare histograms or empirical CDFs.

---

## 4. Minimal 1-D GAN — Code (A: Data, Models, Losses)

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

# Reproducibility & dataset
rng = np.random.default_rng(0)
num_real = 4000
real_data = rng.normal(0.0, 1.0, size=(num_real, 1)).astype("float32")

# Hyperparameters
batch_size = 128
epochs = 1000
noise_dim = 1
lr = 1e-3

# Generator: z -> x_fake
generator = tf.keras.Sequential([
    layers.Input(shape=(noise_dim,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1)
])

# Discriminator: x -> prob_real
discriminator = tf.keras.Sequential([
    layers.Input(shape=(1,)),
    layers.Dense(32, activation="relu"),
    layers.Dense(32, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])

# Optimizers & loss
d_opt = tf.keras.optimizers.Adam(learning_rate=lr)
g_opt = tf.keras.optimizers.Adam(learning_rate=lr)
bce = tf.keras.losses.BinaryCrossentropy(from_logits=False)
```

**Explanation (A).** Small MLPs suffice for a 1D target. The discriminator ends with a sigmoid (probability). Both use Adam with the same learning rate for simplicity.

---

## Minimal 1-D GAN — Code (B: Sampling Utilities & Steps)

```python
def sample_real():
    idx = rng.integers(0, num_real, size=batch_size)
    return real_data[idx]

def sample_noise():
    return rng.standard_normal(size=(batch_size, noise_dim)).astype("float32")

@tf.function
def train_discriminator_step():
    # Real batch and labels
    x_real = sample_real()
    y_real = tf.ones((batch_size, 1), dtype=tf.float32)

    # Fake batch and labels
    z = sample_noise()
    x_fake = generator(z, training=True)
    y_fake = tf.zeros((batch_size, 1), dtype=tf.float32)

    # Loss and update
    with tf.GradientTape() as tape:
        p_real = discriminator(x_real, training=True)
        p_fake = discriminator(x_fake, training=True)
        d_loss = bce(y_real, p_real) + bce(y_fake, p_fake)

    grads = tape.gradient(d_loss, discriminator.trainable_variables)
    d_opt.apply_gradients(zip(grads, discriminator.trainable_variables))
    return d_loss

@tf.function
def train_generator_step():
    # Generator tries to fool D: target labels are 1's
    z = sample_noise()
    with tf.GradientTape() as tape:
        x_fake = generator(z, training=True)
        p_fake = discriminator(x_fake, training=True)
        y_target = tf.ones((batch_size, 1), dtype=tf.float32)
        g_loss = bce(y_target, p_fake)

    grads = tape.gradient(g_loss, generator.trainable_variables)
    g_opt.apply_gradients(zip(grads, generator.trainable_variables))
    return g_loss
```

**Explanation (B).** Two `@tf.function` steps implement alternating updates: $D$ differentiates through real/fake predictions; $G$ differentiates through $D(G(z))$ using the non-saturating objective.

---

## Minimal 1-D GAN — Code (C: Training Loop & Plots)

```python
hist_d, hist_g = [], []
for step in range(epochs):
    d_loss = train_discriminator_step()
    g_loss = train_generator_step()
    hist_d.append(float(d_loss.numpy()))
    hist_g.append(float(g_loss.numpy()))
    if step % 100 == 0:
        print(f"Step {step:04d} | D loss: {hist_d[-1]:.4f} | G loss: {hist_g[-1]:.4f}")

# Loss curves
plt.figure()
plt.plot(hist_d, label="Discriminator loss"); plt.plot(hist_g, label="Generator loss")
plt.title("GAN Training Losses"); plt.xlabel("Training step"); plt.ylabel("Loss")
plt.legend(); plt.show()

# Compare distributions
z_many = rng.standard_normal(size=(5000, noise_dim)).astype("float32")
gen_many = generator(z_many, training=False).numpy().flatten()

plt.figure()
plt.hist(real_data.flatten(), bins=40, alpha=0.5, density=True)
plt.hist(gen_many, bins=40, alpha=0.5, density=True)
plt.title("GAN: Real vs Generated (1-D)"); plt.xlabel("Value"); plt.ylabel("Density")
plt.show()
```

**Explanation (C).** Alternating updates provide a simple and stable schedule for this toy task; loss curves should stabilize and histograms should overlap.

---

# Visualization Interpretation (All Algorithms)

- **GMM:** Elliptical clusters; soft responsibilities near overlap; learned means near blob centers.  
- **PCA:** PC1 aligns with maximal spread; EVR bar chart indicates retained variance.  
- **K-Means:** Piecewise-linear decision boundaries; centroids summarize clusters; watch for empty clusters.  
- **GAN:** Loss curves reflect adversarial balance; overlapping histograms indicate distributional alignment.

---

# Design Rationale & Practical Guidance

- **GMM vs K-Means:** GMM for anisotropic clusters and soft membership; K-Means for speed and spherical clusters.  
- **PCA preprocessing:** Always center; consider scaling when units differ; use randomized SVD for large $d$.  
- **GANs:** Start simple; use non-saturating loss; consider label smoothing/noise; sometimes train $D$ more often than $G$.

---

# Connections & Extensions

- **GMM:** Diagonal/spherical/tied covariances; Bayesian GMM (Dirichlet).  
- **PCA:** Kernel PCA, Probabilistic PCA, Sparse PCA, whitening.  
- **K-Means:** k-means++, Mini-Batch K-Means, K-Medoids, Spectral clustering.  
- **GAN:** WGAN(-GP), DCGAN, StyleGAN; alternative divergences and architectural priors.

---

# Case Studies: Datasets, Math ↔ Code Bridges

## GMM — Old Faithful Geyser (Eruptions vs Waiting)

**Why this dataset?** Classic bimodal structure (short-waiting short-eruption vs long-waiting long-eruption). A natural fit for a 2-component Gaussian mixture.

**Math ↔ Code bridge.**
- The E-step computes responsibilities $\gamma_{ik}$ (posterior component probabilities) via Bayes’ rule. In code, this is the **row-wise normalization** of `weights[k] * N(x_i | μ_k, Σ_k)`.
- The M-step updates $(\pi_k,\mu_k,\Sigma_k)$ with **responsibility-weighted** sufficient statistics; in code, this is the **weighted mean** and **weighted covariance**.

### Load & Standardize

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

# Old Faithful (two columns): 'eruptions' (minutes) and 'waiting' (minutes)
url = "https://raw.githubusercontent.com/vincentarelbundock/Rdatasets/master/csv/datasets/faithful.csv"
faithful = pd.read_csv(url)
Xf = faithful[["eruptions", "waiting"]].to_numpy().astype(float)

# Standardization improves conditioning for full-covariance estimation
scaler = StandardScaler()
Xfz = scaler.fit_transform(Xf)
```

**Explanation.** Standardization approximates sphericality and stabilizes $\Sigma_k$ inversion in the Gaussian pdf.

### Fit GMM (Full-Covariance) & Interpret

```python
from sklearn.mixture import GaussianMixture

gmm = GaussianMixture(n_components=2, covariance_type="full", random_state=0, reg_covar=1e-6)
gmm.fit(Xfz)

resp = gmm.predict_proba(Xfz)             # responsibilities  γ_{ik}
labs = resp.argmax(axis=1)                # hard labels for plotting
mus  = gmm.means_                         
Sig  = gmm.covariances_
pis  = gmm.weights_

plt.figure()
plt.scatter(Xfz[:,0], Xfz[:,1], c=labs, s=10)
plt.scatter(mus[:,0], mus[:,1], marker="x", s=120)
plt.title("GMM on Old Faithful (z-scored)")
plt.xlabel("eruptions (z)"); plt.ylabel("waiting (z)")
plt.show()
```

**Tie-back.**
- `resp[i,k] ≈ γ_{ik}` implements $p(z_i=k \mid x_i)$.
- `mus[k]` is the weighted mean $\mu_k = \frac{1}{N_k}\sum_i \gamma_{ik} x_i$.
- `Sig[k]` matches the weighted covariance formula in the M-step.

**Model selection (optional).** Evaluate BIC for $K\in\{1,\dots,4\}$ to justify $K=2$ on this dataset.

```python
bics = []
for K in range(1,5):
    m = GaussianMixture(n_components=K, covariance_type="full", random_state=0, reg_covar=1e-6)
    m.fit(Xfz)
    bics.append(m.bic(Xfz))

plt.figure()
plt.plot(range(1,5), bics, marker="o")
plt.title("GMM: BIC vs K (Old Faithful)")
plt.xlabel("K"); plt.ylabel("BIC")
plt.show()
```

---

## PCA — UCI Wine (Chemical Profiles → Latent Axes)

**Why this dataset?** Real chemical measurements of wines from three cultivars; PCA reveals latent axes such as overall concentration and correlated acids.

**Math ↔ Code bridge.**
- The sample covariance $\Sigma=\frac{1}{n-1}X^\top X$ underpins PCA; in code, `PCA`’s `components_` are the eigenvectors of $\Sigma$ (right singular vectors of the centered/scaled data).
- Explained variance $\lambda_j$ corresponds to `explained_variance_` and ratios to `explained_variance_ratio_`.

### Load, Scale, PCA

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_wine
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

wine = load_wine()
X = wine.data
y = wine.target
feat_names = wine.feature_names

# Center & scale features to equalize units before PCA
Xz = StandardScaler().fit_transform(X)

pca = PCA(n_components=2, random_state=0)
Z = pca.fit_transform(Xz)                 # projects onto top-2 eigenvectors
evr = pca.explained_variance_ratio_       # λ_j / Σ_i λ_i
V   = pca.components_                     # rows are principal directions v_j^T
```

**Explanation.** Scaling is critical because features (e.g., alcohol vs color intensity) differ in units; otherwise variance is dominated by the largest-scale feature.

### Plots & Loadings (Math → Interpretation)

```python
plt.figure()
plt.scatter(Z[:,0], Z[:,1], s=10)
plt.title("Wine PCA: Projection onto PC1–PC2")
plt.xlabel("PC1"); plt.ylabel("PC2")
plt.show()

plt.figure()
plt.bar(range(1, len(evr)+1), evr)
plt.title("Explained Variance Ratio")
plt.xlabel("Component index"); plt.ylabel("EVR")
plt.show()

# Loadings: contribution of original variables to PCs (entries of eigenvectors)
import pandas as pd
loadings = pd.DataFrame(V.T, index=feat_names, columns=[f"PC{j+1}" for j in range(V.shape[0])])
loadings_pc12 = loadings.iloc[:, :2].sort_values("PC1", ascending=False)
loadings_pc12.head(10)
```

**Tie-back.**
- The bars visualize $\lambda_j/\sum_i\lambda_i$.
- The loadings table are entries of $V$; large magnitude entries indicate variables strongly aligned with the eigen-direction $v_j$.

---

## K-Means — Handwritten Digits (Vector Quantization of 8×8 Images)

**Why this dataset?** Clustering handwritten digits (`sklearn.datasets.load_digits`) illuminates K-Means’ geometry (Voronoi partitions) and its use as a **prototype learner**.

**Math ↔ Code bridge.**
- The assignment step $c_i=\arg\min_k\lVert x_i-\mu_k\rVert^2$ appears as `labels_ = argmin(distances)` internally.
- The update step $\mu_k=\frac{1}{|C_k|}\sum_{x_i\in C_k}x_i$ is the centroid recomputation; in `sklearn`, these are `cluster_centers_`.

### Load, Flatten, Cluster

```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

digits = load_digits()
X = digits.data.astype(float)  # shape (n_samples, 64)
y = digits.target

K = 10
km = KMeans(n_clusters=K, n_init=20, random_state=0)
km.fit(X)
labs = km.labels_
centers = km.cluster_centers_  # 10 prototypes in 64-D

sil = silhouette_score(X, labs)  # geometric tightness/separation
print("Silhouette:", sil)
```

### Visualize Prototypes (Centroids as Images)

```python
fig = plt.figure()
for k in range(K):
    ax = fig.add_subplot(2,5,k+1)
    ax.imshow(centers[k].reshape(8,8))
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("K-Means Prototypes (Digits)")
plt.show()
```

**Tie-back.**
- Each centroid is the **minimizer** of $\sum_{x_i\in C_k}\lVert x_i-\mu\rVert^2$ (derivative $\to$ set to zero $\Rightarrow$ mean).
- Silhouette reflects the objective’s **geometric** consequences: higher silhouettes imply tighter within-cluster SSE relative to between-cluster distances.

**Optional:** Compare to GMM on the same data; anisotropic clusters in pixel space may favor mixtures.

---

## GAN — MNIST (DCGAN-lite, 28×28 Grayscale)

**Why this dataset?** Ubiquitous image benchmark with sufficient diversity to demonstrate GAN training dynamics.

**Math ↔ Code bridge.**
- The discriminator estimates $D(x)\approx \frac{p_\text{data}(x)}{p_\text{data}(x)+p_G(x)}$; in code, this is the **sigmoid output** trained with **binary cross-entropy**.
- The non-saturating generator objective $\max_G \mathbb{E}_z[\log D(G(z))]$ is implemented by training $G$ against **target label 1** on $D(G(z))$.

### Load MNIST & Normalize

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers

(xtr, _), _ = tf.keras.datasets.mnist.load_data()
xtr = (xtr.astype("float32") / 127.5) - 1.0         # scale to [-1, 1]
xtr = np.expand_dims(xtr, -1)                       # (N, 28, 28, 1)

batch_size = 128
z_dim = 100
```

### Define Generator & Discriminator

```python
def make_generator():
    z = layers.Input(shape=(z_dim,))
    x = layers.Dense(7*7*128)(z)
    x = layers.Reshape((7,7,128))(x)
    x = layers.Conv2DTranspose(64, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2DTranspose(32, 4, strides=2, padding="same", activation="relu")(x)
    out = layers.Conv2D(1, 3, padding="same", activation="tanh")(x)
    return tf.keras.Model(z, out)

def make_discriminator():
    x_in = layers.Input(shape=(28,28,1))
    x = layers.Conv2D(32, 4, strides=2, padding="same", activation="relu")(x_in)
    x = layers.Conv2D(64, 4, strides=2, padding="same", activation="relu")(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation="relu")(x)
    out = layers.Dense(1, activation="sigmoid")(x)
    return tf.keras.Model(x_in, out)

G = make_generator()
D = make_discriminator()

d_opt = tf.keras.optimizers.Adam(1e-4)
g_opt = tf.keras.optimizers.Adam(1e-4)
bce = tf.keras.losses.BinaryCrossentropy()
```

### Training Step (Alternating Minimax with Non-saturating $G$)

```python
@tf.function
def train_step(x_real):
    # Sample noise
    z = tf.random.normal((tf.shape(x_real)[0], z_dim))

    # ---- Train D ----
    with tf.GradientTape() as tape_d:
        x_fake = G(z, training=True)
        p_real = D(x_real, training=True)
        p_fake = D(x_fake, training=True)
        d_loss = bce(tf.ones_like(p_real), p_real) + bce(tf.zeros_like(p_fake), p_fake)
    grads_d = tape_d.gradient(d_loss, D.trainable_variables)
    d_opt.apply_gradients(zip(grads_d, D.trainable_variables))

    # ---- Train G (non-saturating) ----
    z = tf.random.normal((tf.shape(x_real)[0], z_dim))
    with tf.GradientTape() as tape_g:
        x_fake = G(z, training=True)
        p_fake = D(x_fake, training=True)
        g_loss = bce(tf.ones_like(p_fake), p_fake)   # maximize log D(G(z))
    grads_g = tape_g.gradient(g_loss, G.trainable_variables)
    g_opt.apply_gradients(zip(grads_g, G.trainable_variables))

    return d_loss, g_loss
```

### Train & Monitor

```python
ds = tf.data.Dataset.from_tensor_slices(xtr).shuffle(10000).batch(batch_size).prefetch(2)

hist_d, hist_g = [], []
for epoch in range(5):   # start small for demonstration
    for xb in ds:
        dL, gL = train_step(xb)
        hist_d.append(float(dL.numpy())); hist_g.append(float(gL.numpy()))

plt.figure()
plt.plot(hist_d, label="D loss"); plt.plot(hist_g, label="G loss")
plt.title("DCGAN-lite on MNIST: Losses")
plt.xlabel("step"); plt.ylabel("loss")
plt.legend(); plt.show()

# Sample images
z = tf.random.normal((25, z_dim))
xg = (G(z, training=False).numpy() + 1.0) * 127.5
fig = plt.figure()
for i in range(25):
    ax = fig.add_subplot(5,5,i+1)
    ax.imshow(xg[i,...,0], cmap=None)    # no explicit color map
    ax.set_xticks([]); ax.set_yticks([])
plt.suptitle("Generated Samples (MNIST)")
plt.show()
```

**Tie-back.**
- The discriminator loss is the sum of two cross-entropies, matching the expectation terms in the minimax objective.
- The generator update uses targets of 1 for $D(G(z))$, implementing the non-saturating objective.

---

## Reproducibility & Ethics Notes

- **Determinism.** Fix random seeds where feasible; document library versions. Some GPU ops remain nondeterministic.  
- **Scaling.** For large datasets, consider mini-batch EM for GMM, randomized SVD for PCA, and mini-batch K-Means.  
- **Ethics.** When clustering people or sensitive attributes, scrutinize downstream uses, fairness, and interpretability.

---

# Appendix: Short “Math-to-Code” Checklists

- **GMM (EM):** compute pdfs $\to$ normalize (E) $\to$ weighted stats (M) $\to$ repeat; check log-likelihood monotonicity.  
- **PCA:** center/scale $\to$ covariance/SVD $\to$ eigenpairs sorted $\to$ projection $\to$ EVR.  
- **K-Means:** init $\mu_k$ $\to$ nearest-centroid labels $\to$ recompute means $\to$ repeat; track SSE.  
- **GAN:** alternate $D/G$ $\to$ cross-entropy losses $\to$ sample/evaluate; watch for mode collapse and instabilities.

---

# Appendix — Computing Eigenvalues and Eigenvectors of a Matrix $X$

> Scope: What eigenvalues/eigenvectors are, how to compute them **reliably**, how they relate to **PCA**, and which algorithms to use for **dense** vs **large/sparse** matrices. Examples are shown with NumPy/SciPy; adapt as needed.

---

## 1) Definitions & Basic Properties

Let $X \in \mathbb{R}^{d \times d}$ (or $\mathbb{C}^{d \times d}$). A nonzero vector $v$ and scalar $\lambda$ satisfy
$$
Xv = \lambda v
$$
iff $\lambda$ is an **eigenvalue** of $X$ and $v$ is a corresponding **eigenvector**.

- **Characteristic polynomial:** $\det(X-\lambda I)=0$ has roots $\{\lambda_i\}_{i=1}^d$.
- **Eigen-decomposition (diagonalizable $X$):**
  $$
  X = V \Lambda V^{-1},\quad \Lambda = \mathrm{diag}(\lambda_1,\ldots,\lambda_d),\ V=[v_1\,\cdots\,v_d].
  $$
- **Symmetric/Hermitian case ($X=X^\top$ or $X=X^*$):**
  - All eigenvalues are **real**.
  - There exists an **orthonormal** eigenbasis: $X = Q \Lambda Q^\top$ (or $Q^*$).
  - Numerically best-conditioned; use symmetry-aware solvers.

---

### Numerical Solution of Eigenvalues and Eigenvectors

To **compute eigenvalues and eigenvectors numerically**, we solve the **eigenvalue problem**:
$$
Xv = \lambda v.
$$
This equation can be rewritten as
$$
(X - \lambda I)v = 0,
$$
which has a **nontrivial solution** only when
$$
\det(X - \lambda I) = 0.
$$
Solving this determinant equation gives the eigenvalues $\lambda_1, \lambda_2, \dots, \lambda_d$; substituting each eigenvalue back yields the corresponding eigenvector(s).

---

#### Step-by-Step Numerical Procedure

1. **Form the characteristic equation**
   $$
   \det(X - \lambda I) = 0.
   $$
   This is a polynomial of degree $d$ in $\lambda$.  
   - For a $2\times2$ matrix, the equation is quadratic.  
   - For larger matrices, the polynomial is of higher order and is solved numerically.

2. **Find eigenvalues numerically**
   - For small matrices, one may expand the determinant explicitly and solve the polynomial.
   - For large matrices, eigenvalues are computed using **iterative numerical methods** such as:
     - **QR algorithm** (default in most libraries)
     - **Jacobi method** (for symmetric matrices)
     - **Power method** (for dominant eigenvalue)
     - **Lanczos/Arnoldi** (for large sparse matrices)
   These algorithms converge to the eigenvalues by successively refining approximate roots.

3. **Solve for eigenvectors**
   For each eigenvalue $\lambda_i$, substitute it into $(X - \lambda_i I)v_i = 0$ and solve the resulting homogeneous linear system for $v_i$.
   - This is equivalent to finding the **null space** of $(X - \lambda_i I)$.
   - Any nonzero scalar multiple of $v_i$ is also an eigenvector (they form a 1D eigenspace for simple eigenvalues).

4. **Normalize eigenvectors**
   Typically, each eigenvector is scaled to unit length:
   $$
   v_i \leftarrow \frac{v_i}{\|v_i\|}.
   $$

---

#### Example 

Given a $2\times2$ matrix
$$
X = \begin{bmatrix}
4 & 2 \\
1 & 3
\end{bmatrix},
$$
we find its eigenvalues and eigenvectors as follows.

1. **Characteristic equation:**
   $$
   \det(X - \lambda I)
   = \begin{vmatrix}
   4 - \lambda & 2 \\
   1 & 3 - \lambda
   \end{vmatrix}
   = (4 - \lambda)(3 - \lambda) - 2(1)
   = \lambda^2 - 7\lambda + 10.
   $$
   Set equal to zero:
   $$
   \lambda^2 - 7\lambda + 10 = 0.
   $$

2. **Solve for eigenvalues:**
   $$
   \lambda = \frac{7 \pm \sqrt{7^2 - 4(1)(10)}}{2}
   = \frac{7 \pm \sqrt{9}}{2}
   = \{5, 2\}.
   $$

3. **Find eigenvectors:**
   - For $\lambda_1 = 5$,
     $$
     (X - 5I)v = 0
     \Rightarrow
     \begin{bmatrix}
     -1 & 2 \\
     1 & -2
     \end{bmatrix}
     \begin{bmatrix}
     v_1 \\
     v_2
     \end{bmatrix}
     = 0
     \Rightarrow v_1 = 2v_2.
     $$
     One eigenvector is $v^{(1)} = \begin{bmatrix}2 \\ 1\end{bmatrix}$.

   - For $\lambda_2 = 2$,
     $$
     (X - 2I)v = 0
     \Rightarrow
     \begin{bmatrix}
     2 & 2 \\
     1 & 1
     \end{bmatrix}
     \begin{bmatrix}
     v_1 \\
     v_2
     \end{bmatrix}
     = 0
     \Rightarrow v_1 = -v_2.
     $$
     One eigenvector is $v^{(2)} = \begin{bmatrix}-1 \\ 1\end{bmatrix}$.

4. **Normalize eigenvectors (optional):**
   $$
   \hat{v}^{(1)} = \frac{1}{\sqrt{5}}\begin{bmatrix}2 \\ 1\end{bmatrix}, \qquad
   \hat{v}^{(2)} = \frac{1}{\sqrt{2}}\begin{bmatrix}-1 \\ 1\end{bmatrix}.
   $$

---

#### Summary of Numerical Solution Philosophy

- **Exact symbolic methods** (determinants, polynomials) are practical only for $d \leq 3$.
- **For larger matrices**, direct polynomial roots are numerically unstable; modern software instead uses:
  - **Orthogonal transformations** to upper-triangular (Schur form).
  - **Iterative refinement** to extract eigenvalues/eigenvectors.
- **Stability principle:** Algorithms (like QR) preserve orthogonality and minimize round-off errors.
- **Verification:** Check residuals $\|Xv_i - \lambda_i v_i\|$ and orthogonality $v_i^\top v_j \approx 0$.

---

#### Key Takeaway

Computing eigenvalues and eigenvectors numerically involves:
1. Reformulating the eigenproblem as $(X - \lambda I)v=0$,
2. Solving for $\lambda$ via stable iterative methods (e.g., QR algorithm),
3. Solving for $v$ as null-space vectors for each $\lambda$,
4. Normalizing and verifying results.

The process transforms an **abstract algebraic condition** into a **numerically stable iterative solution** grounded in linear algebra and matrix factorization theory.

---

### Singular Value Decomposition (SVD): Theory, Numerical Example, and Code Implementation

The **Singular Value Decomposition (SVD)** is one of the most fundamental tools in linear algebra and numerical analysis.  
For any real matrix $ X \in \mathbb{R}^{m \times n} $, there exist orthogonal matrices $ U \in \mathbb{R}^{m \times m} $ and $ V \in \mathbb{R}^{n \times n} $, and a diagonal matrix $ \Sigma \in \mathbb{R}^{m \times n} $ such that

$$
X = U \Sigma V^\top.
$$

---

#### Components of the Decomposition

- $ U $: **Left singular vectors** — orthonormal eigenvectors of $ X X^\top $  
- $ V $: **Right singular vectors** — orthonormal eigenvectors of $ X^\top X $  
- $ \Sigma $: **Singular values** — nonnegative square roots of eigenvalues of $ X^\top X $ (or $ X X^\top $)

$$
\Sigma = \mathrm{diag}(\sigma_1, \sigma_2, \ldots, \sigma_r),
\quad \text{where } \sigma_1 \ge \sigma_2 \ge \cdots \ge \sigma_r > 0,
$$
and $r = \mathrm{rank}(X)$.

**Interpretation:**
- Each $\sigma_i$ measures how much $X$ stretches the vector $v_i$ along direction $u_i$.
- The SVD expresses $X$ as a **sum of rank-one matrices**:
  $$
  X = \sum_{i=1}^r \sigma_i u_i v_i^\top.
  $$

---

#### Step-by-Step Numerical Example 

Consider
$$
X = 
\begin{bmatrix}
3 & 1 \\
1 & 3
\end{bmatrix}.
$$

#### Step 1 — Compute $ X^\top X $ and $ X X^\top $

$$
X^\top X = 
\begin{bmatrix}
3 & 1 \\
1 & 3
\end{bmatrix}
\begin{bmatrix}
3 & 1 \\
1 & 3
\end{bmatrix}
=
\begin{bmatrix}
10 & 6 \\
6 & 10
\end{bmatrix}.
$$

#### Step 2 — Eigenvalues and Right Singular Vectors

Compute eigenvalues of $ X^\top X $:
$$
\det(X^\top X - \lambda I) = 
\begin{vmatrix}
10-\lambda & 6 \\
6 & 10-\lambda
\end{vmatrix}
= (10-\lambda)^2 - 36 = 0.
$$
$$
\lambda_1 = 16,\quad \lambda_2 = 4.
$$
The corresponding eigenvectors (right singular vectors $v_i$) are:
$$
v_1 = \frac{1}{\sqrt{2}}\begin{bmatrix}1 \\ 1\end{bmatrix}, \quad
v_2 = \frac{1}{\sqrt{2}}\begin{bmatrix}1 \\ -1\end{bmatrix}.
$$

#### Step 3 — Singular Values

$$
\sigma_i = \sqrt{\lambda_i} \Rightarrow \sigma_1 = 4,\ \sigma_2 = 2.
$$
So
$$
\Sigma =
\begin{bmatrix}
4 & 0 \\
0 & 2
\end{bmatrix}.
$$

#### Step 4 — Left Singular Vectors

Compute $ u_i = \frac{1}{\sigma_i} X v_i $.

$$
u_1 = \frac{1}{4} X v_1
= \frac{1}{4}
\begin{bmatrix}
3 & 1 \\
1 & 3
\end{bmatrix}
\frac{1}{\sqrt{2}}\begin{bmatrix}1 \\ 1\end{bmatrix}
= \frac{1}{\sqrt{2}}\begin{bmatrix}1 \\ 1\end{bmatrix}.
$$

$$
u_2 = \frac{1}{2} X v_2
= \frac{1}{2}
\begin{bmatrix}
3 & 1 \\
1 & 3
\end{bmatrix}
\frac{1}{\sqrt{2}}\begin{bmatrix}1 \\ -1\end{bmatrix}
= \frac{1}{\sqrt{2}}\begin{bmatrix}1 \\ -1\end{bmatrix}.
$$

Hence,
$$
U = \frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix},\quad
\Sigma =
\begin{bmatrix}
4 & 0 \\
0 & 2
\end{bmatrix},\quad
V = \frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix}.
$$

**Verification:**
$$
X = U \Sigma V^\top =
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix}
\begin{bmatrix}
4 & 0 \\
0 & 2
\end{bmatrix}
\frac{1}{\sqrt{2}}
\begin{bmatrix}
1 & 1 \\
1 & -1
\end{bmatrix}
=
\begin{bmatrix}
3 & 1 \\
1 & 3
\end{bmatrix}.
$$

---

#### Computing SVD from Scratch (Code Example)

    ```python
    import numpy as np

    # Example matrix
    X = np.array([[3., 1.],
                  [1., 3.]])

    # Step 1: Compute symmetric matrices
    XtX = X.T @ X      # (n x n)
    XXt = X @ X.T      # (m x m)

    # Step 2: Eigen decomposition
    eigvals_V, V = np.linalg.eigh(XtX)   # eigenvalues ascending
    idx = np.argsort(eigvals_V)[::-1]    # sort descending
    eigvals_V = eigvals_V[idx]
    V = V[:, idx]

    # Step 3: Singular values
    S = np.sqrt(np.clip(eigvals_V, 0, None))

    # Step 4: Left singular vectors
    U = np.zeros_like(X)
    for i in range(len(S)):
        if S[i] > 1e-12:
            U[:, i] = (X @ V[:, i]) / S[i]

    # Step 5: Verification
    Sigma = np.diag(S)
    X_recon = U @ Sigma @ V.T

    print("Singular values:", S)
    print("Left singular vectors (U):\n", U)
    print("Right singular vectors (V):\n", V)
    print("Reconstruction:\n", X_recon)
    ```

**Explanation (Code):**
- Compute $ X^\top X $ and solve for its eigenpairs.
- Take square roots of eigenvalues to get singular values.
- Compute left singular vectors via normalization of $ Xv_i $.
- Verify $ X = U \Sigma V^\top $ numerically.

---

#### Interpretation & Applications

- **Dimensionality Reduction:** Keep only top $k$ singular values (Truncated SVD or PCA).
- **Noise Reduction / Compression:** Lower singular values represent less significant structure.
- **Pseudo-Inverse:** $ X^+ = V \Sigma^+ U^\top $.
- **Condition Number:** $ \kappa(X) = \sigma_{\max}/\sigma_{\min} $.
- **Latent Structure Discovery:** Used in Latent Semantic Analysis, recommender systems, and deep learning.

---

#### Key Takeaways

| Concept | Symbol | Meaning |
|----------|---------|----------|
| Singular Values | $ \sigma_i $ | Strength of each independent “mode” of variation |
| Left Singular Vectors | $ u_i $ | Basis in the output (row) space |
| Right Singular Vectors | $ v_i $ | Basis in the input (column) space |
| Orthogonality | $ U^\top U = I,\, V^\top V = I $ | Ensures numerical stability |
| Reconstruction | $ X = U\Sigma V^\top $ | Exact (or rank-$k$ approximation) decomposition |

---

**Summary:**  
SVD generalizes eigen-decomposition to **non-square** matrices, providing a numerically stable, orthogonal factorization that underpins PCA, low-rank approximation, and many machine learning algorithms.

---

## 2) PCA Connection (Why eigenpairs matter)

For **centered** data matrix $Y \in \mathbb{R}^{n \times d}$, the sample covariance is
$$
\Sigma = \frac{1}{n-1}Y^\top Y.
$$
If $\Sigma v_j=\lambda_j v_j$ with $\lambda_1\ge\cdots\ge\lambda_d$, then $v_j$ is the $j$-th principal direction and $\lambda_j$ is the variance captured along $v_j$.
- **Explained variance ratio:** $\mathrm{EVR}_j = \lambda_j / \sum_i \lambda_i$.
- **SVD equivalence:** If $Y=U\Sigma_{\text{svd}}V^\top$, then $Y^\top Y = V \Sigma_{\text{svd}}^2 V^\top$, so $V$ are PCA directions and $\lambda_i=\sigma_i^2/(n-1)$.

---

## 3) Reliable Numerical Workflows

### 3.1 Symmetric/Hermitian $X$ (preferred when applicable)
Use a solver that exploits symmetry for stability and orthonormal eigenvectors.

    import numpy as np

    # Ensure symmetry numerically if X should be symmetric:
    # X = 0.5 * (X + X.T)
    evals, evecs = np.linalg.eigh(X)   # eigenvalues ascending; columns of evecs are eigenvectors
    # Sort descending if desired:
    idx = np.argsort(evals)[::-1]
    evals = evals[idx]
    evecs = evecs[:, idx]

Why `eigh`? It is specialized for symmetric/Hermitian matrices, returning **real** eigenvalues and **orthonormal** eigenvectors with better numerical accuracy than general `eig`.

### 3.2 General (possibly non-symmetric) $X$
Use the general Schur/QR-based eigensolver.

    ```python
    import numpy as np

    evals, evecs = np.linalg.eig(X)    # eigenvalues can be complex; evecs columns align with evals
    # Optional: sort by magnitude or real part depending on application
    idx = np.argsort(-np.abs(evals))
    evals = evals[idx]
    evecs = evecs[:, idx]
    ```

Notes:
- Real $X$ can have **complex** eigenpairs (e.g., rotations).
- If $X$ is **defective** (not diagonalizable), you cannot form $V^{-1}XV=\Lambda$; numerical routines return Schur-factor data implicitly.

### 3.3 PCA via SVD (most robust for data)
Prefer SVD to avoid squaring the condition number when forming $Y^\top Y$.
    
    ```python
    import numpy as np

    # Center columns
    Yc = Y - Y.mean(axis=0, keepdims=True)

    # Economy SVD if n >= d (or full_matrices=False to save work/memory)
    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)

    # PCA directions and variances
    V = Vt.T                                      # columns are principal directions
    eigenvalues = (S**2) / (Yc.shape[0] - 1)      # variances per component
    evr = eigenvalues / eigenvalues.sum()         # explained variance ratios

    # Projection onto first m principal components
    m = 2
    Z = Yc @ V[:, :m]
    ```

---

## 4) Iterative Methods for Large/Sparse Problems

When $d$ is large or $X$ is sparse, compute only a few extreme eigenpairs.

### 4.1 Power Iteration (largest-magnitude eigenpair)
Converges if the dominant eigenvalue is unique in magnitude and the start vector has a component in its direction.

    ```python
    import numpy as np

    def power_iteration(X, iters=1000, tol=1e-9, seed=0):
        rng = np.random.default_rng(seed)
        v = rng.standard_normal(X.shape[1])
        v /= np.linalg.norm(v)
        lam_old = 0.0
        for _ in range(iters):
            w = X @ v
            v = w / np.linalg.norm(w)
            lam = v @ (X @ v)  # Rayleigh quotient
            if abs(lam - lam_old) < tol * max(1.0, abs(lam_old)):
                break
            lam_old = lam
        return lam, v
    ```

To target eigenvalues near a shift $\mu$, apply power iteration to $(X-\mu I)^{-1}$ (requires solves).

### 4.2 Lanczos / Arnoldi (multiple eigenpairs)
Use library routines for efficiency and robustness (Krylov subspaces).

    ```python
    # Symmetric sparse case (SciPy):
    from scipy.sparse.linalg import eigsh
    k = 5  # number of largest eigenpairs
    evals, evecs = eigsh(X, k=k, which='LM')  # 'LM' = largest magnitude
    ```

For non-symmetric sparse matrices, use `scipy.sparse.linalg.eigs`.

---

## 5) Verification & Diagnostics

- **Residual per pair:** $\|Xv - \lambda v\|_2$ should be small relative to $\|X\|_2\|v\|_2$.
- **Orthogonality (symmetric case):** $evecs^\top evecs \approx I$.
- **Reconstruction (diagonalizable case):** $\|X - V\Lambda V^{-1}\|$ small.
- **Sensitivity:** Non-normal matrices can have highly sensitive eigenvalues; pseudospectra provide insight.

---

## 6) Common Pitfalls & Remedies

- **PCA preprocessing:** Always **center**; **scale** if units differ greatly.
- **Ill-conditioning:** Prefer **SVD** over eigendecomposing $Y^\top Y$.
- **Complex pairs:** In real problems with rotations/shears, complex conjugate eigenpairs are expected.
- **Defectiveness:** Do not force a full eigenbasis when the matrix is defective; analyze via Schur form instead.

---

## 7) Worked Mini-Examples

### 7.1 Symmetric eigen-decomposition

    ```python
    import numpy as np

    X = np.array([[2.0, 1.0, 0.0],
                  [1.0, 2.0, 1.0],
                  [0.0, 1.0, 2.0]])

    evals, evecs = np.linalg.eigh(X)            # ascending
    idx = np.argsort(evals)[::-1]               # descending
    evals = evals[idx]; evecs = evecs[:, idx]

    residuals = np.linalg.norm(X @ evecs - evecs * evals, axis=0)
    orth_err = np.linalg.norm(evecs.T @ evecs - np.eye(evecs.shape[1]))

    print("Eigenvalues (desc):", evals)
    print("Residuals:", residuals)
    print("Orthonormality error:", orth_err)
    ```
    
### 7.2 PCA via SVD
    
    ```python
    import numpy as np

    rng = np.random.default_rng(0)
    n, d = 200, 3
    Y = rng.normal(size=(n, d))
    Yc = Y - Y.mean(axis=0, keepdims=True)

    U, S, Vt = np.linalg.svd(Yc, full_matrices=False)
    V = Vt.T
    eigvals = (S**2) / (n - 1)
    evr = eigvals / eigvals.sum()

    print("Top-2 EVR:", evr[:2])
    Z = Yc @ V[:, :2]  # 2D projection
    ```
    
---

## 8) Complexity (very rough orders)

- **Dense full eigen or SVD:** $O(d^3)$ for $d\times d$ (eigen), or $O(nd^2)$ for SVD on $n\times d$ with $n \ge d$.
- **Iterative top-$k$:** $O(k \cdot \text{mv-cost} \cdot \text{iters})$, where mv-cost is the cost of one matrix–vector multiply (excellent for sparse matrices).

---

## 9) Quick Checklists

- **Eigen (symmetric):** verify symmetry → `eigh` → sort if needed → residual & orthogonality checks.
- **Eigen (general):** `eig` → handle complex pairs → optional sorting criterion → residual checks.
- **PCA (robust):** center (and often scale) → SVD → directions $V$, variances $S^2/(n-1)$ → project $YV_m$.
- **Large/sparse:** `eigsh`/`eigs` or power/Lanczos/Arnoldi with shifts if targeting interior spectrum.
