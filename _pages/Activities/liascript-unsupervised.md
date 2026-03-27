# K-Means Clustering and Nearest Neighbors — From Scratch to scikit-learn
<!--
author:   William M. Mongan
language: en
narrator: US English Male

comment: Render with https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-unsupervised.md or locally if deployed via https://www.billmongan.com/LiaScript/?https://raw.githubusercontent.com/BillJr99/Ursinus-CS477/gh-pages/_pages/Activities/liascript-unsupervised.md

import: https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md

link:   https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-4
        https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap

-->

## About this course
We learn **K-Means Clustering** and **Nearest Neighbor Classification** step by step — starting from geometry and intuition, building through vectorized NumPy code, and finishing with real-world datasets and `scikit-learn` implementations.

We'll connect ideas from **unsupervised learning (K-Means)** and **instance-based learning (k-NN)**, explaining the math and code line-by-line.

---

## Table of Contents
1. Intuition: Clustering and Similarity
2. K-Means Objective and Lloyd’s Algorithm
3. Distance Computation — Loops vs Broadcasting
4. K-Means in Code (From Scratch)
5. K-Means with scikit-learn
6. Real Example: Wine Dataset
7. K-Means++ Initialization
8. Nearest-Centroid Classification
9. k-Nearest Neighbors (k-NN)
10. Comparing K-Means, Nearest Centroid, and k-NN
11. Choosing k and Validation
12. Summary & Exercises

---

## 1. Intuition: Clustering and Similarity

Before we classify, we often need to **find structure** in unlabeled data — that’s **clustering**.

**Goal:** Group points that are *similar* to each other.

- **Similarity** is usually measured via **distance** (Euclidean, Manhattan, cosine, etc.)
- **K-Means** seeks *compact* spherical clusters.

Imagine points in 2D — we want to draw circles grouping close points.

---

## 2. K-Means Objective and Lloyd’s Algorithm

### The Objective Function

We want to partition $n$ data points $\{x_1, \dots, x_n\}$ into $K$ clusters $C_1, \dots, C_K$, with centroids $\mu_1, \dots, \mu_K$, minimizing the **within-cluster sum of squares (WCSS)**:

$$
\min_{C_1, \dots, C_K, \mu_1, \dots, \mu_K} \sum_{k=1}^K \sum_{x_i \in C_k} \|x_i - \mu_k\|^2.
$$

Each term measures how far points are from their cluster mean.

### Lloyd’s Algorithm

1. **Initialize**: Pick K starting centroids (randomly or with k-means++).
2. **Assignment step**: For each $x_i$, assign it to the nearest centroid:
   $$C_k = \{x_i : \|x_i - \mu_k\|^2 \le \|x_i - \mu_j\|^2, \; \forall j\}$$
3. **Update step**: Recompute each centroid as the mean of its cluster:
   $$\mu_k = \frac{1}{|C_k|} \sum_{x_i \in C_k} x_i$$
4. **Repeat** until assignments stop changing.

This process monotonically decreases the objective and converges to a local minimum.

---

## 3. Distance Computation — Loops vs Broadcasting

We often need to compute distances between all data points and all centroids.

### Loop Implementation
```python
def pairwise_sq_dists_loops(X, centers):
    n, f = X.shape
    k = centers.shape[0]
    D = np.zeros((n, k))
    for i in range(n):
        for j in range(k):
            D[i, j] = np.sum((X[i] - centers[j])**2)
    return D
```
This is easy to read but slow — O(n × k × f) explicit loops.

### Broadcasting Trick (Vectorized NumPy)
```python
D = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
```
This uses **NumPy broadcasting**: expands arrays so that subtraction is done pairwise without loops.

- `X[:, None, :]`: shape (n, 1, f)
- `centers[None, :, :]`: shape (1, k, f)
- Subtraction → shape (n, k, f)
- Squared and summed → (n, k)

---

### 3.1 Broadcasting Trick — Squared Distance Matrix (No Python Loops)

We often need the full matrix of **squared Euclidean distances** between all data points `X` (shape `(n, f)`) and all centroids `centers` (shape `(k, f)`).

**Goal:** build `D` of shape `(n, k)` where  
`D[i, j] = || X[i, :] - centers[j, :] ||^2` — *without* Python loops.

---

#### Core Idea (NumPy Broadcasting)

We can give NumPy compatible shapes so it **automatically expands** dimensions:

- `X[:, None, :]` has shape `(n, 1, f)`  
- `centers[None, :, :]` has shape `(1, k, f)`

Subtracting them gives a tensor of shape `(n, k, f)` with **all pairwise differences**.  
Then square and sum over the **feature axis** (`axis=2`) to get the distance matrix `(n, k)`.

```python
import numpy as np

# X: (n, f), centers: (k, f)
# Pairwise squared Euclidean distances: D: (n, k)
D = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
```

Shapes:
- `X[:, None, :]` → `(n, 1, f)`
- `centers[None, :, :]` → `(1, k, f)`
- subtraction → `(n, k, f)`
- `** 2` → `(n, k, f)`
- `.sum(axis=2)` → `(n, k)`

---

#### Concrete Example (Small Numbers)

Let’s compute distances from 3 points to 2 centroids in 2D:

```python
import numpy as np

X = np.array([
    [1., 2.],   # point 0
    [3., 4.],   # point 1
    [5., 6.]    # point 2
])  # shape (3, 2) -> n=3, f=2

centers = np.array([
    [0., 0.],   # centroid A
    [1., 1.]    # centroid B
])  # shape (2, 2) -> k=2, f=2

D = ((X[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
print(D)
# Expected:
# [
#   [(1-0)^2 + (2-0)^2,  (1-1)^2 + (2-1)^2],   # point 0 to A, B
#   [(3-0)^2 + (4-0)^2,  (3-1)^2 + (4-1)^2],   # point 1 to A, B
#   [(5-0)^2 + (6-0)^2,  (5-1)^2 + (6-1)^2]    # point 2 to A, B
# ]
# → [[ 5.,  1.],
#    [25.,  8.],
#    [61., 20.]]
```

Explanation for row 0 (`[1,2]`):
- to `[0,0]`: `(1-0)^2 + (2-0)^2 = 1 + 4 = 5`
- to `[1,1]`: `(1-1)^2 + (2-1)^2 = 0 + 1 = 1`

---

#### Why This Is Fast

- **No Python loops**: all heavy lifting is done in optimized C/BLAS with vectorized ops.
- **Single pass**: compute all pairwise differences in one shot.

---

#### Memory-Savvy Alternative (Same Result)

For very large `n` and `k`, `(n, k, f)` can be big.  
Use the identity `||x - c||^2 = ||x||^2 + ||c||^2 - 2 x·c` to avoid building the `(n, k, f)` tensor:

```python
# X: (n, f), centers: (k, f)
X_sq = (X**2).sum(axis=1, keepdims=True)          # (n, 1)
C_sq = (centers**2).sum(axis=1, keepdims=True).T  # (1, k)
XC   = X @ centers.T                               # (n, k)
D_id = X_sq + C_sq - 2 * XC                        # (n, k)
```

`D_id` equals `D` (up to minor floating-point differences), but typically uses **less memory**.

---

#### Drop-In for K-Means

Use either `D` or `D_id` when assigning labels:

```python
labels = np.argmin(D, axis=1)     # broadcasting version
# or
labels = np.argmin(D_id, axis=1)  # identity version
```

Both give the nearest centroid index for each data point, which is exactly what we need in the **assignment step** of Lloyd’s algorithm.

---

## 4. K-Means in Code (From Scratch)

```python
def kmeans_from_scratch(X, K, n_iters=100):
    n, f = X.shape
    # Random initialization
    idx = np.random.choice(n, K, replace=False)
    centers = X[idx]

    for _ in range(n_iters):
        # Assign each point to the nearest centroid
        D = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)
        labels = np.argmin(D, axis=1)

        # Recompute centroids
        new_centers = np.array([X[labels == k].mean(axis=0) for k in range(K)])

        # Stop if converged
        if np.allclose(centers, new_centers):
            break
        centers = new_centers

    return centers, labels
```

- Initialization picks random points as centers.
- Assignment step uses vectorized distance computation.
- Update step recomputes the mean of assigned points.

**Convergence:** Each iteration decreases the WCSS objective.

---

## 5. K-Means with scikit-learn

```python
from sklearn.cluster import KMeans

km = KMeans(n_clusters=3, n_init=10, max_iter=300, tol=1e-4, random_state=42)
km.fit(X)
labels = km.labels_
centers = km.cluster_centers_
```

`n_init=10` means it runs the algorithm with 10 random initializations and picks the best result by WCSS.

In practice, scikit-learn’s implementation of K-Means adds two useful **stopping criteria** that control convergence speed and precision:
- **`max_iter`** — the maximum number of Lloyd iterations (assignment + update steps).
- **`tol`** — a small tolerance value that determines when to stop early if centroids stop moving significantly.

**Visualization Tip:** Plot the clusters and centers to see how k-means organizes the data.

---

## 6. Real Example: Wine Dataset

We can use the **UCI Wine dataset** with two chemical features (so we can visualize easily).

Steps:
1. Load dataset → scale features.
2. Apply K-Means with `K=3`.
3. Visualize clusters vs true wine types.

Observe: Clusters may roughly correspond to varieties, but K-Means doesn’t know labels — it only groups by feature similarity.

---

## 7. K-Means++ Initialization

A key weakness of the original K-Means algorithm is **sensitivity to initialization**.  
If the starting centroids are poorly chosen, the algorithm may converge to a **local minimum** or produce unbalanced clusters.

The **K-Means++** initialization method addresses this by spreading out the initial centroids in a statistically sound way.

---

### Algorithm Steps

1. **Choose the first centroid** randomly from the data points.

2. **For each remaining centroid**:
   - Compute the distance \( D(x) \) from each data point \( x \) to the nearest already chosen centroid.
   - Select the next centroid with probability proportional to \( D(x)^2 \).

   This ensures that:
   - Points far from existing centroids have a higher chance of being chosen.
   - Initial centroids are well separated.

3. **Proceed with Lloyd’s algorithm** (assignment + update steps).

---

### Mathematical Intuition

- The probability of picking a new centroid is proportional to the **squared distance** from existing centroids:
  \[
  P(x_i) = \frac{D(x_i)^2}{\sum_j D(x_j)^2}.
  \]
- This spreads centroids across regions of the dataset where variance is large.

Theoretical result:
- K-Means++ initialization guarantees that the expected WCSS (within-cluster sum of squares) is within \( O(\log K) \) of the global optimum before any iteration begins.

---

### Example Implementation

```python
import numpy as np

def kmeans_plusplus_init(X, K, random_state=None):
    rng = np.random.default_rng(random_state)
    n = X.shape[0]

    # Step 1: Choose first centroid uniformly at random
    centroids = [X[rng.choice(n)]]

    # Step 2: Select remaining centroids
    for _ in range(1, K):
        # Compute squared distances to nearest existing centroid
        D_sq = np.min(((X[:, None, :] - np.array(centroids)[None, :, :])**2).sum(axis=2), axis=1)
        # Choose new centroid with probability proportional to distance squared
        probs = D_sq / D_sq.sum()
        new_idx = rng.choice(n, p=probs)
        centroids.append(X[new_idx])

    return np.array(centroids)
```

This function can be used before Lloyd’s iterations to produce a much better starting point than random sampling.

---

### Practical Benefits

- Reduces the number of iterations to convergence.
- Produces more stable, reproducible clusters.
- Minimizes the likelihood of empty or degenerate clusters.

In scikit-learn, this is the default:
```python
KMeans(init='k-means++', n_clusters=K, random_state=42)
```

K-Means++ initialization is thus a **simple yet powerful improvement** that makes clustering faster and more reliable in practice.

---

## 8. Nearest-Centroid Classification

Once we have centroids, we can classify a **new sample** by its nearest centroid.

```python
x_new = np.array([[13.0, 5.0]])
D = ((x_new[:, None, :] - centers[None, :, :])**2).sum(axis=2)
nearest = np.argmin(D, axis=1)
```

This is called **Nearest Centroid Classification** — it assigns points to whichever cluster mean they’re closest to.

- Simple, fast, and interpretable.
- Works well when classes are compact and spherical.
- Fails when classes have irregular boundaries.

---

## 9. k-Nearest Neighbors (k-NN)

### Idea
For a query point $x$, find its $k$ closest **training samples**, not centroids, and let them **vote** on the label.

- **Instance-based learning**: keeps all training points.
- Decision boundaries are **nonlinear** and adapt to data.

### Algorithm
1. Compute distance from $x$ to all training points.
2. Take the $k$ smallest distances.
3. Predict the majority label among those $k$ neighbors.

```python
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
```

**Pros:**
- Simple and effective when data are well-scaled.
- No explicit training.

**Cons:**
- Slow for large datasets (requires all distances).
- Sensitive to feature scaling and irrelevant features.

---

## 10. Comparing K-Means, Nearest Centroid, and k-NN

| Aspect | K-Means | Nearest Centroid | k-NN |
|---------|----------|------------------|------|
| Type | Unsupervised | Supervised (prototype-based) | Supervised (instance-based) |
| Stores | Only centroids | Centroids (from labeled data) | All training points |
| Decision boundary | Linear (Voronoi cells) | Linear | Nonlinear |
| Speed (prediction) | Fast | Very fast | Slower (O(n)) |
| Handles irregular shapes | No | No | Yes |
| Needs labels | No | Yes | Yes |

---

## 11. Choosing k and Validation

### K in K-Means
- Use **Elbow Method**: plot WCSS vs K; find the elbow point.
- Use **Silhouette Score**: average separation vs cohesion.

### k in k-NN
- Small $k$: sensitive to noise, overfits.
- Large $k$: smoother decision boundary, underfits.

Use **cross-validation** to select optimal $k$.

---

## 12. Summary & Exercises

### Summary
- K-Means minimizes within-cluster distances via Lloyd’s alternating updates.
- NumPy broadcasting enables efficient pairwise distance computation.
- scikit-learn provides robust, optimized implementations.
- Nearest-Centroid and k-NN are distance-based classifiers — both rely on proximity, but differ in whether they store prototypes or all instances.

### Exercises
1. Implement K-Means with your own convergence criterion.
2. Compare K-Means clusters with true labels on Iris or Wine datasets.
3. Try Nearest Centroid vs k-NN and visualize decision boundaries.
4. Experiment with feature scaling and distance metrics (e.g., Manhattan).
5. Implement weighted k-NN (weights by inverse distance).

---
