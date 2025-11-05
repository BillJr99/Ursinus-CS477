---
layout: assignment
permalink: Assignments/NeuralNet
title: "Assignment: Implementing a Neural Network"

info:
  points: 100
  goals:
    - Build intuition for forward pass, loss, gradients, and backpropagation by implementing networks from scratch.
    - Progress from a fully-worked linear estimator to partially scaffolded implementations (multilayer, nonlinear, credit score weights, MNIST).
    - Explain each coding step in your own words and validate results with small tests/plots.
  purpose: "This lab is designed to build practical intuition for how neural networks work by implementing each piece from first principles. You will (i) replicate the full learning loop—forward pass, loss, gradients, and parameter updates, (ii) practice chaining layers and applying the chain rule via backpropagation, and (iii) extend from linear models to nonlinear MLPs that use ReLU or tanh. Along the way, you will validate your implementations with small tests/plots, interpret model parameters on a toy credit-scoring dataset, and train a small classifier on MNIST using softmax, cross-entropy, and minibatching. The purpose is to explain each step in your own words and connect the code you write to the underlying mathematical ideas."    
  concepts:
    - Forward pass for linear and multilayer models
    - Mean Squared Error (MSE) and residuals
    - Parameter gradients and update rules (SGD)
    - Backpropagation and the chain rule
    - Composition of linear layers (and why a composition of linear layers is still linear)
    - Activation functions (ReLU, tanh) and their derivatives
    - Xavier/Glorot initialization intuition
    - Gradient checking by finite differences (sanity checks)
    - "Model architecture: Linear layer abstraction (W, b; dW, db)"
    - Training loop structure (loss, backward, parameter update)
    - Softmax and cross-entropy for classification
    - Minibatching and basic learning-curve plotting
    - Interpreting learned weights on a toy credit-score dataset
    - Small-scale MNIST experiment flow
  tasks:
    - Run the fully-worked linear estimator and interpret its behavior.
    - Implement the Linear layer's forward and backward methods.
    - Build and train a two-layer linear network; verify loss decreases.
    - Implement ReLU (and optionally tanh) and their backward passes.
    - Construct a one-hidden-layer MLP using your modules; train and explain.
    - Perform gradient-sanity checks (e.g., tiny batch finite-difference check).
    - Interpret weights on the credit-score toy dataset (regularization, effects).
    - Train a tiny MNIST model with softmax and cross-entropy using minibatches.
    - Document design choices, checks, and results with brief plots/tables.    
  rubric:
    - weight: 30
      description: Implementation
      preemerging: Provides a working implementation aligned to the assignment specification with simple tests.
      beginning: Implements core functionality and demonstrates usage on representative inputs.
      progressing: Implements full specification with clear structure, tests, and discussion of edge cases.
      proficient: Delivers a robust, well-structured implementation with comprehensive tests and justified design choices.
    - weight: 30
      description: Algorithmic Correctness and Reasoning
      preemerging: Explains the algorithmic approach and verifies outputs on basic cases.
      beginning: Explains design decisions and validates outputs on typical cases with reasoning.
      progressing: Provides correctness arguments and empirical checks across varied cases.
      proficient: Presents clear correctness reasoning and evidence of generalization with insightful error analysis.
    - weight: 20
      description: Code Quality and Documentation
      preemerging: Organizes code into readable units with brief inline comments.
      beginning: Uses functions/modules and docstrings to clarify behavior and interfaces.
      progressing: Maintains consistent style, meaningful names, and explanatory docs where non-trivial.
      proficient: Exhibits clean architecture, thoughtful abstractions, and thorough documentation throughout.
    - weight: 10
      description: Design Report
      preemerging: Summarizes goals, approach, and evaluation setup.
      beginning: Explains design decisions and trade-offs with small-scale results.
      progressing: Details design rationale, experiments, and limitations with supporting figures/tables.
      proficient: Delivers a concise, well-structured report with justified choices and actionable future work.
    - weight: 10
      description: Submission Completeness
      preemerging: Provides required artifacts and basic run instructions.
      beginning: Includes all artifacts with clear run instructions and parameters.
      progressing: Includes scripts, configs, and reproducible steps with sample data.
      proficient: Provides a fully reproducible package with results, seeds, and validation notes.

tags:
  - ai
---

# Overview

In this lab, we will walk through building neural networks from scratch. You will:

1. **Run** a fully-worked linear estimator (no missing code).
2. **Fill in** carefully isolated pieces for a multilayer linear network.
3. **Extend** to a nonlinear MLP by implementing activations and their derivatives.
4. **Interpret** weights on a **Credit Score** toy dataset (regularization + feature effects).
5. **Train** a tiny NN on **MNIST**, implementing softmax, cross-entropy, and minibatching.

Each major section explains *why* and *what to change*, with **TODO** comments throughout.  

You can implement your code in Python using an editor/interpreter of your choice, or use [Google Colab](https://colab.research.google.com/) to create a Jupyter notebook.

---

## Stage 0 — Setup (Shared Utilities)

Before each stage, you may want simple helpers for reproducibility and plotting.

```python
import math, random
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
random.seed(42)

def plot_series(xs, ys, title=""):
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("step/epoch")
    plt.ylabel("value")
    if title:
        plt.title(title)
    plt.show()
    
# Check that two matrices are approximately equal in value, with some tolerance for rounding error    
def assert_close(name, got, want, atol=1e-7, rtol=1e-5):
    ok = np.allclose(got, want, atol=atol, rtol=rtol)
    print(f"[{name}] match:", ok, f"(max abs err ~ {np.max(np.abs(got - want)) if got.shape == want.shape else 'shape mismatch'})")
    if not ok:
        print("   shapes:", got.shape, "vs", want.shape)
```

---

## Stage 1 — **Linear Function Estimator (complete; just run it)**

**Goal.** Fit a line <span>\\(\hat{y} = wx + b\\)</span> to noisy data. You will **not** implement anything—just run, read, and interpret it.

**Concepts:** forward pass, mean squared error (MSE), gradients, update rule.

```python
# 1) Generate data from y = a*x + b + noise
n = 200
true_a, true_b = 2.5, -0.7
x = np.linspace(-2, 2, n)
noise = 0.25 * np.random.randn(n)
y = true_a * x + true_b + noise

# 2) Initialize parameters
w = np.random.randn()
b = np.random.randn()

# 3) Training loop (gradient descent)
lr = 0.05
losses = []
for step in range(800):
    yhat = w * x + b                    # forward
    err = yhat - y                      # residuals
    loss = np.mean(err**2)              # MSE
    # gradients dMSE/dw, dMSE/db
    dw = 2 * np.mean(err * x)
    db = 2 * np.mean(err)
    # update
    w -= lr * dw
    b -= lr * db
    losses.append(loss)

print({"w": w, "b": b, "true_a": true_a, "true_b": true_b})
plot_series(range(len(losses)), losses, title="MSE over steps")
```

**Checkpoints (answer briefly in your report):**
- Why does the MSE decrease? What would happen with a much larger learning rate?
- If `x` were all near zero, which parameter (w or b) would be easier to learn? Why?

---

## Stage 2 — **Linear, Multi-Layer (fill in targeted pieces)**

We now stack **linear layers** without nonlinearities. A depth-2 linear network is still linear end-to-end—but this is great practice for shapes and chaining.

**You will implement:**
- A reusable `Linear` layer’s forward and backward.
- A simple training loop that uses those layers.

### 2.1 Define a Linear Layer

Read the skeleton, then implement the TODOs.

```python
class Linear:
    def __init__(self, in_dim, out_dim):
        # Xavier/Glorot for linear-only is fine
        limit = math.sqrt(6/(in_dim+out_dim))
        self.W = np.random.uniform(-limit, limit, size=(out_dim, in_dim))
        self.b = np.zeros((out_dim,))
        # grads
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        # cache
        self.x = None

    def forward(self, x):
        """x: (batch, in_dim) -> out: (batch, out_dim)"""
        self.x = x
        
        # TODO: return x @ self.W.T + self.b 
        out = None  # TODO
        
        return out

    def backward(self, dout):
        """dout: (batch, out_dim) -> returns dx: (batch, in_dim)
        Fills self.dW, self.db."""
        
        # In case a simple list [...] is passed here, make it a matrix [[...]], 
        # since in general it could be a matrix of arbitrary dimension
        dout = np.atleast_2d(np.asarray(dout))   # ensure the shape of dout is (B, out_dim)
        x = np.atleast_2d(np.asarray(self.x)) # ensure the shape of x is (B, in_dim)
        
        # TODO: compute grads wrt W, b, and return dx
        dx = None  # TODO
        
        return dx
```

#### `forward` Function

**Why:**  
- `x @ self.W.T` multiplies each input row by the columns of `W` (so the output has one column per output unit).  The `@` operator performs matrix multiplication in Python.  If your version of Python does not support the `@` operator, you can substitute it with `np.matmul` like this: `np.matmul(x, self.W.T)`.
- `+ self.b` adds the bias to every row.

#### `backward` Function

**What to do:** (inside `def backward(self, dout):`):  
This computes gradients for weights and bias, and the gradient to pass downstream.

1. Set `self.dW` to `dout.T` matrix multiplied by `self.x`
2. Set `self.db` to `dout.sum(axis=0)` to add these elements together
3. Set `dx` to `dout` matrix multiplied by `self.W` and return `dx`

**Why:**  
- `self.dW` asks: "if I nudge a weight, how much does the loss change?" For matrices, the rule gives `dout.T @ X`.  
- `self.db` is the sum across the batch because `b` is added to every row.  
- `dx` tells the previous layer how the loss changes if inputs change: multiply by `W`.

### 2.2 Compose Two Linear Layers

We will create a two-layer linear model: `x -> Linear1 -> Linear2 -> yhat`. No activation yet.  Just run this code.

```python
lin1 = Linear(in_dim=1, out_dim=4)
lin2 = Linear(in_dim=4, out_dim=1)

lr = 0.05
losses = []

# Reuse data from Stage 1
# Generate data from y = a*x + b + noise
n = 200
true_a, true_b = 2.5, -0.7
x = np.linspace(-2, 2, n)
noise = 0.25 * np.random.randn(n)
y = true_a * x + true_b + noise

# Format x and y into matrices to be used by the layer objects
X = x.reshape(-1, 1)  # (n,1)
Y = y.reshape(-1, 1)  # (n,1)

for epoch in range(200):
    # forward
    h = lin1.forward(X)
    yhat = lin2.forward(h)
    # loss
    err = yhat - Y
    loss = np.mean(err**2)
    losses.append(loss)
    # backward (dMSE/dyhat = 2*err/bs)
    bs = X.shape[0]
    dyhat = (2.0/bs) * err
    dh = lin2.backward(dyhat)
    _ = lin1.backward(dh)
    # SGD update
    for layer in (lin1, lin2):
        layer.W -= lr * layer.dW
        layer.b -= lr * layer.db

plot_series(range(len(losses)), losses, title="Two-layer linear: MSE")
```

**Why:**

Here, we **chain two `Linear` layers together** to see how forward and backward passes compose.

1. **Define two layers**  
   - `lin1 = Linear(in_dim=1, out_dim=4)` expands a single input feature into a 4-dimensional hidden representation.  
   - `lin2 = Linear(in_dim=4, out_dim=1)` collapses that hidden vector back to a single prediction.

2. **Forward pass**  
   - First call `h = lin1.forward(X)` to compute hidden features.  
   - Then `yhat = lin2.forward(h)` to produce predictions.  
   This illustrates how modules are chained together to form deeper networks.

3. **Loss calculation**  
   - Compute residuals `err = yhat - Y` and the mean squared error (MSE).  
   - This is the same loss as Stage 1, but now passed through two layers.

4. **Backward pass**  
   - Start with gradient of MSE wrt predictions: `dyhat = (2.0/bs) * err`.  
   - Push it through `lin2.backward(dyhat)` and then `lin1.backward(dh)`.  
   This shows how the **chain rule** is implemented in code: each layer returns gradients for the one before it.

5. **Parameter update**  
   - After gradients are computed, each layer’s weights and biases are updated with SGD:  
     `layer.W -= lr * layer.dW`, `layer.b -= lr * layer.db`.

6. **Summary**  
   Even with two layers, the composition is **still just a single linear function overall** (because no nonlinearity is added yet). This stage is practice for chaining layers and verifying backprop mechanics before introducing nonlinear activations in Stage 3.

**Checkpoints:**
- If loss steadily decreases, your `Linear` layer is working.
- Verify `lin1.backward` and `lin2.backward` by finite differences on a tiny batch (e.g., 3 points). Briefly describe the result.
- Why does stacking linear layers still represent a linear map overall?

---

## Stage 3 — **Nonlinear MLP (implement activations + their grads)**

Introduce <span>\\(\sigma\\)</span> to gain expressivity. We will add **ReLU** by default and let you try **tanh**.

### 3.1 Activation Modules

```python
def relu(x):
    # TODO: return elementwise max(0, x) using np.maximum()
    return None

def relu_backward(x, dout):   
    dx = np.empty_like(x) # make dx an array shaped like the input x so we can loop over it
    
    # TODO: 
    # if an element of x (a 2D matrix) is > 0, set the corresponding element of dx to the corresponding element of dout
    # otherwise, set the corresponding element of dx to 0
    
    return dx

def tanh(x):
    return np.tanh(x)

def tanh_backward(x, dout):
    # TODO: compute dout * d/dx tanh = dout * (1 - tanh(x)^2)
    return None
```

**What to do:**

1. In the `relu` function, use the `np.maximum(0, x)` function to which returns `max(a, b)` for every element of `x`.
2. In the `relu_backward` function, we want to pass through the values corresponding to `z` elements that were not trimmed by the ReLU function.  In other words, those values of `dout` that correspond to positive values of `z`.  For each element of `x` passed to the function, append an element to `dx` (initially an empty array): that value should be `0` if the current element of `x` is less than `0`, and should be the current element of `dout` (the gradient value for that element) otherwise.  This is our ReLU gradient, and you should return that `dx` array.  Note that this can be done in two lines of code rather than with a loop (although a nested loop will work!), using `(x > 0).astype(dout.dtype)`, which returns a matrix of `1` in all places where `x > 0` and `0` in all other locations.  That's not quite the right answer, but you could use that to set all the elements of `dx` that are equal to `1` to their corresponding element of `dout` via a multiplication operation.
3. Fill in the formula for `tanh_backward` and return that result from the function.

**Mini-check:**

```python
z = np.array([-2., 0., 3.])
print("relu(z):", relu(z))              # expect [0., 0., 3.]
dout = np.array([10., 10., 10.])
print("relu_backward:", relu_backward(z, dout))  # expect [0., 0., 10.]
```

**Key Takeaways:**

1. Notice how the chain rule looks in code: multiply the upstream gradient (dout) by the derivative of the activation (σ′(z)), implemented here as a simple mask.
2. Numpy has functions to automatically create bitmasks for arrays.  In other words, you can figure out which values are positive in just one line of code, and then use that mask to pull out the values of `dout` with one more line of code.  See if you can find this!

### 3.2 Multi-Layer Perceptron (MLP) with One Hidden Layer

This code is complete as it uses the functions we defined previously, and you can just run it.

```python
class MLP:
    def __init__(self, in_dim, hidden_dim, out_dim, nonlin="relu"):
        self.l1 = Linear(in_dim, hidden_dim)
        self.l2 = Linear(hidden_dim, out_dim)
        self.nonlin = nonlin
        self.h_pre = None  # pre-activation cache

    def forward(self, x):
        z1 = self.l1.forward(x)
        self.h_pre = z1
        if self.nonlin == "relu":
            h = relu(z1)            
        else:
            h = tanh(z1)           
        yhat = self.l2.forward(h)
        return yhat, h

    def backward(self, dyhat, h):
        dh = self.l2.backward(dyhat)
        if self.nonlin == "relu":
            dz1 = relu_backward(self.h_pre, dh) 
        else:
            dz1 = tanh_backward(self.h_pre, dh) 
        dx = self.l1.backward(dz1)
        return dx
```

### Forward pass (`MLP.forward`)

**Why:**

```python
z1 = self.l1.forward(x)
```
- Pass input `x` through the first linear layer.
- Computes z1 = X W1^T + b1.
- Shape: (B, in_dim) @ (in_dim, hidden_dim) = (B, hidden_dim).

```python
self.h_pre = z1
```
- Cache the raw pre-activation values for use in backprop.
- Needed because ReLU/tanh derivatives depend on the *input* to the nonlinearity.

```python
if self.nonlin == "relu":
    h = relu(z1)
else:
    h = tanh(z1)
```
- Apply the chosen nonlinearity elementwise.
- ReLU: pass positives, zero-out negatives.
- Tanh: squashes values smoothly into [-1, 1].

```python
yhat = self.l2.forward(h)
```
- Pass hidden activations through the second linear layer.
- Computes predictions yhat = H W2^T + b2.

```python
return yhat, h
```
- Return both the predictions and hidden activations.
- h is useful for visualization or debugging.

### Backward pass (`MLP.backward`)

**Why:**

```python
dh = self.l2.backward(dyhat)
```
- Backprop through the second linear layer.
- dyhat is ∂L/∂ŷ from the loss.
- Returns ∂L/∂h.

```python
if self.nonlin == "relu":
    dz1 = relu_backward(self.h_pre, dh)
else:
    dz1 = tanh_backward(self.h_pre, dh)
```
- Backprop through the nonlinearity.
- Uses the cached z1 (self.h_pre).
- Implements chain rule: ∂L/∂z1 = ∂L/∂h ⊙ σ′(z1).

```python
dx = self.l1.backward(dz1)
```
- Backprop through the first linear layer.
- Produces ∂L/∂x, useful if this MLP sits inside a larger model.

```python
return dx
```
- Return ∂L/∂x so the gradient can keep flowing backward.

### 3.3 Train on a **nonlinear** target

This code is complete as it uses the functions we defined previously, and you can just run it.

```python
# Nonlinear data: y = sin(2*pi*x) + noise
n = 400
x2 = np.linspace(-1, 1, n)
y2 = np.sin(2*math.pi*x2) + 0.1*np.random.randn(n)
X2 = x2.reshape(-1,1)
Y2 = y2.reshape(-1,1)

model = MLP(in_dim=1, hidden_dim=32, out_dim=1, nonlin="relu")
lr = 0.005 # a smaller learning rate here
losses = []

for epoch in range(1500):
    yhat, h = model.forward(X2)
    err = yhat - Y2
    loss = np.mean(err**2)
    losses.append(loss)
    # backward
    dyhat = (2.0/X2.shape[0]) * err
    _ = model.backward(dyhat, h)
    # update
    for layer in (model.l1, model.l2):
        layer.W -= lr * layer.dW
        layer.b -= lr * layer.db

plot_series(range(len(losses)), losses, title="Nonlinear MLP: MSE")
```

**Questions:**
- Swap `nonlin="tanh"`. Which fits better here and why? Reference bias/variance.
- Try increasing the number of epochs from `1500` to `15000`.  What happens, and why?
- Plot predictions versus ground truth to visually confirm fit.

---

## Stage 4 — **Credit Score Feature Weight Estimator (interpretability)**

We will fit a **linear classifier/regressor** and discuss **feature weights** and **regularization**.

**You will implement:**
- L2 regularization in the loss and gradients.
- A simple interpretability pass over learned weights.

### 4.1 Dataset & Model Skeleton

Begin by defining sone input features about the people (credit utilization percentage, age, number of late payments), and a parallel array of their credit scores.  We will try to estimate the function that maps these features to credit scores for new people (for whom their actual credit score is not yet known).  We normalize the features so that they are all on the same scale by subtracting the mean of each feature from each element of that feature, and dividing by the standard deviation of that overall feature.

```python
# Example synthetic credit-like features 
# Features: [utilization, age, late_payments]
Xc = np.array([
    [0.10, 45, 0],
    [0.90, 22, 3],
    [0.40, 31, 1],
    [0.20, 52, 0],
    [0.70, 28, 2],
], dtype=float)

# Target score (higher is better)
yc = np.array([720, 580, 660, 740, 610], dtype=float).reshape(-1,1)

# Normalize features for stability
mu, sigma = Xc.mean(axis=0, keepdims=True), Xc.std(axis=0, keepdims=True) + 1e-8
Xn = (Xc - mu)/sigma

lin = Linear(in_dim=Xn.shape[1], out_dim=1)
```

### 4.2 Loss with L2 Regularization (implement)

```python
lam = 1e-2  # try 0, 1e-2, 1e-1
lr = 0.01
losses = []
for epoch in range(1000):
    yhat = lin.forward(Xn)
    err = yhat - yc

    # TODO: Compute the total loss, and append to the losses array
    losses.append(loss)

    # backward
    bs = Xn.shape[0]
    dy = (2.0/bs) * err
    _ = lin.backward(dy)
    
    # TODO: Keep a running total of lin.dW (add to it) with the gradient from L2: d/dW lam||W||^2 = 2*lam*W
    
    # update
    lin.W -= lr * lin.dW
    lin.b -= lr * lin.db

plot_series(range(len(losses)), losses, title="Credit score with L2")

print("weights (on normalized features):\n", lin.W)
print("bias:", lin.b)
```


**What to do:**

1. Inside your training loop, compute the MSE loss (as a scalar from the error vector) so that we can plot it.  The error vector is `yhat - yc`.  Compute this, square it (you can square an entire vector just like you would square a single scalar value), and compute the mean with `np.mean`.  This is the Mean Squared Error (MSE).  We square the values so that they are always positive, so that direction doesn't offset the error artificially.
2. Next, compute the L2 penalty.  This is the sum of the squares of all elements of `lin.W` (use `np.sum` to compute the sum).  Multiply that sum by the regularization strength multiplier `lam`.  This is referred to as lambda, and is a hyperparameter we use to tune training.  A lambda value of `0` disables regularlization and just uses MSE to calculate loss, while larger values of lambda penalize large weights and incentivize smaller weight values.  
3. The total loss is the MSE loss plus this L2 regularlization penalty.  Add these two terms together.  Call this `loss` so that it appends to the `losses` array in the template above.  By adding this to the loss, we consider a result with large weights as more lossy than a result with smaller weights, with the hope that this will allow our model to better generalize to new data.
4. Later, add the gradient of the L2 regularization penalty to `lin.dW`, right before the template code adds `lin.dW` to `lin.W`.  It is computed with the formula given in the `TODO` comment, and is the derivative of `lam*(lin.W**2)`, which is `2*lam*lin.W`.  This applies the L2 regularization penalty to the weight adjustments.

**Why:**

- L2 regularization (also called *weight decay*) discourages large weights by adding a penalty proportional to the squared magnitude of all weights.  
- Mathematically, if  
  L_total = L_data + λ ||W||²  
  then  
  ∂L_total/∂W = ∂L_data/∂W + 2λW
- That’s why you add `2 * lam * W` to the gradient.  
- The effect is to shrink weights toward zero:
  * Smaller weights reduce variance and help generalization.  
  * Larger `lam` increases the penalty (more bias, less variance).  
- Note: only the weights (W) are regularized, not the biases (b).

**Mini-check:** Compare gradients with and without regularization on the same batch.

```python
np.random.seed(3)
lin = Linear(in_dim=3, out_dim=1)
Xn = np.random.randn(5,3)
yc = np.random.randn(5,1)

# forward
yhat = lin.forward(Xn)
err = yhat - yc
bs = Xn.shape[0]
dy = (2.0/bs) * err

# grads without reg
_ = lin.backward(dy)
dW_no_reg = lin.dW.copy()

# grads with reg
lam = 0.01
_ = lin.backward(dy)            # recompute base grads
lin.dW += 2*lam*lin.W
dW_with_reg = lin.dW

assert_close("L2 grad delta", dW_with_reg - dW_no_reg, 2*lam*lin.W)
```

**Questions:**
1. Which features carry the largest magnitude weights? What does the **sign** imply?  
2. How does increasing `lam` affect weights and fit? Connect to **bias–variance** and overfitting risk.
3. By hand, use the weights and biases to calculate the credit score of someone with `[0.30, 21, 0]`.  How does this compare to a person with `[0.30, 65, 0]`, and what does this mean?

---

## Stage 5 — **MNIST: Tiny Neural Net (classification)**

Implement a minimal **two-layer** classifier with **softmax + cross-entropy**, minibatching, and accuracy. We provide most code; you fill in the key formulas.

### 5.1 Data Loading (you may replace with your notebook’s loader)

This code is complete.  It imports the MNIST dataset, and creates a function `one_hot` which creates an array of all `0` values except for the one target element whose value is `1`.

```python
from sklearn.datasets import load_digits
digits = load_digits()
X, y = digits.data, digits.target

# train/val split (small subset to speed up)
N = 20000
X, y = X[:N], y[:N]
num_classes = 10

# one-hot helper
def one_hot(y, C):
    oh = np.zeros((y.shape[0], C), dtype=np.float32)
    oh[np.arange(y.shape[0]), y] = 1.0
    return oh
Y = one_hot(y, num_classes)
```

### 5.2 Softmax + Cross-Entropy (fill in)

When training neural networks for **classification tasks** such as MNIST digit recognition, a common combination is the **softmax activation** in the output layer with a **cross-entropy loss** function. 

The network outputs raw **logits** <span>\\(z\\)</span>, which can be any real numbers. To interpret them as probabilities over classes (that add up to `100%` probability, or `1.0`), we apply the softmax function: 

$$
\hat{y}_i = \frac{e^{z_i}}{\sum_{j=1}^C e^{z_j}}
$$

- Each output <span>\\(\hat{y}_i\\)</span> is in <span>\\([0,1]\\)</span>.  
- The probabilities sum to 1, making them interpretable as class likelihoods.

**Cross-Entropy Loss: Comparing Probabilities to Labels:**
For a one-hot label vector <span>\\(y\\)</span>, the cross-entropy loss is:

<span>
\\(
L = -\sum_{i=1}^C y_i \log(\hat{y}_i)
\\)
</span>

- If the correct class is <span>\\(k\\)</span>, this reduces to <span>\\(L = -\log(\hat{y}_k)\\)</span>.  
- Intuitively, this penalizes the model heavily if it assigns low probability to the correct class.

**Derivative of the Cross-Entropy Loss Function:**
The derivative of the cross-entropy loss with respect to the logits (after softmax) simplifies dramatically:

<span>
\\(\nabla_z L = \hat{y} - y\\)
</span>

This is the difference between the predicted probabilities and the true labels, without computing the derivatives of the softmax function.

```python
def softmax(logits):
    # logits: (batch, C)
    z = logits - logits.max(axis=1, keepdims=True)
    expz = np.exp(z)
    return expz / expz.sum(axis=1, keepdims=True)

def cross_entropy(probs, Y_true):
    # TODO: mean negative log-likelihood; use eps for stability
    eps = 1e-9
    return None

def softmax_cross_entropy_backward(probs, Y_true):
    # d/dlogits of CE with softmax fusion: (probs - Y)/batch
    bs = probs.shape[0]
    return (probs - Y_true) / bs
```

**What to do:** 

Implement the mean negative log-likelihood (use a tiny epsilon value `eps` to avoid accidentally computing `log(0)`).

1. Compute the log (use `np.log`) of the probability vector `probs`.  You can add `eps` to this to add the epsilon value to all elements in one line of code, just as if `probs` was a scalar value.  Similarly, you can pass that whole result to `np.log` which will calculate on all elements of the vector.
2. Calculate the log-likelihood of each element being the correct class by calculating the product of `Y_true` by the log probability vector you just computed.  Since the log of a probability should always be negative (since it is the log of a value between `0` and `1`), multiply this result by `-1` to make it a negative log likelihood.  Then, call `.sum(axis=1)` on that vector.  You will end up with a vector that is `0` in all positions except the one-hot position, since we multiplied it by the one-hot vector `Y_true` earlier.  The one-hot position will have a likelihood corresponding to that correct classification.  To spread this likelihood out over the remaining positions, return the mean of the negative log likelihood vector, which corresponds to a measure of how "surprised" the model is by this classification.  For example:

```
Y_true[i] = [0, 1, 0]
logp[i]   = [-2.0, -0.1, -3.0]
Y_true[i] * logp[i] = [0, -0.1, 0]
```

**Why:**  
- Cross-entropy for one-hot `Y_true` is the negative log-probability assigned to the true class, averaged over the batch.
- Probabilities are between 0 and 1, so their logarithms are ≤ 0.
- If we summed raw log-probabilities, we would be maximizing a negative number, which doesn’t fit the idea of a "loss".
- By taking the **negative log**, we turn the objective into a positive quantity that can be minimized with gradient descent.
- Mathematically:
  - **Likelihood**:  
    <span>\\(L(\theta) = \prod_i p_\theta(y_i \mid x_i)\\)</span>

  - **Log-likelihood** (to maximize):  
    <span>\\(\log L(\theta) = \sum_i \log p_\theta(y_i \mid x_i)\\)</span>

  - **Negative log-likelihood (NLL)** (to minimize):  
    <span>\\(\sum_i \log p_\theta(y_i \mid x_i)\\)</span>
- Intuition:  
  - If the model assigns high probability to the correct class (e.g. 0.99), log(prob) ≈ –0.01 → NLL ≈ 0.01 (good, small loss).  
  - If the model assigns low probability (e.g. 0.01), log(prob) ≈ –4.6 → NLL ≈ 4.6 (large penalty).
  
**Mini-check:** If the model predicts the true class with probability 1, the loss is ~0.

```python
probs = np.array([[0.0, 1.0, 0.0]], dtype=np.float32)  # 100% on class 1
Yt = np.array([[0,1,0]], dtype=np.float32)             # true class is 1
print("CE perfect:", cross_entropy(probs, Yt))         # ~ 0.0

probs = np.array([[0.5, 0.5, 0.0]], dtype=np.float32)  # 50/50 split between 0 & 1
Y0 = np.array([[1,0,0]], dtype=np.float32)             # true class is 0
print("CE 50/50:", cross_entropy(probs, Y0))           # ~ 0.693 (ln 2)
```

*(Backward is already given to you — `(probs - Y_true)/batch` — so no TODO here.)*

### 5.3 Two-Layer Classifier

This code is complete and runs our code above.

```python
D = X.shape[1]     # e.g. 784 for MNIST
H = 64             # you can try other sizes: 32,128 etc.
C = num_classes    # 10 usually for the number of digits being classified
EPOCHS = 500

model = MLP(in_dim=D, hidden_dim=H, out_dim=C, nonlin="relu")

lr = 0.01
batch = 128
losses_mlp = []
accs_mlp = []

for epoch in range(EPOCHS):
    perm = np.random.permutation(X.shape[0]) # shuffle the data
    X_shuf, Y_shuf = X[perm], Y[perm]
    for i in range(0, X.shape[0], batch):
        Xb = X_shuf[i:i+batch]
        Yb = Y_shuf[i:i+batch]

        # Forward through model
        logits, h = model.forward(Xb)   # logits shape: (batch, C)
        probs = softmax(logits)

        # Loss
        loss = cross_entropy(probs, Yb)
        losses_mlp.append(loss)

        # Backward
        dyhat = softmax_cross_entropy_backward(probs, Yb)
        _ = model.backward(dyhat, h)

        # Update parameters
        for layer in (model.l1, model.l2):
            layer.W -= lr * layer.dW
            layer.b -= lr * layer.db

    # Optional: evaluate accuracy on a validation / subset
    probs_all, _ = model.forward(X[:2000])
    pred = probs_all.argmax(axis=1)
    acc = (pred == y[:2000]).mean()
    accs_mlp.append(acc)
    print(f"(MLP) epoch {epoch} acc ~ {acc:.3f}")

# Plotting
plot_series(range(len(losses_mlp)), losses_mlp, title="MNIST MLP: loss over steps")
plot_series(range(len(accs_mlp)), accs_mlp, title="MNIST MLP: running accuracy (subset)")
```

**Why:**
- This section puts everything together — linear layers, ReLU activation, softmax, and cross-entropy — into a *trainable classifier*.
- The structure `X → fc1 → ReLU → fc2 → softmax` is the simplest neural net that can do non-linear classification.
- We use **minibatch SGD** to train efficiently on subsets of MNIST data.  
  - Each minibatch gives a noisy gradient estimate, but that noise helps escape poor local minima.  
- The **loss curve** (from `cross_entropy`) shows whether the model is learning; the **accuracy curve** checks generalization.
- Forward and backward steps reuse the building blocks from earlier stages (`Linear.forward`, `relu`, `softmax_cross_entropy_backward`), reinforcing the idea that complex models are built from simple, composable parts.
- By coding the loop manually, students see how deep-learning libraries like PyTorch or TensorFlow automate these mechanics.
- Intuition: after Stage 5.3, they’ve basically built a mini-PyTorch for MNIST — proving they understand the computational graph and gradient flow.

**Checkpoints:**
- Derive the fused gradient of softmax+CE to justify `softmax_cross_entropy_backward`.
- Try different `H` and learning rates. How do training curves change?

---

# Stage 6: Try it with TensorFlow

```python
import math
import random
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Reproducibility
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

def plot_series(xs, ys, title=""):
    plt.figure()
    plt.plot(xs, ys)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    if title:
        plt.title(title)
    plt.show()

def standardize_features(X):
    """
    Standardize columns to zero mean / unit variance.
    Returns X_std, mu, sigma
    """
    mu = X.mean(axis=0, keepdims=True)
    sigma = X.std(axis=0, keepdims=True) + 1e-8
    Xn = (X - mu) / sigma
    return Xn, mu, sigma

def unstandardize_linear_weights(W_std, b_std, mu, sigma):
    """
    Convert weights learned on standardized features back to original feature units.

    Model on standardized inputs:
        y = b_std + sum_j W_std[j] * (x_j - mu_j) / sigma_j

    Rearranged into original feature units:
        y = b_orig + sum_j W_orig[j] * x_j
    where:
        W_orig[j] = W_std[j] / sigma_j
        b_orig    = b_std - sum_j W_std[j] * mu_j / sigma_j
    """
    W_orig = W_std / sigma.ravel()
    b_orig = b_std - np.sum(W_std * (mu.ravel() / sigma.ravel()))
    return W_orig, b_orig

def main():
    # -----------------------------
    # 1) Toy "credit-like" dataset
    # Features: [utilization (0-1), age (years), late_payments (count)]
    # Target: FICO-like credit score (higher is better)
    # -----------------------------
    feature_names = ["utilization", "age", "late_payments"]
    X = np.array([
        [0.10, 45, 0],
        [0.90, 22, 3],
        [0.40, 31, 1],
        [0.20, 52, 0],
        [0.70, 28, 2],
    ], dtype=float)

    y = np.array([720, 580, 660, 740, 610], dtype=float).reshape(-1, 1)

    # Standardize inputs (the lab does this before linear regression)
    Xn, mu, sigma = standardize_features(X)

    # -----------------------------
    # 2) TensorFlow/Keras linear model with L2 regularization
    # -----------------------------
    lam = 1e-2  # try 0, 1e-2, 1e-1 to see the effect of regularization strength
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(Xn.shape[1],)),
        tf.keras.layers.Dense(
            units=1,
            activation=None,               # linear regression
            use_bias=True,
            kernel_regularizer=tf.keras.regularizers.L2(lam)  # adds λ * sum(W^2) to the loss
        )
    ])

    model.compile(
        optimizer=tf.keras.optimizers.SGD(learning_rate=0.1),
        loss="mse"  # base data loss; regularization term is added automatically to total loss
    )

    # Custom callback to capture total loss each epoch
    class LossHistory(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.losses = []
        def on_epoch_end(self, epoch, logs=None):
            # logs["loss"] includes data loss + regularization penalties
            self.losses.append(float(logs["loss"]))

    history_cb = LossHistory()

    # Train
    EPOCHS = 1000
    model.fit(Xn, y, epochs=EPOCHS, verbose=0, callbacks=[history_cb])

    # Plot training curve (total loss: data + L2)
    plot_series(range(1, EPOCHS + 1), history_cb.losses, title=f"Credit score with L2 (λ={lam})")

    # -----------------------------
    # 3) Inspect learned parameters
    # -----------------------------
    W_std_tf, b_std_tf = model.layers[0].get_weights()  # W shape (in_dim, 1), b shape (1,)
    W_std = W_std_tf[:, 0].copy()           # (3,)
    b_std = float(b_std_tf[0])

    print("\n=== Weights on STANDARDIZED features (interpret magnitudes; sign shows direction) ===")
    for name, w in zip(feature_names, W_std):
        print(f"  {name:>14s}: {w:+.6f}")
    print(f"  {'bias':>14s}: {b_std:+.6f}")

    # Convert to original units for practitioner-friendly interpretation
    W_orig, b_orig = unstandardize_linear_weights(W_std, b_std, mu, sigma)

    print("\n=== Weights in ORIGINAL feature units (per unit change in raw feature) ===")
    for name, w in zip(feature_names, W_orig):
        print(f"  {name:>14s}: {w:+.6f} points per unit of {name}")
    print(f"  {'bias':>14s}: {b_orig:+.6f} (score when all features are zero)")

    # Rank features by absolute weight on standardized scale (safest for comparison)
    order = np.argsort(-np.abs(W_std))
    print("\nFeature influence ranking (by |weight| on standardized features):")
    for k in order:
        direction = "↑ (positive)" if W_std[k] >= 0 else "↓ (negative)"
        print(f"  {feature_names[k]:>14s}: |w|={abs(W_std[k]):.6f}  direction={direction}")

    # -----------------------------
    # 4) Example predictions
    # -----------------------------
    def predict_raw(features_row):
        x_row = np.array(features_row, dtype=float).reshape(1, -1)
        x_row_std = (x_row - mu) / sigma
        y_hat = model.predict(x_row_std, verbose=0)[0, 0]
        return y_hat

    p1 = [0.30, 21, 0]  # utilization 30%, age 21, no late payments
    p2 = [0.30, 65, 0]  # same utilization, older age, no late payments

    y1 = predict_raw(p1)
    y2 = predict_raw(p2)

    print("\n=== Example predictions (higher ≈ better) ===")
    print(f"  Person A {p1}: predicted score ≈ {y1:.2f}")
    print(f"  Person B {p2}: predicted score ≈ {y2:.2f}")

    # If desired, also compute using original-unit weights analytically:
    def predict_with_unstandardized(W_orig, b_orig, x_row):
        return float(np.dot(W_orig, np.array(x_row)) + b_orig)

    y1_alt = predict_with_unstandardized(W_orig, b_orig, p1)
    y2_alt = predict_with_unstandardized(W_orig, b_orig, p2)
    print("\nCheck (closed-form using original-unit weights):")
    print(f"  Person A: {y1_alt:.2f} | Person B: {y2_alt:.2f}")

if __name__ == "__main__":
    main()
```	

---

# Stage 7: Train Your Own Neural Network

Choose or create your own dataset, and train a neural network either from scratch, using the scikit libraries, or using TensorFlow.  Write a report about how you chose your hyperparameters for your model, how many layers and neurons you used, how you split your dataset, how you evaluated your model, and your training and testing results.

---

# What to Submit

1) **Code** for all stages (you may continue using/augmenting the provided notebooks).  
2) **Short report** (Markdown or PDF) addressing the **Checkpoints** and reflecting on what each added component buys you (depth, nonlinearity, regularization, minibatching, softmax/CE).  
3) **Plots**: at minimum, training loss for each stage; for MNIST, include accuracy over epochs.

# Hints & Troubleshooting

- Start small: run on tiny slices and verify shapes.
- If loss is `nan` or diverges, reduce learning rate and check gradients.
- Use finite-difference checks on tiny networks to validate your backward code.
- Normalize inputs (zero-mean/unit-var) for stability when appropriate.
- Seed randomness for repeatability when comparing trials.

## Common Fixes When You’re Stuck

- **Shape mismatch?** Print shapes right before a failing line. For matrix multiplies: `(A @ B)` needs inner dimensions equal.  
- **Loss not decreasing?** Try a smaller learning rate; check ReLU backward mask uses the **pre-activation** (z1), not the post-activation (h).  
- **Getting `nan`?** Reduce learning rate; for CE, make sure you’re using `np.log(probs + 1e-9)` (never log(0)).  
- **Randomness?** Set `np.random.seed(42)` at the top for repeatable runs.
