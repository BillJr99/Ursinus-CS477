---
layout: assignment
permalink: Assignments/XOREstimator
title: "Assignment: XOR Formula Estimator (Step‑by‑Step Tutorial)"

info:
  points: 100
  goals:
    - Understand why XOR is not linearly separable and how a small neural net solves it.
    - Read weights/biases from an interactive model (TensorFlow Playground) to derive an explicit formula.
    - Fit a linear estimator to XOR, evaluate failure modes, and reason about feature engineering.
  purpose: "You will explore the limitations of applying linear estimators to non-linear functions, but instead of overcoming this by adding a nonlinear activation function to the neural net, you will add a nonlinear feature that enables one to classify the output linearly."    
  concepts:
    - "Neural networks work by applying linear perceptrons in series, with a nonlinear activation function."
    - "Nonlinear relationships cannot be modeled by linear perceptrons alone."
    - "One can create nonlinear features as inputs to a neural network that map linearly to the outputs for linear classifications.  This feature engineering enables one to have better control over the input features to a neural network, and to better understand the underlying model."
  tasks:  
    - "Run a neural network on the XOR truth table using TensorFlow playground to observe the model formula weights."
    - "Interpret the neural network weights as a closed-form equation relating the inputs to the outputs via the computed weights and biases."
    - "Attempt to train a linear classifier on the XOR dataset (which is not possible), and add a nonlinear activation function to improve classification."
    - "Create a higher-dimension feature and use that to classify the data using linear classifiers only."
  rubric:
    - weight: 30
      description: Implementation
      preemerging: Provides a working implementation aligned to the assignment specification with simple tests.
      beginning: Implements the core functionality accurately and demonstrates usage on representative inputs.
      progressing: Implements the full specification with clear structure, tests, and discussion of edge cases.
      proficient: Delivers a robust, well‑structured implementation with comprehensive tests and justified design choices.
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
      progressing: Maintains consistent style, meaningful names, and explanatory docs where non‑trivial.
      proficient: Exhibits clean architecture, thoughtful abstractions, and thorough documentation throughout.
    - weight: 10
      description: Design Report
      preemerging: Summarizes goals, approach, and evaluation setup.
      beginning: Explains design decisions and trade‑offs with small‑scale results.
      progressing: Details design rationale, experiments, and limitations with supporting figures/tables.
      proficient: Delivers a concise, well‑structured report with justified choices and actionable future work.
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

In this lab we will walk through training a neural network to learn the bitwise XOR function using three stages:

1. **Playground Walkthrough (no coding):** Train a tiny neural net on XOR in TensorFlow Playground and **read off** the weights.
2. **From Weights → Formula:** Use those weights to write an explicit formula for <span>\\(Z = \operatorname{XOR}(A_1, A_2)\\)</span> via hidden neurons.
3. **Linear Estimator (coding):** Fit a **linear** model to the XOR truth table; evaluate performance vs. the neural net, and discuss why checking **two accuracies** is useful.

---

## The XOR Function

The **exclusive OR (XOR)** is a classic logical function that outputs **true (1)** when exactly one of its two inputs is true, and **false (0)** otherwise. It is often denoted as: <span>\\(Z = A_1 \oplus A_2\\)</span>, where <span>\\(A_1, A_2 \in \(0,1\)\\)</span>.

XOR is fundamental in AI and machine learning because it is **not linearly separable**: no single straight line (or hyperplane) can perfectly classify XOR in the input space. This property makes it a canonical example for demonstrating the need for **nonlinear models** and **hidden layers** in neural networks.

### XOR Truth Table

| <span>\\(A_1\\)</span>   |   <span>\\(A_2\\)</span> |   <span>\\(Z = A_1 \oplus A_2\\)</span>  |
|---------|---------|-------------------------|
|    0    |    0    |            0            |
|    0    |    1    |            1            |
|    1    |    0    |            1            |
|    1    |    1    |            0            |

### Key Observations
- XOR outputs **1** when inputs differ, **0** when they are the same.
- A single linear classifier cannot solve XOR.
- Adding nonlinear transformations or hidden units allows a neural network to represent XOR correctly.

---

## Stage 1 — TensorFlow Playground: Solve XOR and Read the Weights 

Open this preset link to the XOR dataset:

<https://playground.tensorflow.org/#activation=tanh&batchSize=10&dataset=xor&regDataset=reg-plane&learningRate=0.03&regularizationRate=0&noise=0&networkShape=4,2&seed=0.40997&showTestData=false&discretize=false&percTrainData=50&x=true&y=true&xTimesY=false&xSquared=false&ySquared=false&cosX=false&sinX=false&cosY=false&sinY=false&collectStats=false&problem=classification&initZero=false&hideText=false>

**Instructions**

1. Click **Play** and let it train until training loss stabilizes (or accuracy <span>\\(> 0.95\\)</span>).  
2. **Pause** training. Click on each hidden neuron to view its **incoming weights** (from <span>\\(A_1\\)</span> and <span>\\(A_2\\)</span>) and its **bias**. Record these in your notes:  
   <span>\\(h_j = \tanh(w_{j1} A_1 + w_{j2} A_2 + b_j)\\)</span>.
3. Click the output neuron to view **outgoing weights** <span>\\(v_j\\)</span> from hidden neurons and output bias <span>\\(c\\)</span>. Record:  
   <span>\\(\hat{Z} = \sigma\big(\sum_j v_j h_j + c\big)\\)</span>.
4. Sketch the **decision regions** you see. Which hidden neurons carve out which quadrants? How does the output combine them to realize XOR?

**Checkpoint (report briefly):**
- Write the specific numeric formula you see (fill in your <span>\\(w_{j1}, w_{j2}, b_j, v_j, c\\)<\span>).
- Explain (in your own words) how two perpendicular hidden hyperplanes make XOR possible.

---

## Stage 2 — From Weights to an Explicit XOR Formula 

We now turn the Playground parameters into an explicit function that maps <span>\\((A_1, A_2) \in \(0,1\)^2\\)</span> to <span>\\(Z\in\(0,1\)\\)</span>.

**Given** your recorded parameters:

- Hidden units: <span>\\(h_j = \tanh(w_{j1} A_1 + w_{j2} A_2 + b_j)\\)</span>.
- Output: <span>\\(\hat{Z} = \sigma\Big(\sum_j v_j h_j + c\Big)\\)</span>.

**Tasks**

1. Substitute <span>\\(A_1, A_2\in\(0,1\)\\)</span> and compute <span>\\(h_j\\)</span> for the **four truth‑table inputs**:  
   <span>\\((0,0), (0,1), (1,0), (1,1)\\)</span>.
2. Evaluate <span>\\(\hat{Z}\\)</span> for all four inputs and threshold at 0.5 to obtain predicted bits. Confirm XOR is realized.
3. (Optional) Use the **sign approximation** for <span>\\(\tanh\\)</span> to reason geometrically: the hidden neurons implement two half‑planes that, when linearly combined, produce XOR.

**Deliverable:** A concise **formula** filled with your numeric weights, plus a **table** of the four evaluations.

---

## Stage 3 — Linear Function Estimator on XOR 

In this part you will train a **linear model** <span>\\(\hat{Z} = w_1 A_1 + w_2 A_2 + b\\)</span> on the XOR truth table and evaluate it. 

### 3.1 Data: XOR Truth Table

```python
import numpy as np

# Truth table (A1, A2) -> Z = XOR(A1, A2)
X_tt = np.array([
    [0., 0.],
    [0., 1.],
    [1., 0.],
    [1., 1.],
], dtype=np.float64)
Z_tt = np.array([0., 1., 1., 0.], dtype=np.float64)
```

### 3.2 Linear Estimator (MSE fit)

```python
# Initialize parameters
rng = np.random.default_rng(0)
w = rng.normal(size=(2,))
b = rng.normal()

# Training hyperparameters
lr = 0.1
steps = 2000

for t in range(steps):
    # forward
    yhat = X_tt @ w + b                   # (4,)
    err = yhat - Z_tt
    loss = np.mean(err**2)

    # backward (d/dw MSE): 2/N X^T (yhat - y)
    grad_w = 2.0/len(X_tt) * (X_tt.T @ err)
    grad_b = 2.0/len(X_tt) * err.sum()

    # update
    w -= lr * grad_w
    b -= lr * grad_b

print({"w": w, "b": b, "loss": loss})
print(f"Linear formula: Z_hat = {w[0]:.3f}*A1 + {w[1]:.3f}*A2 + {b:.3f}")
```

**Checkpoint:** Does the **linear** model achieve zero error on the four truth‑table points? Why/why not? (Recall: XOR is **not** linearly separable).  How can you modify this code to perform better?

#### Revising the Linear classifier

Modify this code to use two layers, and to use a ReLU activation function.  What happens?

### 3.3 Evaluate Two Accuracies 

1. **Generalization accuracy** on *new random test points* sampled from <span>\\(\(0,1\)^2\\)</span>.
2. **Self‑consistency accuracy**: Feed the **original truth‑table inputs** through your **trained Playground neural net** *without labels* and compare its predictions with the ground truth.

```python
# 1) Accuracy on random test points
num_tests = 200
X_test = np.stack([rng.integers(0, 2, size=num_tests), rng.integers(0, 2, size=num_tests)], axis=1).astype(float)
Z_true = (X_test[:,0] != X_test[:,1]).astype(float)
Z_lin = (X_test @ w + b >= 0.5).astype(float)
acc_lin_random = (Z_lin == Z_true).mean()
print("Linear accuracy on random XOR test points:", acc_lin_random)

# 2) Accuracy of the trained neural net on the original truth table
# (You will manually enter the neural net’s formula from Stage 2 into a function.)

def neural net_playground_predict(A1, A2):
    # TODO: Replace the placeholders with your recorded weights from Stage 1.
    # Example with H hidden units, tanh in hidden, sigmoid at output.
    # h_j = tanh(wj1*A1 + wj2*A2 + bj); Z_hat = sigmoid(sum_j vj*h_j + c)
    # Fill: wj1, wj2, bj, vj, c using your Playground values.
    raise NotImplementedError

# Evaluate on the four truth-table points
from math import isclose
sig_neural net = []
for a1, a2, z in zip(X_tt[:,0], X_tt[:,1], Z_tt):
    # TODO: zhat = neural net_playground_predict(a1, a2)
    # TODO: sig_neural net.append( (zhat >= 0.5) == (z == 1.0) )
    pass

# TODO: acc_neural net_truth = np.mean(sig_neural net)
# print("neural net accuracy on original truth-table inputs:", acc_neural net_truth)
```

**Questions to answer in your report**
- Why is it useful to compute **both** accuracies? What might it mean if the linear model’s random‑test accuracy is high but the neural net’s truth‑table accuracy is low (or vice versa)? Consider overfitting, calibration thresholds (0.5), and numerical quirks.
- If your neural net and linear accuracies differ, which model’s behavior do you trust more for XOR, and why?

---

## Stage 4 — Make Linear Work via Feature Engineering

The XOR function is **not linearly separable** in the original input space defined by the two binary features <span>\\(A_1\\)</span> and <span>\\(A_2\\)</span>. No straight line in the <span>\\((A_1,A_2)\\)</span> plane can cleanly divide the points labeled “1” (the off-diagonal) from those labeled “0” (the diagonal). This is why a simple linear model fails, even if it is trained optimally.

### Adding Interaction Features

By introducing the **interaction feature** <span>\\(A_1 \cdot A_2\\)</span>, we are effectively **lifting the data into a higher-dimensional space**: <span>\\(X_{\text{aug}} = \big(A_1,\; A_2,\; A_1\cdot A_2\big)\\)</span>.

In this augmented feature space, a linear model is no longer constrained to operate only on the raw inputs. The product term captures the logical “AND” relationship between the two inputs, which is precisely the nonlinearity needed to represent XOR.  

- When both <span>\\(A_1=1\\)</span> and <span>\\(A_2=1\\)</span>, the interaction term is 1; otherwise, it is 0.  
- This allows the linear model in the augmented space to distinguish the <span>\\((1,1)\\)</span> case from the <span>\\((1,0)\\)</span> and <span>\\((0,1)\\)</span> cases, something impossible in the original two-dimensional input space.

In other words, the **linear separator in augmented space corresponds to a nonlinear separator in the original space**. This is a classical example of the **kernel trick** intuition: you can solve a non-linear problem by embedding it into a feature space where it becomes linearly separable.  Another approach is to use a non-linear activation function like the `relu`, as we have done before.

**What to do:** Add the interaction feature <span>\\(A_1\cdot A_2\\)</span> (or a nonlinear basis) so that a linear model in the **augmented** space can fit XOR.

```python
# Concatenate (np.c_) a column to the end of the X_tt truth table
# This will be a new data feature: the product of X_tt[0] and X_tt[1]
X_aug = np.c_[X_tt, (X_tt[:,0]*X_tt[:,1])]

# Re‑fit linear model on augmented features
w = rng.normal(size=(3,))
b = rng.normal()
for t in range(2000):
    yhat = X_aug @ w + b
    err = yhat - Z_tt
    grad_w = 2.0/len(X_aug) * (X_aug.T @ err)
    grad_b = 2.0/len(X_aug) * err.sum()
    w -= 0.1 * grad_w
    b -= 0.1 * grad_b

pred = (X_aug @ w + b >= 0.5).astype(float)
print("Augmented‑linear accuracy on XOR truth table:", (pred == Z_tt).mean())
print(f"Augmented formula: Z_hat = {w[0]:.3f}*A1 + {w[1]:.3f}*A2 + {w[2]:.3f}*(A1*A2) + {b:.3f}")
```

**Question:** how might this new feature influence your ability to classify the XOR truth table using a linear function?  What might that formula be?  It will be of the form `b + w1 * x1 + w2 * x2 + w3 * x3` where `x3 = x1 * x2`.  What might `b` and the `wi` weight values be?  Note that you can allow rounding to the nearest integer, so if you obtain a value of `0.5` or above, you can round that to `1`, and round smaller values to `0`.

### Representation vs. Learning

- **Representation (feature engineering):** By hand, we engineered the new feature <span>\\(A_1 \cdot A_2\\)</span>. This makes XOR linearly solvable, but only because we already *knew* what feature to add. In early machine learning practice, much effort went into designing clever features for each domain (e.g., polynomial expansions, Fourier bases, image descriptors).

- **Learning (neural networks):** A neural network learns its own internal “features.” In Stage 1, the hidden neurons applied nonlinear activations (tanh) to affine combinations of <span>\\(A_1\\)</span> and <span>\\(A_2\\)</span>. These hidden activations **implicitly created new feature spaces**, such as “is <span>\\(A_1\\)</span> positive but <span>\\(A_2\\)</span> negative?” or “are both inputs large?”—which are nonlinear partitions of the input space. By combining them, the network effectively invented an internal representation equivalent to the engineered <span>\\(A_1 \cdot A_2\\)</span> feature, but without us needing to guess it.

### Reflection
- What does this say about **representation** vs. **learning**? How does the neural net implicitly learn useful features that make XOR linearly solvable at the output layer, versus how we provide input features to the neural net?

---

# What to Submit

1. **Notes** with your recorded Playground weights, explicit formula, and truth‑table evaluation.  
2. **Code** for Stage 3 and Stage 4.  
3. A **short report** (1–2 pages) answering the Checkpoint questions and discussing accuracy differences and their meaning.
