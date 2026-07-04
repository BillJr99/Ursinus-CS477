# Decision Trees: Entropy, Information Gain, Pruning, and Random Forests
<!--
author:   William M. Mongan
language: en
narrator: US English Male
comment:  Render via https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-trees.md
import:   https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md
link:     https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-5
           https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Decision Trees: Entropy, Information Gain, Pruning, and Random Forests

Our first full machine learning model! A **decision tree** learns by asking the *most informative questions first* — like a well-played game of Twenty Questions. Trees are accurate, fast, and (rare among ML models) **readable**: you can print the learned model and argue with it.

---

## 1. Intuition First: Twenty Questions with Data

You are guessing an animal. Do you open with "Is it a platypus?" Of course not — you ask "**Is it bigger than a breadbox?**" because the answer, *whatever it is*, cuts the possibilities roughly in half.

A decision tree learner plays the same game against a dataset:

1. Consider every feature-question it could ask ("humidity high?", "income > \$50k?").
2. Measure how much each question would **reduce uncertainty** about the label.
3. Ask the best question; split the data on the answer.
4. Recurse on each branch until the answers are (nearly) certain.

The measure of "uncertainty" is **entropy**, and the "reduction" is **information gain**.

---

## 2. Entropy: Measuring Impurity

For a node whose examples have class proportions $p_1, \dots, p_k$:

$$
H = -\sum_{i=1}^{k} p_i \log_2 p_i \qquad \text{(bits)}
$$

Interpretation with concrete numbers (two classes):

| Mix at node | $H$ | Reading |
|---|---|---|
| 8 yes / 0 no | $0$ bits | pure — nothing left to learn |
| 6 yes / 2 no | $0.811$ bits | mostly predictable |
| 4 yes / 4 no | $1$ bit | maximally impure — a coin flip |

Micro-check of the middle row, step by step:

1. Proportions: $p_{\text{yes}} = 6/8 = 0.75$, $p_{\text{no}} = 2/8 = 0.25$.
2. Plug in: $H = -(0.75 \log_2 0.75 + 0.25 \log_2 0.25)$.
3. Evaluate the logs: $\log_2 0.75 \approx -0.415$, $\log_2 0.25 = -2$.
4. $H = -(0.75 \times -0.415 + 0.25 \times -2) = -(-0.311 - 0.5) = 0.811$ bits. ✓

---

## 3. One Split, Fully By Hand

**Dataset:** 14 weekend days; label = *Play tennis?* (9 Yes, 5 No). Candidate question: split on **Humidity** (High / Normal).

| Humidity | Yes | No | Total |
|---|---|---|---|
| High | 3 | 4 | 7 |
| Normal | 6 | 1 | 7 |
| **All** | **9** | **5** | **14** |

**Step 1 — Entropy of the parent node.**

1. $p_{\text{yes}} = 9/14 \approx 0.643$, $\;p_{\text{no}} = 5/14 \approx 0.357$.
2. $H(\text{parent}) = -(0.643 \log_2 0.643 + 0.357 \log_2 0.357)$.
3. $\log_2 0.643 \approx -0.637$ and $\log_2 0.357 \approx -1.485$.
4. $H(\text{parent}) = -(0.643 \times -0.637 + 0.357 \times -1.485) = 0.410 + 0.530 = 0.940$ bits.

**Step 2 — Entropy of each child.**

- **High** (3 Yes, 4 No): $H = -(\tfrac{3}{7}\log_2\tfrac{3}{7} + \tfrac{4}{7}\log_2\tfrac{4}{7}) = -(0.429 \times -1.222 + 0.571 \times -0.807) = 0.524 + 0.461 = 0.985$ bits. *(Nearly a coin flip — bad.)*
- **Normal** (6 Yes, 1 No): $H = -(\tfrac{6}{7}\log_2\tfrac{6}{7} + \tfrac{1}{7}\log_2\tfrac{1}{7}) = -(0.857 \times -0.222 + 0.143 \times -2.807) = 0.190 + 0.401 = 0.592$ bits. *(Much purer.)*

**Step 3 — Weighted average of the children.** Each child holds $7/14 = 0.5$ of the data:

$$
H(\text{after split}) = 0.5 \times 0.985 + 0.5 \times 0.592 = 0.789 \text{ bits.}
$$

**Step 4 — Information gain.**

$$
IG(\text{Humidity}) = H(\text{parent}) - H(\text{after}) = 0.940 - 0.789 = \mathbf{0.151} \text{ bits.}
$$

The learner computes this for *every* candidate feature and greedily picks the largest. (On the classic full tennis dataset, *Outlook* wins with $IG \approx 0.247$, so it becomes the root; Humidity is asked next, but only inside the "Sunny" branch — the tree tailors its second question to the answer of the first.)

---

## 4. Gini Impurity: The Practical Cousin

CART-style trees (including scikit-learn's default) replace entropy with **Gini impurity**:

$$
G = 1 - \sum_i p_i^2
$$

- Same 6-yes/2-no node: $G = 1 - (0.75^2 + 0.25^2) = 1 - 0.625 = 0.375$.
- Interpretation: the probability of misclassifying a random example if you labeled it by drawing a label from the node's mix.
- Entropy and Gini almost always choose the same splits; Gini skips the logarithms, so it is faster. **Continuous features** are handled by trying threshold questions like $x_j \le t$ at candidate cut points between sorted values.

---

## 5. Overfitting and Pruning: When Trees Memorize

Grown to purity, a tree happily creates a leaf *per training example* — perfect training accuracy, terrible generalization (recall the data/features module!). A pure-grown tree is a **high-variance** model: tiny changes in the training data yield very different trees.

Remedies:

1. **Pre-pruning (early stopping):** cap `max_depth`, require `min_samples_leaf`, or demand a minimum information gain before splitting.
2. **Post-pruning:** grow the full tree, then collapse subtrees whose removal does not hurt performance on a held-out **validation set** (or use cost-complexity pruning: penalize a tree by $\alpha \times$ number of leaves).
3. Judge either strategy the honest way — on data the tree never saw.

---

## 6. Random Forests: Wisdom of a Decorrelated Crowd

If one deep tree overfits, average many *different* deep trees and the individual quirks cancel. **Random forests** manufacture the required diversity twice over:

1. **Bagging:** each tree trains on a bootstrap sample (draw $n$ examples *with replacement*).
2. **Feature subsampling:** at every split, only a random subset of features (typically $\sqrt{d}$) may be asked about — so the trees cannot all lean on the same dominant feature.
3. **Predict** by majority vote (classification) or averaging (regression).

Intuition: averaging $T$ nearly-independent estimators divides variance by roughly $T$. The forest loses the single tree's readability, but **feature importances** (how much impurity each feature removes, forest-wide) recover a useful summary — a preview of the Explainable AI module's concerns.

---

## 7. Runnable Code: Trees vs. Forests

```python
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

X, y = make_moons(n_samples=400, noise=0.3, random_state=477)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.5, random_state=477)

for depth in [1, 3, None]:
    t = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
    print(f"depth={str(depth):>4}: train={t.score(Xtr, ytr):.2f}  test={t.score(Xte, yte):.2f}")

rf = RandomForestClassifier(n_estimators=200, random_state=0).fit(Xtr, ytr)
print(f"forest    : train={rf.score(Xtr, ytr):.2f}  test={rf.score(Xte, yte):.2f}")

print(export_text(DecisionTreeClassifier(max_depth=2, random_state=0).fit(Xtr, ytr),
                  feature_names=["x1", "x2"]))
```

**What to look for:**

- `depth=None` (fully grown): train accuracy 1.00 but the *worst* test gap — overfitting made visible in two numbers.
- `depth=3` usually beats both extremes — the pruning story in miniature.
- The forest matches or beats the best single tree without tuning depth.
- `export_text` prints the learned model as nested if/then questions — try reading it aloud.

---

## 8. Guided Demo: Trees Meet Their Linear Rivals

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/linear_logistic_regression_demo.ipynb)

This demo notebook builds **linear and logistic regression step by step** on simple 2-D data. Run it now, *before* week 7, wearing your tree goggles:

- **Early cells (data + line fitting):** a linear model draws one straight boundary through feature space. Ask: where would a depth-1 tree ("decision stump") put its single axis-aligned split on the same data?
- **Logistic regression cells:** the model outputs a *probability* that varies smoothly across space; a tree outputs a *constant* probability per leaf (the leaf's class mix) — a staircase instead of a ramp.
- **Boundary plots:** linear models are confident even far from the data along the boundary's direction; trees can carve non-linear, axis-aligned regions but never a diagonal line (they approximate it with steps).
- **Takeaway table to hold onto:**

| | Linear/logistic | Decision tree |
|---|---|---|
| Boundary shape | one straight line/plane | axis-aligned staircase |
| Feature scaling needed? | yes (for gradient descent) | no |
| Interpretable? | weights | printed rules |
| Extrapolates smoothly? | yes (maybe too boldly) | no — constant outside data range |

There is no dedicated tree notebook in the course collection (yet) — the runnable cells in Section 7 plus this comparison are your lab for today, and *Machine Learning for Engineers* Chapter 4 (Ensemble Learning, linked in the schedule) continues the forest story.

---

## Comprehension Quiz

[[MC]]
A node holds 4 positive and 4 negative examples. Its entropy is:
- ( ) 0 bits.
- ( ) 0.5 bits.
- (x) 1 bit.
- ( ) 2 bits.

[[MC]]
In the worked split, Humidity earned $IG = 0.151$ bits while Outlook earns $\approx 0.247$ bits on the same data. The learner should:
- (x) split on Outlook first — greedy selection takes the largest information gain.
- ( ) split on Humidity first because its children are balanced in size.
- ( ) split on both simultaneously.
- ( ) always prefer binary features.

[[MC]]
Why do random forests randomly restrict the features available at each split?
- ( ) To make each tree train faster, nothing more.
- (x) To decorrelate the trees so their errors cancel when averaged.
- ( ) To guarantee each tree is unbiased.
- ( ) To prevent any feature from ever dominating predictions.

---

## Discussion Prompt

> Decision trees are prized in medicine and lending precisely because they can be read and audited. A random forest is more accurate but opaque. If you were the regulator, when would you insist on the tree — and what evidence would let you accept the forest?
