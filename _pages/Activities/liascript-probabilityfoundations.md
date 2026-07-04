# Probability Foundations for AI: Uncertainty, Conditioning, and Bayes Intuition
<!--
author:   William M. Mongan
language: en
narrator: US English Male
comment:  Render via https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-probabilityfoundations.md
import:   https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md
link:     https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-5
           https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Probability Foundations for AI: Uncertainty, Conditioning, and Bayes Intuition

Logic breaks the moment the world gets noisy: sensors lie, patients present ambiguous symptoms, emails are *probably* spam. This module builds the probability toolkit — axioms, conditional probability, independence, expectation, and an intuition-first preview of **Bayes' rule** — that powers everything from Naïve Bayes to Kalman filters later in the course.

---

## Open Colab: Diagnostic Testing & Belief Updating

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Bayesian_Diagnostic_Test_Belief_Updating.ipynb)

---

## Open Colab: Maximum Likelihood from First Principles

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/MLE_From_First_Principles_Worked_Examples.ipynb)

---

## 1. Intuition First: Probability as Calibrated Belief

Say "there is a 30% chance of rain." Nothing about *today* happens 30 times. The number is a **degree of belief**: over many days you announce "30%," it should rain on about 3 in 10 of them. An AI agent keeps exactly such numbers about everything it cannot observe directly — and **updates them as evidence arrives**. Probability theory is simply the bookkeeping that keeps those beliefs self-consistent.

---

## 2. The Three Axioms (and Everything They Buy Us)

For events $A, B$ in a sample space $\Omega$:

1. $0 \le P(A) \le 1$ — beliefs live between impossible and certain.
2. $P(\Omega) = 1$ — *something* happens.
3. If $A$ and $B$ are mutually exclusive, $P(A \cup B) = P(A) + P(B)$ — probabilities of non-overlapping events add.

Useful consequences, each one line from the axioms:

- **Complement:** $P(\lnot A) = 1 - P(A)$.
- **Inclusion–exclusion:** $P(A \cup B) = P(A) + P(B) - P(A \cap B)$ *(don't double-count the overlap)*.
- **Marginalization / total probability:** if $\{B_i\}$ partition $\Omega$,
  $$P(A) = \sum_i P(A \cap B_i) = \sum_i P(A \mid B_i)\,P(B_i).$$
  Plain language: to get the probability of $A$, average its probability over every way the world could be. This one identity is the engine of every "sum over hidden states" we do this semester (HMMs, localization, mixture models).

---

## 3. Conditional Probability: The Zoom Lens

$$
P(A \mid B) = \frac{P(A \cap B)}{P(B)} \qquad (P(B) > 0)
$$

**Read it as a zoom:** once we learn $B$ happened, the relevant universe *shrinks* to $B$, and we re-measure $A$ inside it.

**Worked micro-example (two dice).** What is $P(\text{sum} = 8 \mid \text{first die} = 3)$?

1. Unconditioned, $P(\text{sum}=8) = 5/36$. *(Five ways: 2+6, 3+5, 4+4, 5+3, 6+2.)*
2. Learn the first die shows 3 → the universe shrinks to 6 equally likely worlds: (3,1)...(3,6).
3. Among them, only (3,5) sums to 8 → $P = 1/6$. *(Evidence changed the answer from $5/36 \approx 0.14$ to $\approx 0.17$.)*

Rearranging the definition gives the **product rule**, $P(A \cap B) = P(A \mid B) P(B)$, and chaining it gives the **chain rule** — the factorization trick behind Bayesian networks and language models alike:

$$
P(A, B, C) = P(A)\,P(B \mid A)\,P(C \mid A, B).
$$

---

## 4. Independence: The License to Multiply

$A$ and $B$ are **independent** iff $P(A \cap B) = P(A)P(B)$, equivalently $P(A \mid B) = P(A)$ — learning $B$ tells you nothing about $A$.

- Dice are independent; symptoms of one disease are **not** (they share a cause).
- **Conditional independence** — $P(A, B \mid C) = P(A \mid C) P(B \mid C)$ — is AI's favorite compromise: fever and cough are correlated, but *given* that you know the patient has the flu, they become (approximately) independent. This single assumption turns an exponential joint table into a product of small ones, and is the entire secret of **Naïve Bayes** (week 11).

---

## 5. Random Variables and Expectation

A **random variable** $X$ assigns a number to each outcome; its **distribution** lists $P(X = x)$.

$$
\mathbb{E}[X] = \sum_x x \, P(X = x), \qquad \mathrm{Var}(X) = \mathbb{E}\big[(X - \mathbb{E}[X])^2\big]
$$

**Micro-example — should the agent buy the ticket?** A raffle ticket costs \$2; it wins \$50 with probability $0.03$.

1. $\mathbb{E}[\text{winnings}] = 50 \times 0.03 + 0 \times 0.97 = 1.50$. *(Average payout per ticket over many plays.)*
2. Net value $= 1.50 - 2.00 = -0.50 < 0$ → a rational agent declines. *(Acting to maximize expected utility — the formal core of "rational agent."*)

Every learning algorithm we meet minimizes an **expected loss**; every RL agent maximizes an **expected return**. Expectation is the quiet workhorse of the whole course.

---

## 6. Bayes' Rule, Intuition First

Flip the conditioning direction of the product rule and you get the most important equation in this course:

$$
\underbrace{P(H \mid E)}_{\text{posterior}} \;=\; \frac{\overbrace{P(E \mid H)}^{\text{likelihood}} \; \overbrace{P(H)}^{\text{prior}}}{\underbrace{P(E)}_{\text{evidence}}}
$$

- **Prior** $P(H)$: belief before the evidence.
- **Likelihood** $P(E \mid H)$: how well the hypothesis *explains* the evidence.
- **Posterior** $P(H \mid E)$: belief after — prior reweighted by explanatory power.

**The famous worked example — rare disease, good test.** Disease prevalence 1%; test sensitivity $P(+\mid D) = 0.9$; false-positive rate $P(+\mid \lnot D) = 0.05$. You test positive. What is $P(D \mid +)$?

1. Imagine **1,000 people**. About $10$ have the disease; $990$ do not. *(Prior as head-counts.)*
2. Of the 10 sick: $0.9 \times 10 = 9$ test positive. *(True positives.)*
3. Of the 990 healthy: $0.05 \times 990 = 49.5$ test positive. *(False positives — the crowd is huge, so even 5% of it is a lot.)*
4. Total positives $= 9 + 49.5 = 58.5$; of them, sick $= 9$.
5. $P(D \mid +) = 9 / 58.5 \approx 0.15$.

A positive result from a "90% accurate" test leaves only a **15%** chance of disease — because the prior was so small. Ignoring the prior is the **base-rate fallacy**, and it has real consequences in medicine, security screening, and any ML classifier applied to rare events.

```python
prior = 0.01          # P(D)
sens  = 0.90          # P(+|D)
fpr   = 0.05          # P(+|~D)

evidence  = sens * prior + fpr * (1 - prior)      # P(+), total probability
posterior = sens * prior / evidence               # Bayes' rule
print(f"P(+)      = {evidence:.4f}")
print(f"P(D | +)  = {posterior:.4f}")             # ~0.1538

# Evidence compounds: a SECOND independent positive test?
prior2 = posterior
evidence2  = sens * prior2 + fpr * (1 - prior2)
print(f"P(D | ++) = {sens * prior2 / evidence2:.4f}")   # ~0.7657
```

**What to look for:** yesterday's posterior becomes today's prior — beliefs update *sequentially*. That loop, run once per sensor reading, is exactly the Bayes filter we will build for robot localization and the pursuit game in the **Bayesian Inference** module.

---

## 7. Where This Toolkit Goes Next

| This week's idea | Where it reappears |
|---|---|
| Total probability / marginalization | HMM forward algorithm, localization heatmaps |
| Conditional independence | Naïve Bayes, Bayesian networks |
| Bayes' rule | Bayesian inference module; the enemy AI in `bayesian_chase.py` |
| Expectation | Loss minimization (regression), expected return (RL) |
| Likelihood | Maximum-likelihood estimation → logistic regression, GMMs |

---

## Comprehension Quiz

[[MC]]
A test is 90% sensitive with a 5% false-positive rate, and the condition affects 1% of the population. Why is $P(\text{disease} \mid +)$ only about 15%?
- ( ) The test's sensitivity is too low for inference.
- (x) The healthy population is so large that its 5% false positives outnumber the true positives.
- ( ) The events are not independent.
- ( ) The calculation violates the axioms.

[[MC]]
$P(A \mid B) = P(A)$ tells us that...
- ( ) $A$ and $B$ are mutually exclusive.
- ( ) $B$ causes $A$.
- (x) $A$ and $B$ are independent: observing $B$ does not change belief in $A$.
- ( ) $P(B) = 1$.

---

## Discussion Prompt

> Spam filters, fraud detectors, and TSA screening all hunt for rare events. Using the base-rate reasoning above, explain why "99% accurate" systems can still generate mostly false alarms — and who bears the cost of those false alarms.
