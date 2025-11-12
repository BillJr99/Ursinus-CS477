# Heuristic Search: Hill Climbing and Minimax
<!--
author:   William M. Mongan
language: en
narrator: US English Male

comment: Render with https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-heuristicsearch.md or locally if deployed via https://www.billmongan.com/LiaScript/?https://raw.githubusercontent.com/BillJr99/Ursinus-CS477/gh-pages/_pages/Activities/liascript-heuristicsearch.md

import: https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md

link:   https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-4
        https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Heuristic Search: Hill Climbing and Minimax

This module develops two foundational heuristic search strategies:

1. **Hill Climbing** for single-agent optimization under limited information (greedy local improvement).
2. **Minimax** for adversarial search in deterministic, perfect-information, zero-sum games with optional **alpha–beta pruning**.

We will move from **intuition → formal models → algorithms → complexity → pitfalls → practice** and connect each concept to small executable code cells and proofs or proof sketches.

---

## Open Colab: Hill Climbing Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HillClimbing_Pacman_From_Scratch.ipynb)

---

## Open Colab: Minimax Tutorial

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Minimax_Tutorial.ipynb)

---

---

## Open Colab: Connect-4 Minimax with Alpha–Beta (From Scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Connect4_Minimax_AlphaBeta_From_Scratch.ipynb)


## 0. Environment & Utilities

We will use plain Python for pseudocode-like demonstrations. No internet access is required. The Colab notebooks above provide richer, runnable examples (including a Pacman gridworld for hill climbing and a simple game tree for minimax).

---

## Code Cell
```python
import math
import random
from typing import Callable, Iterable, Tuple, Optional, List

random.seed(42)
print("Utilities ready.")
```

---

# Part I — Hill Climbing

## 1. Problem Setting & Heuristic Landscape

We assume a search space $ \mathcal{S} $ of states with a **neighbor** relation $ N: \mathcal{S} \to 2^{\mathcal{S}} $ and an **objective/score function** $ f: \mathcal{S} \to \mathbb{R} $ to be **maximized** (for minimization, use $ -f $).

**Goal.** Find a state $ s^\* $ such that
$$
s^\* \in \arg\max_{s \in \mathcal{S}} f(s).
$$

**Locality assumption.** We only evaluate neighbors $ N(s) $ of the current state, not the global space.

**Landscape hazards.**
- **Local maxima:** states whose neighbors are not better, but which are not globally optimal.
- **Plateaus:** flat regions where many neighbors share the same value.
- **Ridges:** ascent requires a sequence of lateral moves.

---

## 2. Greedy Ascent (Standard Hill Climbing)

At each step, move to the neighbor with the **highest** value if it strictly improves the objective.

### Pseudocode

```text
function HILL-CLIMB(s, f, N):
    repeat:
        s_best := argmax_{s' in N(s)} f(s')
        if f(s_best) <= f(s): return s
        s := s_best
```

**Termination.** Stops at a local maximum or a flat plateau where no strict improvement exists.

**Time per step.** $ O(\lvert N(s)\rvert) $ evaluations.

---

## Code Cell — Generic Hill Climber

```python
def hill_climb(s0, f: Callable, neighbors: Callable) -> Tuple:
    s = s0
    while True:
        nbrs = list(neighbors(s))
        if not nbrs:
            return s
        s_best = max(nbrs, key=f)
        if f(s_best) <= f(s):
            return s
        s = s_best

# Toy landscape: integers with a concave parabola and small noise
def f_int(x: int) -> float:
    return -(x-7)**2 + 50 + (0.05 if x % 5 == 0 else 0.0)

def N_int(x: int) -> Iterable[int]:
    yield x - 1
    yield x + 1

sol = hill_climb(0, f_int, N_int)
print("Local optimum:", sol, "value:", f_int(sol))
```

---

## 3. Variants & Remedies

1. **Steepest-Ascent vs. First-Choice.** Evaluate **all** neighbors and pick the best vs. sample neighbors randomly and move to the **first** improvement. First-choice reduces evaluation overhead when $ \lvert N(s)\rvert $ is large.
2. **Stochastic Hill Climbing.** Select a better neighbor with probability proportional to its value gain.
3. **Sideways Moves.** Allow $ f(s') = f(s) $ for up to $ k $ steps to cross plateaus; risk of cycling.
4. **Random-Restart.** Repeat hill climbing from multiple random seeds; expected to find the global optimum in multimodal landscapes as the number of restarts grows.
5. **Simulated Annealing (connection).** Occasionally accept **worse** moves with probability $ \exp(-\Delta/T) $, where $ T $ is a temperature parameter that decreases over time. Guarantees of global optimality require slow cooling schedules.

---

## Code Cell — First-Choice and Random-Restart

```python
def first_choice_hc(s0, f, neighbors, max_checks=1000):
    s = s0
    for _ in range(max_checks):
        # sample neighbors lazily; accept first improvement
        for s2 in neighbors(s):
            if f(s2) > f(s):
                s = s2
                break
        else:
            return s  # no improvement found
    return s

def random_restart_hc(mk_random_state: Callable, f, neighbors, restarts=20):
    best = None
    for _ in range(restarts):
        s0 = mk_random_state()
        sol = hill_climb(s0, f, neighbors)
        if best is None or f(sol) > f(best):
            best = sol
    return best

# Demonstration
mk_state = lambda: random.randint(-50, 50)
best = random_restart_hc(mk_state, f_int, N_int, restarts=25)
print("RR-HC best:", best, "value:", f_int(best))
```

---

## 4. Convergence, Complexity, and Optimality

- **Completeness.** Hill climbing is **not complete**: it can terminate at non-optimal plateaus or local maxima.
- **Optimality.** Not optimal without exhaustive restarts; **random-restart** improves empirical optimality but provides only probabilistic guarantees.
- **Time/Space.** Each iteration uses $ O(\lvert N(s)\rvert) $ time and constant space; total time depends on the shape of the landscape and number of restarts.

**Theorem (Plateau trapping).** If $ \exists s \in \mathcal{S} $ with a closed neighborhood in which $ f $ is constant and no strictly better neighbor exists within that neighborhood, standard hill climbing terminates on the plateau when entered.

*Proof sketch.* Improvement rule requires $ f(s') > f(s) $ to move; hence no exit if all accessible neighbors share equal value.

---

## 5. Heuristics for Pacman (Intuition)

Let $ s $ denote a Pacman grid state with agent position $ p $, remaining food set $ F $, and ghosts $ G $. A typical heuristic combines distances:

$$
h(s) = -\alpha \cdot d(p, \text{nearest food}) \;-\; \beta \cdot |F| \;+\; \gamma \cdot \min_{g \in G} d(p, g),
$$

with positive weights $ \alpha, \beta, \gamma $. Hill climbing attempts to **maximize** $ h $, so larger ghost distance and smaller food distance count as improvement.

---

# Part II — Minimax (Adversarial Search)

## 6. Game Model

A two-player, deterministic, zero-sum, perfect-information game is defined by
$ (\mathcal{S}, A, P, T, U) $ where

- $ \mathcal{S} $: states; initial state $ s_0 $.
- $ A(s) $: actions available at $ s $.
- $ P(s) \in \{\text{MAX}, \text{MIN}\} $: player to act.
- $ T(s,a) $: successor function.
- $ U(s) $: terminal utility for MAX; MIN’s utility is $ -U(s) $.

**Assumptions.** Perfect play; both players are rational and minimize/maximize the same scalar utility.

---

## 7. Minimax Value and Policy

Define the **minimax value** $ V^\*(s) $ recursively:
$$
V^\*(s) = \begin{cases}
U(s), & \text{if } s \text{ is terminal},\\\\
\max_{a \in A(s)} V^\*(T(s,a)), & \text{if } P(s)=\text{MAX},\\\\
\min_{a \in A(s)} V^\*(T(s,a)), & \text{if } P(s)=\text{MIN}.\\
\end{cases}
$$

The **optimal policy** for MAX at $ s $ selects an action attaining the outer max.

**Complexity.** On a uniform game tree of branching factor $ b $ and depth $ d $, naive minimax examines $ O(b^d) $ nodes.

---

## Pseudocode — Depth-Limited Minimax

```text
function MINIMAX(s, depth, eval):
    if TERMINAL(s) or depth = 0:
        return eval(s)
    if P(s) = MAX:
        value := -infinity
        for a in A(s):
            value := max(value, MINIMAX(T(s,a), depth-1, eval))
        return value
    else:  # MIN
        value := +infinity
        for a in A(s):
            value := min(value, MINIMAX(T(s,a), depth-1, eval))
        return value
```

**Evaluation function $ \text{eval}(s) $.** Approximates $ U(s) $ when $ s $ is non-terminal at the depth limit. Design mirrors the domain’s strategic features (material, mobility, threats, etc.).

---

## 8. Alpha–Beta Pruning

**Idea.** Track bounds $ \alpha $ (best value for MAX so far on the path) and $ \beta $ (best for MIN). If a node’s value cannot improve its ancestor’s choice, **prune** its subtree.

### Correctness Invariant

- At any node $ n $ on a path, $ \alpha $ is a lower bound on the achievable value for MAX above $ n $, and $ \beta $ is an upper bound for MIN above $ n $.
- If at a MAX node we find a child with value $ v \ge \beta $, MIN will avoid this branch earlier; symmetric for MIN when $ v \le \alpha $.

### Complexity

With perfect move ordering, alpha–beta reduces time from $ O(b^d) $ to $ O(b^{d/2}) $ in the best case; space remains $ O(d) $ due to depth-first recursion.

---

## Pseudocode — Alpha–Beta

```text
function ALPHABETA(s, depth, eval, alpha=-infinity, beta=+infinity):
    if TERMINAL(s) or depth = 0:
        return eval(s)
    if P(s) = MAX:
        value := -infinity
        for a in ORDERED(A(s)):   # good ordering improves pruning
            value := max(value, ALPHABETA(T(s,a), depth-1, eval, alpha, beta))
            alpha := max(alpha, value)
            if alpha >= beta:     # beta-cutoff
                break
        return value
    else:  # MIN
        value := +infinity
        for a in ORDERED(A(s)):
            value := min(value, ALPHABETA(T(s,a), depth-1, eval, alpha, beta))
            beta := min(beta, value)
            if alpha >= beta:     # alpha-cutoff
                break
        return value
```

---

## 9. Horizon Effect and Quiescence

**Horizon effect.** Depth limit may hide imminent tactical events (captures, checks, forced wins).

**Quiescence search.** Extend search selectively in **non-quiescent** positions (volatile positions) until a stable evaluation is possible.

**Transposition tables.** Memoize previously evaluated positions to avoid recomputation in games with repeated states.

---

## 10. Worked Micro-Example

We evaluate a small game tree by hand to illustrate alpha–beta cutoffs. Numbers denote returned values from leaves (MAX’s utility).

```text
           MAX
         /     \
       MIN     MIN
      / | \   /   \
     3  12 8  2   14

Move ordering: examine left to right.
- At left MIN: value = min(3,12,8) = 3; alpha becomes max(-inf,3)=3 at root (MAX).
- At right MIN: with alpha=3 and beta=+inf, first child 2 yields beta=min(+inf,2)=2;
  since alpha (3) >= beta (2), prune the remaining child 14.
Final value at root: max(3,2) = 3.
```

---

## Code Cell — Tiny Minimax Engine (for experimentation)

```python
INF = float("inf")

def minimax(s, depth, eval_fn, max_turn: bool):
    if depth == 0 or not s["children"]:
        return eval_fn(s)
    if max_turn:
        v = -INF
        for c in s["children"]:
            v = max(v, minimax(c, depth-1, eval_fn, False))
        return v
    else:
        v = INF
        for c in s["children"]:
            v = min(v, minimax(c, depth-1, eval_fn, True))
        return v

def alphabeta(s, depth, eval_fn, alpha=-INF, beta=INF, max_turn=True):
    if depth == 0 or not s["children"]:
        return eval_fn(s)
    if max_turn:
        v = -INF
        for c in s["children"]:
            v = max(v, alphabeta(c, depth-1, eval_fn, alpha, beta, False))
            alpha = max(alpha, v)
            if alpha >= beta:
                break
        return v
    else:
        v = INF
        for c in s["children"]:
            v = min(v, alphabeta(c, depth-1, eval_fn, alpha, beta, True))
            beta = min(beta, v)
            if alpha >= beta:
                break
        return v

leaf = lambda val: {"children": [], "val": val}
tree = {
    "children": [
        {"children": [leaf(3), leaf(12), leaf(8)]},
        {"children": [leaf(2), leaf(14)]}
    ]
}
eval_leaf = lambda s: s["val"]

print("Minimax:", minimax(tree, depth=2, eval_fn=eval_leaf, max_turn=True))
print("AlphaBeta:", alphabeta(tree, depth=2, eval_fn=eval_leaf, max_turn=True))
```

---

## 11. Linking Heuristics to Evaluation Functions

For non-terminal states,
- **Material features** (e.g., piece values in chess, token counts).
- **Mobility** (number of legal moves).
- **Positional factors** (control of center, king safety, distance-to-goal, etc.).

**Design principle.** Make $ \text{eval}(s) $ correlate strongly with **expected game outcome** under optimal play, while being cheap to compute. Normalize features to comparable scales and combine linearly or with learned weights.

---

## 12. Theoretical Guarantees (Highlights)

- **Minimax optimality.** If both players play optimally and the game is finite and deterministic, minimax returns the game-theoretic value.
- **Alpha–beta correctness.** Pruning preserves the value of the root due to bound propagation invariants; pruned branches cannot influence the root’s value.
- **Ordering effect.** With perfect ordering, the effective branching factor becomes $ \sqrt{b} $; with poor ordering, alpha–beta can degenerate to plain minimax.

---

# Part III — Synthesis & Practice

## 13. When to Use What?

- **Hill climbing** excels in continuous or combinatorial spaces where local improvements are meaningful and evaluation is cheap; use **random-restart** and **sideways moves** to mitigate local traps.
- **Minimax** is appropriate for adversarial domains with explicit turn-taking and a clear terminal utility; use **alpha–beta**, **move ordering**, and **quiescence** to search deeper.

---

## 14. Exercises

1. *Hill climbing on plateaus.* Construct a 2D grid landscape with broad flat regions. Compare standard vs. sideways-move hill climbing for varying $ k $. Report success rates and steps-to-convergence.
2. *Random-restart analysis.* Empirically estimate the probability of finding the global optimum as a function of the number of restarts on a multimodal function (e.g., Rastrigin variants).
3. *Minimax implementation.* Implement depth-limited minimax with alpha–beta on a simple game (tic-tac-toe with symmetry reduction). Measure node counts with different move orderings.
4. *Evaluation design.* Propose and test an evaluation function for a connect-four variant. Compare plain minimax vs. alpha–beta with the same depth limit; report pruning statistics.
5. *Horizon mitigation.* Implement a basic quiescence search that extends only capture moves until no immediate captures remain.

---

## 15. Further Reading

- R. S. Sutton and A. G. Barto. *Reinforcement Learning: An Introduction* (for connections between evaluation and policy improvement).
- S. Russell and P. Norvig. *Artificial Intelligence: A Modern Approach* (chapters on local search and adversarial search).

---

## Appendix A — Formal Notes

**Hill climbing as greedy local search.**
Given a graph $ G=(\mathcal{S},E) $ and $ f: \mathcal{S}\to\mathbb{R} $, the algorithm induces a path $ s_0, s_1, \dots, s_k $ with
$ s_{i+1} \in \arg\max_{s' \in N(s_i)} f(s') $ and $ f(s_{i+1}) > f(s_i) $ when possible.
Termination occurs when $ \forall s' \in N(s_k),\ f(s') \le f(s_k) $.

**Alpha–beta pruning bound.**
Let $ \alpha $ be a lower bound on MAX’s achievable value along the current path, and $ \beta $ an upper bound on MIN’s. If at a MIN node a child returns $ v \le \alpha $, then the parent MIN node will choose a value $ \le v \le \alpha $, which is $ \le $ the value already achievable elsewhere for MAX, so exploration cannot affect the ancestor choice; prune. The MAX case is symmetric.

---

## Open Colab Links 

- Hill Climbing: [Open in Colab](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HillClimbing_Pacman_From_Scratch.ipynb)
- Minimax: [Open in Colab](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Minimax_Tutorial.ipynb)
- Alpha Beta Pruning with Minimax using Connect 4: [Open in Colab](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Connect4_Minimax_AlphaBeta_From_Scratch.ipynb)

---

# Part I Supplement — Hill Climbing (Deep Dive)

## Why Local Search Works (and Fails)

Hill climbing performs **greedy local improvement** by following the gradient of a discrete landscape defined by a neighbor relation $N(s)$ and objective $f(s)$. It is attractive because of **$O(|N(s)|)$ time per step** and **$O(1)$ space**, but it is susceptible to:

- **Local maxima** and **ridges** (requires sequences of lateral moves).
- **Plateaus** (zero gradient regions).
- **Noisy objectives** (stochastic evaluations).

**Guarantee.** With strictly improving moves, the algorithm halts at a **1-opt** solution:
$$
\forall s' \in N(s_\mathrm{halt}):\ f(s') \le f(s_\mathrm{halt}).
$$

---

## Remedies with Theory Backing

1. **First-Choice Hill Climbing:** reduces evaluation overhead; in large neighborhoods it approximates a random sample of $N(s)$ and succeeds quickly if the **density of improving neighbors** is nontrivial.
2. **Sideways Moves ($k$-limit):** escape plateaus but risk cycles. Use **tabu memory** (recently visited states) to avoid short cycles.
3. **Random Restarts:** If global optima occupy nonzero measure in the basin landscape, the success probability after $r$ independent restarts is
$$
1-(1-p)^r,
$$
where $p$ is the single-run success probability.
4. **Stochastic Acceptance (Simulated Annealing):** occasional downhill moves with probability $\exp(-\Delta/T)$; with logarithmic cooling and ergodicity, global optimality is guaranteed in the limit.

---

## Worked Exercise (Plateau Crossing)

1. Construct a plateau where all $|N(s)|=d$ neighbors have equal value except one “exit” two steps away.  
2. Compare expected steps for **steepest-ascent**, **first-choice**, and **sideways-$k$** variants.  
3. Report empirical success over 100 trials.

[[MC]]
Which variant is most likely to traverse broad plateaus **without** exhaustive neighbor evaluation?
- (x) First-Choice
- ( ) Steepest-Ascent
- ( ) Pure Random Walk
- ( ) Deterministic Gradient with $k=0$

---

# Part II Supplement — Minimax & Alpha–Beta (Deep Dive with Connect‑4)

## From Game Theory to Search

Two-player, zero-sum, perfect-information games admit a **game-theoretic value** defined by
$$
V^*(s) = \begin{cases}
U(s), & \text{terminal} \\
\max_{a \in A(s)} V^*(T(s,a)), & P(s)=\text{MAX} \\
\min_{a \in A(s)} V^*(T(s,a)), & P(s)=\text{MIN}.
\end{cases}
$$

On a uniform tree with branching factor $b$ and depth $d$, naive minimax evaluates $O(b^d)$ nodes.

---

## Alpha–Beta in Practice

Alpha–beta maintains bounds $\alpha$ and $\beta$ that permit **pruning** subtrees that cannot affect the root choice. With **good move ordering**, time shrinks to $O(b^{d/2})$ while space remains $O(d)$.

**Move ordering heuristics** (especially effective in Connect‑4):
- Prefer **center columns** first (more winning lines).
- Try **immediate threats** and **forced wins** before neutral moves.
- Reuse the previous iteration’s **principal variation** (iterative deepening).

---

## Evaluation Functions for Connect‑4

Design a fast, informative $\text{eval}(s)$ for nonterminal $s$:
- **Material:** counts of open 2‑in‑a‑row and 3‑in‑a‑row patterns.
- **Blocking:** penalties if opponent has immediate 3 with an open 4.
- **Positional:** center-column weights.
- **Terminal checks:** detect 4‑in‑a‑row swiftly; return $\pm \infty$ at leaves.

A linear model is common:
$$
\text{eval}(s)=w_1\,\Phi_\text{two}(s)+w_2\,\Phi_\text{three}(s)+w_3\,\Phi_\text{center}(s)-w_4\,\Phi_\text{oppThreat}(s).
$$

---

## Notebook Integration: Connect‑4 Minimax + Alpha–Beta

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/Connect4_Minimax_AlphaBeta_From_Scratch.ipynb)

This notebook walks through:
- Board representation and **legal move generation**.
- Depth‑limited minimax with **alpha–beta pruning**.
- **Iterative deepening** and **move ordering** strategies.
- Instrumentation: **node counts**, **prune counts**, and **effective branching factor**.

---

## Code Excerpt — Ordering + Alpha–Beta (Illustrative)

```python
def order_moves(state, moves, player):
    # Heuristic: center-first, then proximity to center
    center = state.width // 2
    return sorted(moves, key=lambda m: abs(m - center))

def alphabeta_connect4(state, depth, alpha, beta, maximizing):
    if depth == 0 or state.is_terminal():
        return evaluate(state), None
    best_move = None
    if maximizing:
        value = -float("inf")
        for m in order_moves(state, state.legal_moves(), +1):
            v, _ = alphabeta_connect4(state.play(m), depth-1, alpha, beta, False)
            if v > value:
                value, best_move = v, m
            alpha = max(alpha, value)
            if alpha >= beta:  # beta cutoff
                break
        return value, best_move
    else:
        value = float("inf")
        for m in order_moves(state, state.legal_moves(), -1):
            v, _ = alphabeta_connect4(state.play(m), depth-1, alpha, beta, True)
            if v < value:
                value, best_move = v, m
            beta = min(beta, value)
            if alpha >= beta:  # alpha cutoff
                break
        return value, best_move
```

---

## Quiescence, Transpositions, and Iterative Deepening

- **Quiescence:** extend only **tactical** nodes (immediate wins/blocks) until calm.  
- **Transposition Table:** hash positions; store (depth, value, bounds) to avoid recomputation.  
- **Iterative Deepening:** search depth $1,2,\dots,d$; reuse best line (PV) to improve ordering.

[[MC]]
Which technique primarily **improves move ordering** for deeper alpha–beta searches?
- ( ) Quiescence
- (x) Iterative deepening with PV reordering
- ( ) Transposition tables
- ( ) Aspiration windows

---

## Measuring Search Efficiency (Activity)

Instrument your Colab with counters:
1. Number of nodes expanded per depth.
2. Number of prunes (alpha/beta cutoffs).
3. Effective branching factor $\hat{b} = N^{1/d}$.

**Exercise.** Compare **center-first ordering** vs. **random ordering** at the same depth; report speedup and PV stability.

---

## Discussion Prompt

> In Connect‑4, why is center control disproportionately valuable for search efficiency and evaluation accuracy?

---

# Part III Supplement — Synthesis & Practice Enhancements

- **When to Use Local vs. Adversarial Search:** Map problems (TSP neighborhood search) vs. games (Connect‑4).  
- **Hybridization:** Use **local search** to tune evaluation weights used by minimax.  
- **Empirical Protocol:** fix seeds, count nodes, plot depth vs. nodes on log-scale; verify $O(b^{d/2})$ behavior with improved ordering.

