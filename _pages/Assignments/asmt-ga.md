---
layout: assignment
permalink: Assignments/GeneticAlgorithm
title: "Assignment: Optimization with Genetic Algorithms (GA) — From Scratch and with Libraries"

info:
  points: 100
  goals:
    - Build intuition for stochastic search and evolutionary optimization.
    - Implement a Genetic Algorithm (GA) from scratch for a meaningful toy problem.
    - Compare your implementation with a reference library (e.g., DEAP) on the same task.
    - Analyze convergence, hyperparameters, and solution quality.
  purpose: "This assignment develops practical competency with Genetic Algorithms by taking you from first principles to a robust implementation. You will engineer representation, selection, crossover, mutation, and replacement; study convergence; and reflect on design choices and ethics around optimization."
  concepts:
    - Representation (encoding) and fitness functions
    - Selection (roulette, tournament), crossover, mutation, elitism
    - Premature convergence and diversity maintenance
    - Constraint handling and penalty methods
    - Statistical comparison across runs (random seeds)
  tasks:
    - Implement a GA from scratch on a constrained knapsack-style problem.
    - Instrument and visualize fitness over generations.
    - Re-solve with a GA library (DEAP) and compare performance.
    - Conduct sensitivity analysis over population size, mutation rate, and selection scheme.
  rubric:
    - weight: 30
      description: Implementation
      preemerging: Basic GA loop with partial operators; fragile under edge cases.
      beginning: Correct selection and variation operators; runs on provided task.
      progressing: Modular, well-tested GA with instrumentation and plots.
      proficient: Robust design with clean APIs, reproducible experiments, and insightful diagnostics.
    - weight: 30
      description: Algorithmic Correctness and Reasoning
      preemerging: Limited justification of choices and parameters.
      beginning: Explains encoding and operators with small tests.
      progressing: Provides ablations and compares operators quantitatively.
      proficient: Deep reasoning, constraint treatment, and statistical comparison across seeds.
    - weight: 20
      description: Code Quality and Documentation
      preemerging: Minimal comments and structure.
      beginning: Clear structure with docstrings and basic tests.
      progressing: Clean modular design, typing hints, and readable plots/tables.
      proficient: Excellent organization, experiment harness, and reproducibility.
    - weight: 10
      description: Analysis and Interpretation
      preemerging: Surface-level commentary.
      beginning: Explains trends in fitness curves and parameter effects.
      progressing: Connects diversity, selection pressure, and convergence behavior.
      proficient: Insightful narrative with evidence-backed conclusions.
    - weight: 10
      description: Submission Completeness
      preemerging: Partial artifacts or missing results.
      beginning: Includes code and brief write-up.
      progressing: Includes code, results, and plots with clear instructions.
      proficient: Fully reproducible package with report, code, configs, and seed control.

tags:
  - ai
  - optimization
  - evolutionary-computation
---

# Overview

In this assignment, you will implement a **Genetic Algorithm (GA)** to optimize a small but meaningful problem and compare it to a library-based GA solution. You will study the effects of representation, selection pressure, crossover, mutation, and elitism on solution quality and convergence.

We use **$K$-item knapsack** as a running example: select items with values $v_i$ and weights $w_i$ to maximize total value subject to capacity $W$. The binary decision vector is $$x \in \{0,1\}^n$$.

**Objective (penalized form):**
$$
\max_{x \in \{0,1\}^n} \; f(x) = \sum_{i=1}^n v_i x_i - \lambda \max\!\left(0, \sum_{i=1}^n w_i x_i - W \right).
$$

---

## Stage 0 — Setup

Use Python 3.9+ and the libraries below. The DEAP section is optional if DEAP is unavailable in your environment; you may implement both parts in a single notebook.

```python
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(42)

# Optional (library portion)
try:
    import deap
    from deap import base, creator, tools, algorithms
    HAS_DEAP = True
except Exception as e:
    HAS_DEAP = False
    print("DEAP not available; you can still complete the from-scratch portion.")
```

---

## Stage 1 — Problem Instance & Fitness

We create a toy knapsack with $$n=40$$ items; values and weights are positive integers; capacity $W$ controls feasibility. The fitness uses a penalty coefficient $$\lambda$$ for overweight solutions.

```python
n = 40
values  = rng.integers(10, 100, size=n)
weights = rng.integers(5, 40, size=n)
W = int(0.3 * weights.sum())  # capacity
lam = 10.0                    # penalty coefficient

def fitness(x):
    x = np.asarray(x, dtype=int)
    total_v = (values * x).sum()
    overflow = max(0, (weights * x).sum() - W)
    return total_v - lam * overflow
```

**Checkpoint:** What happens to solutions if you set $$\lambda=0$$? Explain why a positive penalty is required for constraint satisfaction.

---

## Stage 2 — GA From Scratch

### 2.1 Representation & Initialization

We use a binary vector (0/1) of length $n$.

```python
def init_individual(n, p_one=0.3):
    return (rng.random(n) < p_one).astype(int)

def init_population(pop_size, n):
    return [init_individual(n) for _ in range(pop_size)]
```

### 2.2 Selection, Crossover, Mutation, Elitism

We implement **tournament selection**, **one-point crossover**, and **flip-bit mutation**. Elitism keeps the top $k$ individuals from the previous generation.

```python
def tournament_select(pop, k=3):
    i = rng.integers(0, len(pop), size=k)
    cand = [pop[j] for j in i]
    fit = [fitness(c) for c in cand]
    return cand[int(np.argmax(fit))].copy()

def one_point_crossover(a, b, p_c=0.9):
    if rng.random() >= p_c or len(a) < 2:
        return a.copy(), b.copy()
    cx = rng.integers(1, len(a))
    c1 = np.concatenate([a[:cx], b[cx:]])
    c2 = np.concatenate([b[:cx], a[cx:]])
    return c1, c2

def bit_flip_mutation(x, p_m=0.01):
    mask = rng.random(len(x)) < p_m
    y = x.copy()
    y[mask] = 1 - y[mask]
    return y

def elitism(pop, new_pop, k=2):
    elite_idx = np.argsort([-fitness(ind) for ind in pop])[:k]
    elites = [pop[i].copy() for i in elite_idx]
    # Replace worst k in new_pop with elites
    worst_idx = np.argsort([fitness(ind) for ind in new_pop])[:k]
    for idx, e in zip(worst_idx, elites):
        new_pop[idx] = e
    return new_pop
```

### 2.3 Main GA Loop with Instrumentation

```python
def run_ga(pop_size=80, generations=150, p_c=0.9, p_m=0.01, k_tourn=3, elite_k=2, seed=42):
    global rng
    rng = np.random.default_rng(seed)
    pop = init_population(pop_size, n)
    best_hist, mean_hist = [], []
    for g in range(generations):
        fitnesses = np.array([fitness(ind) for ind in pop])
        best_hist.append(fitnesses.max())
        mean_hist.append(fitnesses.mean())

        # Mating
        new_pop = []
        while len(new_pop) < pop_size:
            p1, p2 = tournament_select(pop, k_tourn), tournament_select(pop, k_tourn)
            c1, c2 = one_point_crossover(p1, p2, p_c=p_c)
            c1, c2 = bit_flip_mutation(c1, p_m=p_m), bit_flip_mutation(c2, p_m=p_m)
            new_pop.extend([c1, c2])
        new_pop = new_pop[:pop_size]
        pop = elitism(pop, new_pop, k=elite_k)

    # Final eval
    fitnesses = np.array([fitness(ind) for ind in pop])
    best = pop[int(np.argmax(fitnesses))]
    return best, best_hist, mean_hist

best, best_hist, mean_hist = run_ga()
plt.plot(best_hist, label="Best")
plt.plot(mean_hist, label="Mean")
plt.xlabel("Generation"); plt.ylabel("Fitness"); plt.legend(); plt.show()

print("Best fitness:", fitness(best))
print("Weight used:", (weights*best).sum(), "/ capacity", W)
```

**Checkpoint:** Report final constraint satisfaction $$\sum_i w_i x_i \le W$$ and total value $$\sum_i v_i x_i$$ of your best individual.

---

## Stage 3 

### Stage 3a - Library GA using DEAP

If DEAP is available, re-solve the same knapsack instance.

```python
if HAS_DEAP:
    def eval_knapsack(individual):
        total_v = int(np.dot(values, individual))
        overflow = max(0, int(np.dot(weights, individual)) - W)
        return (total_v - int(lam*overflow),)

    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", np.ndarray, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("attr_bool", rng.integers, 0, 2)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", eval_knapsack)
    toolbox.register("mate", tools.cxOnePoint)
    toolbox.register("mutate", tools.mutFlipBit, indpb=0.01)
    toolbox.register("select", tools.selTournament, tournsize=3)

    pop = toolbox.population(n=80)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values[0])
    stats.register("avg", np.mean); stats.register("max", np.max)

    pop, log = algorithms.eaSimple(pop, toolbox, cxpb=0.9, mutpb=0.2, ngen=150, stats=stats, halloffame=hof, verbose=False)
    print("DEAP best fitness:", hof[0].fitness.values[0])
    print("DEAP weight:", int(np.dot(weights, hof[0])), "/ capacity", W)
else:
    print("Skip DEAP portion; library not available in this environment.")
```

**Comparison prompts**
- How do best/mean fitness curves differ between your GA and DEAP?
- Does DEAP converge faster or reach higher fitness? Hypothesize why (operator implementations, mutation rates, selection pressure).

---

### Stage 3b — Traveling Salesman Problem (TSP) with a GA (Setup)

We extend the GA to a classic **combinatorial optimization** task: the **Traveling Salesman Problem** (TSP). You will implement a **permutation-based GA** that searches over city visit orders to minimize total tour length.

> **You are given scaffolding below.** Implement the missing operators and complete the GA loop. You may also compare against a DEAP-based permutation GA if available.

### Problem Definition

Given $N$ cities with coordinates $$\{(x_i, y_i)\}_{i=1}^N$$, find a permutation $$\pi$$ of $$\{1,\dots,N\}$$ minimizing the closed tour length:
$$
L(\pi) \;=\; \sum_{k=1}^{N} d\big(\pi_k,\, \pi_{k+1}\big), \quad \text{with } \pi_{N+1} \equiv \pi_1,
$$
where $d(i,j)$ is Euclidean distance between cities $i$ and $j$.

### 3b.1 Data & Distance Matrix

```python
import numpy as np
import matplotlib.pyplot as plt
rng = np.random.default_rng(7)

# Generate toy cities on the unit square
N = 30
coords = rng.random((N, 2))  # shape (N, 2)

# Precompute distance matrix
D = np.sqrt(((coords[:,None,:] - coords[None,:,:])**2).sum(axis=2))

def tour_length(perm):
    # perm: array of city indices (permutation of range(N))
    return sum(D[perm[i], perm[(i+1) % N]] for i in range(N))
```

**Checkpoint.** Verify `tour_length(np.arange(N))` returns a finite number and decreases if you rearrange cities sensibly (e.g., nearest-neighbor heuristic).

### 3b.2 Representation & Initialization

- **Representation:** a **permutation** (no duplicates) of `[0, 1, ..., N-1]`.
- **Initialization:** random permutations.

```python
def init_perm():
    p = np.arange(N)
    rng.shuffle(p)
    return p

def init_population_perm(pop_size):
    return [init_perm() for _ in range(pop_size)]
```

### 3b.3 Selection, Crossover, Mutation (Implement Me)

For permutations, use **permutation-safe** operators:

- **Selection:** tournament selection (works unchanged).
- **Crossover (must be permutation-safe):**
  - **Ordered Crossover (OX)**, **Partially Mapped Crossover (PMX)**, or **Cycle Crossover (CX)**.
- **Mutation (permutation-preserving):**
  - **Swap mutation** (swap two positions),
  - **Insert mutation** (remove an element and reinsert elsewhere),
  - **Scramble mutation** (randomly permute a slice).

Below is **scaffolding**; complete `pmx_crossover` **or** `ordered_crossover`, and at least one mutation:

```python
def tournament_select_perm(pop, k=3):
    idx = rng.integers(0, len(pop), size=k)
    cand = [pop[i] for i in idx]
    fit = [-tour_length(c) for c in cand]  # maximize negative length
    return cand[int(np.argmax(fit))].copy()

# TODO: Implement ONE of the permutation crossovers (PMX or OX)
def pmx_crossover(a, b, p_c=0.9):
    a, b = a.copy(), b.copy()
    if rng.random() >= p_c:
        return a, b
    n = len(a)
    i, j = sorted(rng.integers(0, n, size=2))
    # --- Your PMX implementation here ---
    # 1) Copy slice a[i:j] into child1, b[i:j] into child2
    # 2) Map remaining positions by resolving conflicts using the mapping induced by the slices
    # Return children c1, c2
    # For grading, keep the API the same:
    c1, c2 = a.copy(), b.copy()  # placeholder (replace)
    return c1, c2

# Alternative (if you prefer OX):
def ordered_crossover(a, b, p_c=0.9):
    a, b = a.copy(), b.copy()
    if rng.random() >= p_c:
        return a, b
    n = len(a)
    i, j = sorted(rng.integers(0, n, size=2))
    # --- Your OX implementation here ---
    # 1) Copy a[i:j] into c1, preserve order of remaining from b
    # 2) Copy b[i:j] into c2, preserve order of remaining from a
    c1, c2 = a.copy(), b.copy()  # placeholder (replace)
    return c1, c2

# TODO: Implement at least one permutation mutation
def swap_mutation(p, p_m=0.2):
    q = p.copy()
    if rng.random() < p_m:
        i, j = rng.integers(0, len(p), size=2)
        q[i], q[j] = q[j], q[i]
    return q

# (Optional) insert_mutation / scramble_mutation can be added similarly
```

### 3b.4 GA Loop (Permutation Version)

```python
def run_ga_tsp(pop_size=120, generations=400, p_c=0.9, p_m=0.2, k_tourn=3, elite_k=4, seed=7):
    global rng
    rng = np.random.default_rng(seed)
    pop = init_population_perm(pop_size)
    best_hist, mean_hist = [], []

    for g in range(generations):
        lengths = np.array([tour_length(ind) for ind in pop])
        best_hist.append(lengths.min())
        mean_hist.append(lengths.mean())

        # Mating
        new_pop = []
        while len(new_pop) < pop_size:
            p1 = tournament_select_perm(pop, k_tourn)
            p2 = tournament_select_perm(pop, k_tourn)
            # Choose your implemented crossover:
            c1, c2 = ordered_crossover(p1, p2, p_c=p_c)  # or pmx_crossover(...)
            c1, c2 = swap_mutation(c1, p_m=p_m), swap_mutation(c2, p_m=p_m)
            new_pop.extend([c1, c2])
        new_pop = new_pop[:pop_size]

        # Elitism: copy best tours forward
        elite_idx = np.argsort([tour_length(ind) for ind in pop])[:elite_k]
        elites = [pop[i].copy() for i in elite_idx]
        # Replace worst elites in new_pop
        worst_idx = np.argsort([tour_length(ind) for ind in new_pop])[-elite_k:]
        for wi, e in zip(worst_idx, elites):
            new_pop[wi] = e
        pop = new_pop

    lengths = np.array([tour_length(ind) for ind in pop])
    best_idx = int(np.argmin(lengths))
    return pop[best_idx], best_hist, mean_hist

best_perm, best_hist_tsp, mean_hist_tsp = run_ga_tsp()
print("Best tour length:", tour_length(best_perm))

plt.figure(figsize=(5,3))
plt.plot(best_hist_tsp, label="Best tour length")
plt.plot(mean_hist_tsp, label="Mean tour length")
plt.xlabel("Generation"); plt.ylabel("Tour length"); plt.legend(); plt.tight_layout(); plt.show()
```

### 3b.5 Visualization

```python
def plot_tour(perm, title="GA TSP Tour"):
    cyc = np.r_[perm, perm[0]]
    plt.figure(figsize=(4.5,4.5))
    plt.scatter(coords[:,0], coords[:,1], c="k", s=15)
    plt.plot(coords[cyc,0], coords[cyc,1], lw=1.5)
    for i,(x,y) in enumerate(coords):
        plt.text(x, y, str(i), fontsize=7)
    plt.title(title); plt.axis("equal"); plt.tight_layout(); plt.show()

plot_tour(best_perm, title=f"GA TSP Tour (L={tour_length(best_perm):.3f})")
```

### 3b.6 (Optional) DEAP Permutation GA

If DEAP is available, you may implement a permutation GA using `tools.cxOrdered` or `tools.cxPartialyMatched`, and `tools.mutShuffleIndexes`. Compare convergence and final tour length to your scratch implementation.

**Reporting Prompts**
- Which crossover (OX vs. PMX) performed better on your instance?  
- How did $p_c$, $p_m$, and elitism affect convergence and diversity?  
- Compare your GA tour length to a simple **Nearest Neighbor** heuristic baseline.

---

## Stage 4 — Sensitivity and Ablations

Conduct **at least two** of the following ablations, plotting best/mean fitness per generation:

1. **Population size:** $$N \in \{40, 80, 160\}$$  
2. **Mutation rate:** $$p_m \in \{0.005, 0.01, 0.05\}$$  
3. **Tournament size:** $$k \in \{2, 3, 5\}$$  
4. **Crossover prob:** $$p_c \in \{0.6, 0.9\}$$  
5. **Elitism:** Enable vs. disable (set elite\_k to 0)

Discuss effects on **diversity**, **convergence speed**, and **solution quality**. Explain any observed **premature convergence**.

---

## Stage 5 — Ethics & Responsible Optimization (Short Reflection)

Optimization choices can embed values and trade-offs:

- **Penalty design** influences which constraints are prioritized.  
- **Fairness:** when optimizing allocations (e.g., resources), ensure constraints do not disproportionately exclude groups.  
- **Reproducibility:** report random seeds and hyperparameters to support auditability.

Provide a short paragraph reflecting on these issues in your design choices.

---

# What to Submit

1. A notebook (or scripts) implementing the **from-scratch GA** and, if available, the **DEAP** version.  
2. Plots of best/mean fitness across generations for your baseline and ablations.  
3. A brief report (2–3 pages) describing: encoding, operators, parameter choices, results, and a sensitivity analysis.  
4. Your final best solution and a feasibility check: $$\sum_i w_i x_i \le W$$ and total value $$\sum_i v_i x_i$$.  
5. (Optional) Explore **repair operators** that fix infeasible individuals instead of using penalties; compare outcomes.

---
