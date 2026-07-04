# Optimization and Local Search: From Hill Climbing to Genetic Algorithms
<!--
author:   William M. Mongan
language: en
narrator: US English Male
comment:  Render via https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-optimizationga.md
import:   https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md
link:     https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-5
           https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Optimization and Local Search: From Hill Climbing to Genetic Algorithms

Last week we searched for a **path**. This week we search for a **thing**: the best schedule, the best circuit layout, the best set of knapsack items. When only the final configuration matters (not how we got there), we can throw away the path and use **local search** — and when the landscape is treacherous, we can evolve solutions with **genetic algorithms (GAs)**.

---

## Open Colab: Hill Climbing (Pacman, From Scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/HillClimbing_Pacman_From_Scratch.ipynb)

---

## Open Colab: Genetic Algorithm (From Scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/GeneticAlgorithm_FromScratch_ToyExample.ipynb)

---

## 1. Intuition First: Fog on the Mountain

Picture yourself on a foggy mountainside trying to reach the highest peak. You cannot see the summit — only the slope under your feet. A reasonable plan: **always step uphill**. That is **hill climbing**, and its failure modes are visible immediately:

- You might summit a **local maximum** — a small hill that is not the mountain.
- You might wander a **plateau** where every direction feels flat.
- You might straddle a **ridge** where every single step goes down even though a diagonal *pair* of steps goes up.

Every fix we study — random restarts, simulated annealing's "sometimes go downhill," and the population-based exploration of genetic algorithms — is a strategy for escaping these traps.

---

## 2. The Optimization Problem, Formally

We seek

$$
x^* = \arg\max_{x \in \mathcal{X}} f(x)
$$

where $\mathcal{X}$ is a (often astronomically large) space of **candidate solutions** and $f$ is an **objective** (or **fitness**) function. Local search keeps a *current* candidate and moves among **neighbors** $N(x)$; it uses $O(1)$ memory — no frontier at all.

Micro-steps for hill climbing:

1. Start at some $x$. *(Anywhere — often random.)*
2. Look at neighbors $N(x)$ and their scores $f$. *(Feel the slope.)*
3. If some neighbor $x'$ has $f(x') > f(x)$, move there and repeat. *(Step uphill.)*
4. Otherwise stop: $x$ is a **local optimum**. *(No promise it is global!)*

The full treatment of hill climbing and its variants (first-choice, random-restart, simulated annealing) appears in the **Heuristic Search** module later in the course — today we take just enough to motivate evolution.

---

## 3. Genetic Algorithms: Evolution as a Search Strategy

A GA maintains not one candidate but a **population**, and improves it with operators borrowed from biology:

| Biology | GA Operator | What it does for search |
|---|---|---|
| Chromosome | **Encoding** (e.g., a bit string) | Represents one candidate solution |
| Survival of the fittest | **Selection** | Focuses effort on promising candidates (exploitation) |
| Reproduction | **Crossover** | Combines *parts* of two good solutions (big jumps) |
| Mutation | **Mutation** | Randomly flips pieces (local exploration; preserves diversity) |
| Generations | **Replacement / elitism** | Iterates; elitism guarantees the best never gets lost |

The loop:

1. **Initialize** a random population of $P$ individuals.
2. **Evaluate** the fitness $f$ of each individual.
3. **Select** parents with probability increasing in fitness (roulette wheel or tournaments).
4. **Crossover** pairs of parents to produce children.
5. **Mutate** each child with small probability per gene.
6. **Replace** the old population (optionally keeping the elite) and go to 2.

---

## 4. One Generation, Fully By Hand

**Problem:** maximize $f(x) = x^2$ for integers $x \in [0, 31]$, encoding $x$ as 5 bits. Population size 4.

**Step 1 — Evaluate.**

| # | Chromosome | $x$ | $f(x)=x^2$ | Share of total |
|---|---|---|---|---|
| 1 | `01101` | 13 | 169 | $169/1170 \approx 0.14$ |
| 2 | `11000` | 24 | 576 | $576/1170 \approx 0.49$ |
| 3 | `01000` | 8  | 64  | $64/1170 \approx 0.06$ |
| 4 | `10011` | 19 | 361 | $361/1170 \approx 0.31$ |

Total fitness $= 169 + 576 + 64 + 361 = 1170$.

**Step 2 — Select (roulette wheel).** Spin a wheel whose slice sizes are the shares above. Individual 2 (49%) will typically be drawn about twice; individual 3 (6%) will often vanish. Suppose we draw parents $\{1, 2\}$ and $\{2, 4\}$. *(Fitter individuals earn more offspring — but nothing is certain, which preserves diversity.)*

**Step 3 — Crossover.** Pick a random cut point, say after bit 4, for parents `0110|1` and `1100|0`:

- Child A = `0110` + `0` = `01100` ($x=12$)
- Child B = `1100` + `1` = `11001` ($x=25$, fitness $625$ — **better than either parent!**)

This is the magic of crossover: it can assemble the *high-order bits* of one parent with the *low-order bits* of another. *(Formally, short high-fitness patterns — "building blocks" — tend to proliferate.)*

**Step 4 — Mutate.** With probability, say, $p_m = 0.01$ per bit, flip bits. If child B's third bit flips: `11101` ($x=29$, fitness $841$). Mutation is the only way to reintroduce a bit value that has died out of the population entirely.

**Step 5 — Replace and repeat.** After a few generations the population converges toward `11111` ($x=31$, fitness $961$).

---

## 5. Runnable Code: A Tiny GA

```python
import random
random.seed(477)

TARGET_LEN = 20                      # maximize number of 1s ("OneMax")
POP, GENS, P_MUT = 30, 40, 0.02

def fitness(ind):        return sum(ind)
def random_ind():        return [random.randint(0, 1) for _ in range(TARGET_LEN)]

def tournament(pop, k=3):
    return max(random.sample(pop, k), key=fitness)

def crossover(a, b):
    cut = random.randrange(1, TARGET_LEN)
    return a[:cut] + b[cut:]

def mutate(ind):
    return [1 - g if random.random() < P_MUT else g for g in ind]

pop = [random_ind() for _ in range(POP)]
for gen in range(GENS):
    best = max(pop, key=fitness)
    if gen % 10 == 0:
        print(f"gen {gen:3d}  best fitness = {fitness(best)}/{TARGET_LEN}")
    elite = [best]                                   # elitism: keep the champion
    pop = elite + [mutate(crossover(tournament(pop), tournament(pop)))
                   for _ in range(POP - 1)]
print("final best:", max(pop, key=fitness), "fitness", fitness(max(pop, key=fitness)))
```

**What to look for:** best fitness climbs quickly at first (easy gains from selection + crossover), then crawls — the classic **exploration/exploitation** tension. Try raising `P_MUT` to 0.2: too much mutation destroys good solutions faster than selection can keep them.

---

## 6. Design Choices That Make or Break a GA

1. **Encoding.** Bit strings suit knapsack-style problems; permutations suit routing/scheduling (with order-preserving crossover); real-valued vectors suit parameter tuning.
2. **Fitness shaping.** For *constrained* problems, subtract a **penalty** for constraint violations (e.g., knapsack weight over capacity) rather than discarding infeasible children outright.
3. **Selection pressure.** Tournaments of size $k$: bigger $k$ → faster convergence but higher risk of **premature convergence** to a mediocre solution.
4. **Diversity maintenance.** Mutation rate, immigrants (fresh random individuals), or fitness sharing keep the population from collapsing to clones.
5. **Statistics across seeds.** GAs are randomized — report means over multiple runs, never a single lucky run. (You will do exactly this in the GA programming assignment.)

---

## 7. When Should You Reach for a GA?

- ✅ Objective is a **black box** (no gradients), the landscape is rugged, or candidates are discrete structures.
- ✅ You can afford many fitness evaluations and want *good* solutions, not provably optimal ones.
- ❌ The problem is smooth and differentiable — **gradient descent** (coming in the regression module!) will be far more efficient.
- ❌ A classical exact algorithm exists (shortest paths → use UCS/A*, not evolution).

GAs return later in the course as an alternative to reinforcement learning for policy search — see the **Reinforcement Learning** module.

---

## Comprehension Quiz

[[MC]]
In the worked example, crossover of `01101` and `11000` produced `11001` with fitness 625, exceeding both parents (169, 576). Which GA principle does this illustrate?
- ( ) Mutation reintroduces lost genetic material.
- (x) Crossover can combine complementary building blocks from two parents.
- ( ) Elitism guarantees monotone improvement.
- ( ) Roulette-wheel selection is unbiased.

[[MC]]
Your GA's population becomes nearly identical after 5 generations and stops improving, far from the optimum. Which change is *most* directly aimed at this failure?
- ( ) Decrease the mutation rate to zero.
- ( ) Increase tournament size from 3 to 10.
- (x) Increase the mutation rate or inject random individuals to restore diversity.
- ( ) Remove elitism so the best individual can be lost.

---

## Discussion Prompt

> Optimization is never neutral: a GA maximizes exactly the fitness function you write. Suppose you evolve staff schedules with fitness = "labor cost minimized." What stakeholder concerns does that objective silently ignore, and how would you encode them?
