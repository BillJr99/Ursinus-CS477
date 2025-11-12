# Informed Search: A* and Admissible Heuristics
<!--
author:   William M. Mongan
language: en
narrator: US English Male

comment: Render with https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-informedsearch.md or locally if deployed via https://www.billmongan.com/LiaScript/?https://raw.githubusercontent.com/BillJr99/Ursinus-CS477/gh-pages/_pages/Activities/liascript-informedsearch.md

import: https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md

link:   https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-4
        https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Informed Search: A* and Admissible Heuristics

This module develops **informed search** with a focus on **A\***, admissible and consistent heuristics, and common variants. We move from **intuition $\rightarrow$ formal models $\rightarrow$ algorithmics $\rightarrow$ guarantees $\rightarrow$ practice** with code cells and small proofs or proof sketches.

---

## Open Colab: A* (Step-by-Step)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/A_star_step_by_step.ipynb)

---

## Open Colab: A* With Graphics

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/AStar_Search_With_Graphics.ipynb)

---

## 0. Environment & Utilities

We will use small Python fragments to match the algorithmic descriptions. The Colab notebooks above provide full, visualized implementations.

---

## Code Cell
```python
import heapq
from typing import Dict, Tuple, Callable, List

print("A* utilities ready.")
```

---

# Part I — Problem Formulation

## 1. Graph Search Model

A search problem is a tuple $(\mathcal{S}, A, T, c, s_0, \mathcal{G})$ where:

- $\mathcal{S}$: states; $s_0 \in \mathcal{S}$ initial state.
- $A(s)$: actions available at $s$.
- $T(s,a)$: successor function, returning $s' = T(s,a)$.
- $c(s,a)>0$: step-cost; path cost is $g(s) = \sum c(s_i,a_i)$.
- $\mathcal{G}\subseteq \mathcal{S}$: goal set.

**Objective.** Find a least-cost path from $s_0$ to any $g \in \mathcal{G}$.

---

## 2. Heuristics

A **heuristic** is a function $h:\mathcal{S}\to \mathbb{R}_{\ge 0}$ that estimates the cheapest remaining cost to a goal.

- **Admissible:** $0 \le h(s) \le h^*(s)$ for all $s$, where $h^*(s)$ is the true cost-to-go.
- **Consistent (Monotone):** for all transitions $s \xrightarrow{a} s'$, 
  $$
  h(s) \le c(s,a) + h(s') \quad \text{and} \quad h(g)=0 \text{ for } g\in\mathcal{G}.
  $$

Consistency $\Rightarrow$ admissibility. With consistency, $f$-values along any path are nondecreasing.

---

# Part II — A* Search

## 3. Evaluation Function

A* expands nodes by nondecreasing
$$
f(n) = g(n) + h(n),
$$
where $g(n)$ is the cost from the start to $n$ and $h(n)$ estimates the cost to goal.

---

## Pseudocode — Graph-Search A* (with Closed List)

```text
function ASTAR(start, is_goal, neighbors, cost, h):
    open := priority-queue ordered by f = g + h
    push start with g(start)=0, f(start)=h(start)
    closed := empty map  # best known g for each state

    while open not empty:
        n := pop node with smallest f
        if is_goal(n.state): return reconstruct_path(n)
        if n.state in closed and n.g >= closed[n.state]: continue
        closed[n.state] := n.g
        for (s2, a) in neighbors(n.state):
            g2 := n.g + cost(n.state, a, s2)
            f2 := g2 + h(s2)
            push/update (s2, g2, f2, parent=n, via=a)
    return failure
```

**Tie-breaking.** When $f$-ties occur, breaking in favor of **larger $g$** can reduce re-expansions with consistent $h$.

---

## 4. Optimality Guarantee (Sketch)

If $h$ is **admissible** and either tree-search avoids duplicates or graph-search uses **closed** with **best-$g$** retention, A* returns an optimal goal.

*Proof sketch.* Let $G^*$ be the optimal goal with cost $C^*$. A* never expands a node $n$ with $f(n) > C^*$ before some optimal frontier node $n^*$ with $f(n^*) \le C^*$. Once a goal is popped with $g= C^*$, no cheaper path exists. With consistency, each state is expanded at most once because $f$ is nondecreasing along paths and the best-$g$ entry is expanded first.

---

## 5. Consistency and Re-expansions

- With **consistent** $h$, each state is expanded at most once in graph-search A*.
- With **admissible but inconsistent** $h$, a state may be re-expanded when a better $g$ arrives later; closed must store best $g$ and allow improvements.

---

## Code Cell — Minimal A* in Python

```python
def astar(start, is_goal, neighbors, cost, h):
    # Node: (f, g, state, parent_index, action)
    open_heap: List[Tuple[float,float,object,int,object]] = []
    heapq.heappush(open_heap, (h(start), 0.0, start, -1, None))
    closed: Dict[object, float] = {}
    parents: List[Tuple[int, object]] = [(-1, start)]  # (parent_index, state)
    backpointers: Dict[object, Tuple[int, object]] = {start: (-1, None)}

    while open_heap:
        f, g, s, _, a = heapq.heappop(open_heap)
        if is_goal(s):
            # Reconstruct via backpointers
            path = []
            cur = s
            while True:
                p_idx, act = backpointers[cur]
                path.append((cur, act))
                if p_idx == -1:
                    break
                cur = parents[p_idx][1]
            path.reverse()
            return path, g

        if s in closed and g >= closed[s]:
            continue
        closed[s] = g

        for s2, act in neighbors(s):
            g2 = g + cost(s, act, s2)
            if s2 in closed and g2 >= closed[s2]:
                continue
            parents.append((parents.index(( -1, start)) if s == start else parents.index((backpointers[s][0], s)), s))
            backpointers[s2] = (parents.index((parents[-1][0], parents[-1][1])), act)
            heapq.heappush(open_heap, (g2 + h(s2), g2, s2, parents.index((parents[-1][0], parents[-1][1])), act))
    return None, float("inf")

# Tiny grid demo (4-connected, unit cost)
W, H = 4, 3
walls = {(1,1)}
def inb(x,y): return 0 <= x < W and 0 <= y < H and (x,y) not in walls
moves = [(1,0),(-1,0),(0,1),(0,-1)]
start, goal = (0,0), (3,2)
def is_goal(s): return s == goal
def neighbors(s):
    x,y = s
    for dx,dy in moves:
        nx,ny = x+dx, y+dy
        if inb(nx,ny): yield (nx,ny), (dx,dy)
def cost(s,a,s2): return 1.0
def h_manhattan(s): return abs(s[0]-goal[0]) + abs(s[1]-goal[1])

path, total = astar(start, is_goal, neighbors, cost, h_manhattan)
print("Path cost:", total)
print("Path:", path)
```

---

# Part III — Designing Heuristics

## 6. Admissible Heuristics via Relaxations

A common method is to relax constraints to obtain a cheaper problem whose **optimal cost** lower-bounds the original cost. If the relaxed problem’s optimal cost is $h_{\text{relax}}(s)$, then $h_{\text{relax}}(s) \le h^*(s)$, hence **admissible**.

Examples:
- **Gridworld:** Manhattan distance for 4-connected motion with unit costs is admissible.
- **Euclidean:** For continuous domains, the Euclidean distance lower-bounds any path with positive segment costs.
- **Pattern databases:** Precompute exact distances in an abstracted state space; lookups yield admissible heuristics.

---

## 7. Consistency Check

To verify consistency for $h$, check all edges $s \to s'$ that
$$
h(s) \le c(s,a) + h(s'), \quad h(g)=0.
$$
For Manhattan distance on 4-connected unit grids, this inequality holds, hence $h$ is consistent.

---

## 8. Dominance and Combining Heuristics

If $h_1(s) \le h_2(s)$ for all $s$, then $h_2$ **dominates** $h_1$ and yields fewer or equal expansions under A* (with the same tie-breaking). If $h_1,\dots,h_k$ are admissible, then
$$
h_{\max}(s)=\max_i h_i(s)
$$
is also admissible and dominates each $h_i$.

---

# Part IV — Variants & Extensions

## 9. Weighted A*

Use
$$
f(n) = g(n) + w \cdot h(n), \quad w \ge 1.
$$
For $w>1$, fewer expansions but optimality is lost; solutions are **bounded-suboptimal** for consistent $h$:
$$
g(\hat{G}) \le w \cdot C^*.
$$

## 10. Iterative Deepening A* (IDA*)

Performs depth-first searches using increasing $f$-thresholds. Memory usage is $O(d)$ while preserving optimality with admissible $h$.

## 11. Recursive Best-First Search (RBFS)

A memory-bounded best-first method that keeps the **best** and a **backup bound** for the next best alternative, unwinding recursion when the current branch exceeds the bound.

---

# Part V — Worked Examples

## 12. Hand Trace on a Grid (with Manhattan $h$)

Consider a $5\times 5$ grid, unit costs, start at $(0,0)$, goal at $(4,4)$, obstacles forming a wall with a one-cell gap. Show that A* expands nodes in bands of nondecreasing $f=g+h$, and that detours only occur when obstacles force increased $g$.

**Exercise.** Construct a case where Euclidean $h$ is **inadmissible** due to weighted moves (e.g., diagonal cost $> \sqrt{2}$), and discuss the effect.

---

## 13. Inconsistent Heuristic Example

Define $h$ that equals Manhattan everywhere except at one state where it is lowered by $2$. Show that this remains **admissible** but **inconsistent**, leading to potential re-expansions. Measure re-expansions in the Colab notebook by instrumenting a counter.

---

# Part VI — Analysis & Guarantees

## 14. Completeness, Optimality, and Time/Space

- **Completeness:** A* is complete on finite graphs with positive step-costs.
- **Optimality:** Guaranteed with **admissible** $h$ (and duplicate detection); expansions are minimized among admissible **best-first** algorithms for a given consistent $h$ up to tie-breaking.
- **Time/Space:** In worst case still exponential in solution depth $d$. Open list memory is a common bottleneck; consider IDA* or RBFS.

---

## 15. Proof Sketches

**Consistency $\Rightarrow$ nondecreasing $f$.** For any transition with cost $c$, we have
$$
f(s') = g(s')+h(s') = g(s)+c + h(s') \ge g(s) + h(s) = f(s).
$$

**A* optimality with admissible $h$.** Suppose A* selects a suboptimal goal $\hat{G}$ with cost $C'>C^*$. Some optimal path node $n^*$ remains on the frontier with $f(n^*) \le C^*$, so A* would have selected $n^*$ before $\hat{G}$—contradiction.

---

# Part VII — Practice

## 16. Exercises

1. **Heuristic dominance.** Implement $h_1$=Manhattan and $h_2$=Chebyshev (with unit diagonal cost) on an 8-connected grid with diagonals costing $1$. Verify that $h_2$ dominates $h_1$ and compare node expansions.
2. **Weighted A*.** Vary $w\in\{1.0,1.2,1.5,2.0\}$ and plot solution cost vs. expansions; verify the suboptimality bound.
3. **IDA*.** Implement IDA* on the same grid and compare memory usage and node generations to A*.
4. **Pattern database.** Build a small PDB for a sliding-tile subset and integrate $h_{\max}$ with Manhattan; measure improvement.
5. **Inconsistency impact.** Create an admissible but inconsistent $h$ and quantify re-expansions on a maze.

---

## 17. Further Reading

- S. Russell and P. Norvig, *Artificial Intelligence: A Modern Approach*, chapters on informed search and heuristics.
- R. E. Korf, “Real-time heuristic search” and “Depth-first iterative-deepening.”
- H. Felner, “Position Paper: Additive pattern database heuristics.”

---

## Appendix — Code Snippets

### A* Node Accounting (Counting Expansions)

```python
def astar_counting(start, is_goal, neighbors, cost, h):
    import heapq
    open_heap = [(h(start), 0.0, start, None)]
    closed: Dict[object,float] = {}
    expansions = 0
    while open_heap:
        f,g,s,parent = heapq.heappop(open_heap)
        if is_goal(s):
            return g, expansions
        if s in closed and g >= closed[s]:
            continue
        closed[s] = g
        expansions += 1
        for s2,a in neighbors(s):
            g2 = g + cost(s,a,s2)
            if s2 in closed and g2 >= closed[s2]:
                continue
            heapq.heappush(open_heap, (g2 + h(s2), g2, s2, s))
    return float("inf"), expansions
```

---

## Open Colab Links 

- A* (Step-by-Step): [Open in Colab](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/A_star_step_by_step.ipynb)
- A* With Graphics: [Open in Colab](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/AStar_Search_With_Graphics.ipynb)
