# Intelligent Agents and Uninformed Search: BFS, DFS, and Uniform-Cost
<!--
author:   William M. Mongan
language: en
narrator: US English Male
comment:  Render via https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-uninformedsearch.md
import:   https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md
link:     https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-5
           https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Intelligent Agents and Uninformed Search: BFS, DFS, and Uniform-Cost

How does a program *solve a problem it has never seen before*? The classical AI answer: describe the problem as a **space of states**, then **search** that space for a path to a goal. This module builds the search machinery we will later supercharge with heuristics (A*) and adversaries (minimax).

---

## 1. Intuition First: The Maze in Your Head

Imagine standing at the entrance of a corn maze:

- Every junction is a **state**.
- Every corridor you can walk down is an **action**.
- The exit is the **goal**.
- Your plan — the sequence of turns — is a **solution path**.

Different exploration *personalities* give different algorithms:

- **Cautious ring-by-ring explorer** (check everything 1 turn away, then everything 2 turns away, ...) → **Breadth-First Search (BFS)**.
- **Bold spelunker** (charge down one corridor to its end, backtrack only when stuck) → **Depth-First Search (DFS)**.
- **Thrifty accountant** (always extend the *cheapest* route found so far, when corridors have different lengths) → **Uniform-Cost Search (UCS)**.

"**Uninformed**" means none of them use a compass or a map hint — no estimate of how far the goal is. (Adding that estimate is exactly what makes A* "informed.")

---

## 2. Formalizing a Search Problem (Five Ingredients)

A search problem is a tuple:

1. **States** $S$ — configurations of the world (e.g., "agent at junction 7"). *(What situations are possible?)*
2. **Initial state** $s_0 \in S$ — where we start.
3. **Actions** $A(s)$ — what we may do in state $s$.
4. **Transition model** $\mathrm{Result}(s, a)$ — the state an action leads to, with **step cost** $c(s, a, s') \ge 0$.
5. **Goal test** — is this state a goal?

A **solution** is a path from $s_0$ to a goal; an **optimal** solution minimizes the *sum* of step costs.

**Worked micro-example — the 8-puzzle:** states are tile arrangements (there are $9!/2 = 181{,}440$ reachable ones); actions slide the blank Up/Down/Left/Right; every step costs 1. Notice we never store the whole state space — we *generate* states on demand. That is the key trick that lets search scale.

---

## 3. The Generic Search Skeleton

Every uninformed algorithm is the same loop with a different **frontier** (the set of discovered-but-unexpanded nodes):

```text
frontier  ← { start }
explored  ← {}
loop:
    if frontier is empty: return FAILURE
    node ← remove a node from frontier        # ← the ONLY line that varies!
    if node is a goal: return the path to node
    add node.state to explored
    for each child of node:
        if child.state not in explored and not in frontier:
            insert child into frontier
```

- Frontier = **FIFO queue** → BFS.
- Frontier = **LIFO stack** → DFS.
- Frontier = **priority queue ordered by path cost $g(n)$** → UCS.

---

## 4. BFS, Step by Step, with Concrete Numbers

Consider this graph (edge costs all 1); start at **A**, goal is **F**:

```text
        A
       / \
      B   C
     / \    \
    D   E    F
```

1. Frontier (queue): `[A]`. Expand **A** → enqueue B, C. *(Everything 1 step away is now discovered.)*
2. Frontier: `[B, C]`. Expand **B** → enqueue D, E.
3. Frontier: `[C, D, E]`. Expand **C** → generate **F**. Goal found!
4. Solution: `A → C → F`, length 2. We expanded just 3 nodes.

BFS expands nodes in order of **depth**, so the *first* goal it finds is a **shallowest** goal — optimal whenever all step costs are equal.

**Cost of that guarantee:** with branching factor $b$ and solution depth $d$, BFS stores $O(b^d)$ nodes. At $b = 10$, $d = 10$, that is ~10 billion nodes. Memory, not time, is usually what kills BFS.

---

## 5. DFS and the Depth-Limited Fix

DFS pops the **newest** node first. On the same graph it might explore `A, B, D, E, C, F` — finding F last! DFS:

- uses only $O(bm)$ memory ($m$ = maximum depth) — *excellent*;
- is **not optimal** and, on infinite (or looping) spaces without an explored set, **not even complete**.

Two classic repairs:

1. **Depth-limited search:** refuse to go deeper than a cutoff $\ell$.
2. **Iterative deepening (IDS):** run depth-limited search with $\ell = 0, 1, 2, \dots$ IDS gets BFS's completeness/optimality (unit costs) with DFS's memory. Re-expanding shallow nodes is cheaper than it looks: for $b=10$ the repeated work adds only about 11% more node expansions, because almost all nodes live at the deepest level.

---

## 6. Uniform-Cost Search: When Steps Aren't Equal

Now let edges have different costs. UCS always expands the frontier node with the smallest **path cost so far** $g(n)$.

**Worked micro-example** — start **S**, goal **G**:

```text
S --1--> A --1--> G
S --------3-----> G
```

1. Frontier: `{S: 0}`. Expand S → A with $g=1$, G with $g=3$.
2. Frontier: `{A: 1, G: 3}`. Expand **A** (cheapest) → G via A with $g = 1 + 1 = 2$; keep the better G entry.
3. Frontier: `{G: 2}`. Expand G → goal test passes. Answer: `S → A → G`, cost **2** — not the direct edge of cost 3!

Note the subtle rule: UCS applies the goal test **when a node is expanded**, not when it is first generated — otherwise step 1 would have returned the worse cost-3 path. UCS is optimal for any nonnegative step costs, and it is exactly **Dijkstra's algorithm** wearing an AI hat (and exactly **A\*** with heuristic $h(n) = 0$).

---

## 7. Runnable Code: All Three Searches in 40 Lines

```python
from collections import deque
import heapq

graph = {  # (neighbor, cost)
    'S': [('A', 1), ('G', 3)],
    'A': [('G', 1)],
    'G': [],
}

def bfs(start, goal):
    frontier = deque([[start]])
    explored = set()
    while frontier:
        path = frontier.popleft()          # FIFO
        node = path[-1]
        if node == goal: return path
        if node in explored: continue
        explored.add(node)
        for nbr, _ in graph[node]:
            frontier.append(path + [nbr])

def dfs(start, goal):
    frontier = [[start]]
    explored = set()
    while frontier:
        path = frontier.pop()              # LIFO — the only change!
        node = path[-1]
        if node == goal: return path
        if node in explored: continue
        explored.add(node)
        for nbr, _ in graph[node]:
            frontier.append(path + [nbr])

def ucs(start, goal):
    frontier = [(0, [start])]              # priority queue on g(n)
    explored = set()
    while frontier:
        cost, path = heapq.heappop(frontier)
        node = path[-1]
        if node == goal: return cost, path
        if node in explored: continue
        explored.add(node)
        for nbr, c in graph[node]:
            heapq.heappush(frontier, (cost + c, path + [nbr]))

print("BFS:", bfs('S', 'G'))   # fewest edges: S->G
print("DFS:", dfs('S', 'G'))
print("UCS:", ucs('S', 'G'))   # cheapest: (2, S->A->G)
```

**What to look for:** BFS returns the *fewest-edges* path `S→G`, while UCS returns the *cheapest* path `S→A→G`. Same skeleton, different frontier, different guarantee.

---

## 8. Comparing the Algorithms (UDL Summary Table)

| Property | BFS | DFS | IDS | UCS |
|---|---|---|---|---|
| Frontier | FIFO queue | LIFO stack | DFS with growing limit | Priority queue on $g$ |
| Complete? | Yes (finite $b$) | No (infinite depth) | Yes | Yes (costs $\ge \epsilon > 0$) |
| Optimal? | Yes, if unit costs | No | Yes, if unit costs | **Yes** |
| Time | $O(b^d)$ | $O(b^m)$ | $O(b^d)$ | $O(b^{1 + \lfloor C^*/\epsilon \rfloor})$ |
| Memory | $O(b^d)$ 💥 | $O(bm)$ ✅ | $O(bd)$ ✅ | $O(b^{1 + \lfloor C^*/\epsilon \rfloor})$ |

($d$ = shallowest goal depth, $m$ = max depth, $C^*$ = optimal cost, $\epsilon$ = minimum step cost.)

---

## 9. Looking Ahead

- Add a **heuristic estimate** $h(n)$ of remaining cost and order the frontier by $g(n) + h(n)$: you get **A\*** (see the *Informed Search* module).
- Replace "reach a goal state" with "outsmart an opponent": **minimax** (see the *Heuristic Search* module).
- Replace "find a path" with "find a good configuration": **local search and genetic algorithms** (next module!).

---

## Comprehension Quiz

[[MC]]
You are searching a space where every action costs 1 and memory is your binding constraint. Which algorithm gives an optimal solution with the least memory?
- ( ) Breadth-first search.
- ( ) Depth-first search without a depth limit.
- (x) Iterative deepening search.
- ( ) Uniform-cost search.

[[MC]]
In the UCS worked example, why is the goal test applied when a node is *expanded* rather than when it is *generated*?
- (x) A goal generated early may still be reachable by a cheaper path not yet discovered.
- ( ) It makes the algorithm run faster.
- ( ) Generated nodes have no path cost yet.
- ( ) It is a historical convention with no correctness consequence.

---

## Discussion Prompt

> Web crawlers, social-network "friend of a friend" features, and puzzle solvers all use variants of these searches. Pick one and identify its states, actions, and why its designers chose BFS-like or DFS-like exploration.
