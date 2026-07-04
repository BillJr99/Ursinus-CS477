# Knowledge Representation and Logical Reasoning
<!--
author:   William M. Mongan
language: en
narrator: US English Male
comment:  Render via https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-knowledgelogic.md
import:   https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md
link:     https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-5
           https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Knowledge Representation and Logical Reasoning

Search finds solutions by *trying things*. A **knowledge-based agent** finds them by *thinking*: it stores facts and rules in a **knowledge base (KB)** and derives new conclusions by **inference**. This week we build that machinery — propositional logic, inference rules, and a taste of first-order logic and expert systems — and see both its power and the brittleness that pushed AI toward probability and learning.

---

## 1. Intuition First: The Locked-Room Detective

A detective reasons: "The window was locked from the inside. If the window was locked from the inside, the intruder used the door. Therefore the intruder used the door." No trial and error, no search — just **combining known facts with rules** to obtain a fact nobody directly observed.

That inference pattern has a 2,000-year-old name, **modus ponens**:

$$
\frac{P, \quad P \Rightarrow Q}{Q}
$$

read: "from $P$ and $P \Rightarrow Q$, conclude $Q$." A knowledge-based agent is a program that does this at scale, thousands of times per second.

---

## 2. Propositional Logic: Syntax and Semantics

**Syntax** — sentences are built from proposition symbols ($P$, $Q$, `Rain`, `WetGrass`, ...) and connectives:

| Connective | Symbol | Reading | True when... |
|---|---|---|---|
| Negation | $\lnot P$ | "not P" | $P$ is false |
| Conjunction | $P \land Q$ | "P and Q" | both true |
| Disjunction | $P \lor Q$ | "P or Q" | at least one true |
| Implication | $P \Rightarrow Q$ | "if P then Q" | **always, except when $P$ true and $Q$ false** |
| Biconditional | $P \Leftrightarrow Q$ | "P iff Q" | both same |

**Semantics** — a **model** (or "world") assigns true/false to every symbol. A sentence is **satisfiable** if *some* model makes it true, and **valid** if *every* model does.

**The big definition — entailment:**

$$
KB \models \alpha \quad \iff \quad \text{every model that makes } KB \text{ true also makes } \alpha \text{ true.}
$$

Plain language: $\alpha$ is *guaranteed* by what we know — there is no possible world consistent with our knowledge where $\alpha$ fails.

---

## 3. Checking Entailment by Truth Table (Fully Worked)

**KB:** (1) $R \Rightarrow W$ ("if it rained, the grass is wet"), (2) $R$ ("it rained"). **Query:** $W$?

Enumerate all $2^2 = 4$ worlds:

| $R$ | $W$ | $R \Rightarrow W$ | KB true? | $W$ true? |
|---|---|---|---|---|
| T | T | T | **✔ (both facts hold)** | ✔ |
| T | F | F | ✘ | – |
| F | T | T | ✘ ($R$ fails) | – |
| F | F | T | ✘ ($R$ fails) | – |

Exactly one world satisfies the KB, and $W$ holds there → $KB \models W$. ✅

Numbered takeaways:

1. Entailment is **model checking**: look only at rows where the KB is true. *(Worlds ruled out by our knowledge are irrelevant.)*
2. The table has $2^n$ rows for $n$ symbols. *(Sound and complete, but exponential — the price of brute force.)*
3. Practical systems avoid full tables using inference rules, resolution, or fast **SAT solvers** — the same technology that verifies hardware and schedules airlines today.

---

## 4. Inference Rules and Forward Chaining

For KBs of **Horn clauses** (rules with at most one positive conclusion, like $A \land B \Rightarrow C$), inference is efficient: repeatedly apply modus ponens until nothing new appears — **forward chaining**. (Its mirror image, **backward chaining**, starts from the query and works backwards — you have met this idea as recursion.)

**Worked example.** KB: facts $A$, $B$; rules $A \land B \Rightarrow C$, $\;C \Rightarrow D$, $\;D \land B \Rightarrow E$.

1. Known: $\{A, B\}$. Rule 1 fires → add $C$.
2. Known: $\{A, B, C\}$. Rule 2 fires → add $D$.
3. Known: $\{A, B, C, D\}$. Rule 3 fires → add $E$. Fixpoint reached.

```python
facts = {"A", "B"}
rules = [({"A", "B"}, "C"),
         ({"C"}, "D"),
         ({"D", "B"}, "E")]

changed = True
while changed:
    changed = False
    for premises, conclusion in rules:
        if premises <= facts and conclusion not in facts:
            print(f"{' AND '.join(sorted(premises))} => {conclusion}")
            facts.add(conclusion)
            changed = True
print("Derived knowledge base:", sorted(facts))
```

**What to look for:** the loop is *data-driven* — each pass scans the rules and fires any whose premises are all known. This is exactly the engine inside classic **expert systems**.

---

## 5. First-Order Logic in One Slide

Propositional logic cannot say "*all* humans are mortal" without a symbol per human. **First-order logic (FOL)** adds objects, relations, and quantifiers:

- $\forall x\; \mathrm{Human}(x) \Rightarrow \mathrm{Mortal}(x)$ — "for all $x$..."
- $\exists x\; \mathrm{Enrolled}(x, \mathrm{CS477}) \land \mathrm{Loves}(x, \mathrm{Logic})$ — "there exists..."

The classic syllogism becomes mechanical:

1. $\forall x\; \mathrm{Human}(x) \Rightarrow \mathrm{Mortal}(x)$ *(rule)*
2. $\mathrm{Human}(\mathrm{Socrates})$ *(fact)*
3. **Instantiate** the rule with $x = \mathrm{Socrates}$, then apply modus ponens → $\mathrm{Mortal}(\mathrm{Socrates})$. ∎

FOL powers logic programming (Prolog), semantic-web ontologies, and database query languages — SQL's `WHERE` clause is a fragment of it.

---

## 6. Expert Systems: Triumph and Brittleness

1970s–80s **expert systems** (MYCIN for infections, DENDRAL for chemistry, XCON for configuring computers) encoded hundreds of `IF–THEN` rules from human specialists and, within their domains, matched expert performance.

Why they faded — and why it matters for the rest of this course:

| Weakness | The fix (later in this course) |
|---|---|
| Certainty required; real evidence is noisy | **Probability & Bayesian inference** (weeks 5, 11–12) |
| Rules hand-written; knowledge acquisition bottleneck | **Learning rules/models from data** (weeks 6–10) |
| No graceful degradation outside the rule set | Statistical generalization; ensembles |

A modern synthesis: today's systems often pair a learned model with logical **constraints** (e.g., "an insulin dose must never exceed X") — knowledge representation is not dead, it moved into the guardrails.

---

## 7. Ethics Checkpoint: Whose Knowledge?

A rule base is a frozen viewpoint. MYCIN encoded 1970s Stanford treatment norms; a loan-approval rule set encodes its authors' assumptions about "creditworthiness." Two questions to carry forward:

1. **Provenance:** who wrote the rules (or labeled the data), and what did they not know?
2. **Contestability:** an `IF–THEN` rule can at least be *inspected and appealed* — a property we will fight to recover when we reach neural networks and **explainable AI**.

---

## Comprehension Quiz

[[MC]]
$KB \models \alpha$ means:
- ( ) $\alpha$ can be proven in fewer than $2^n$ steps.
- (x) every possible world satisfying the KB also satisfies $\alpha$.
- ( ) $\alpha$ is true in at least one world satisfying the KB.
- ( ) the KB contains $\alpha$ verbatim.

[[MC]]
In the forward-chaining example, why can the algorithm safely stop when a full pass fires no new rules?
- (x) The known-fact set only grows, so a pass with no additions means a fixpoint: no future pass can differ.
- ( ) All queries have been answered.
- ( ) Horn clauses expire after one use.
- ( ) The rules are removed as they fire.

---

## Discussion Prompt

> Choose a policy you know well (a course syllabus rule, a return policy, a dorm policy). Write it as 3–5 formal rules, then find an edge case where the formalization and the *intent* disagree. What does this teach you about deploying rule-based AI?
