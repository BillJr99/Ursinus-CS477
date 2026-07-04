# What Is Artificial Intelligence? Course Introduction and Tools
<!--
author:   William M. Mongan
language: en
narrator: US English Male
comment:  Render via https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-introai.md
import:   https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md
link:     https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-5
           https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# What Is Artificial Intelligence? Course Introduction and Tools

Welcome to CS477! Before we write a single algorithm, let's build an intuition for what "intelligence" means to a computer scientist, see where AI came from, and set up the tools (Python, Jupyter, and Colab) that we will use all semester.

---

## 1. Intuition First: What Makes Something "Intelligent"?

Think about three everyday systems:

- A **thermostat** turns the heat on when the room is cold.
- A **GPS app** finds the fastest route across town, even during rush hour.
- A **spam filter** decides which emails you never see.

All three *sense* something about the world, *decide* something, and *act*. The differences are in **how much they must reason** and **how much they must learn**:

| System | Senses | Decides by... | Learns? |
|---|---|---|---|
| Thermostat | Temperature | A fixed rule ("if temp < 68, heat on") | No |
| GPS routing | Map + traffic | **Searching** many possible routes | Not usually |
| Spam filter | Email text | **Statistics learned from examples** | Yes |

This table previews the two great traditions of AI that structure this course:

1. **Knowledge-based / search-based AI** (the first half of the arc): we tell the machine *how the world works* (states, actions, rules, probabilities) and it **reasons** — searching, planning, and inferring.
2. **Data-driven machine learning** (the second half): we give the machine *examples* and it **learns** the rules itself — regression, neural networks, and beyond.

> **Checkpoint intuition:** A chess program that looks ahead ten moves is doing *search*. A program that predicts your next word by having read billions of sentences is doing *learning*. Modern systems (like game-playing agents and chatbots) combine both.

---

## 2. A Two-Minute History in Four Waves

1. **1950s–1960s: The founding.** Alan Turing asks "Can machines think?" and proposes the **imitation game** (the Turing Test). The 1956 Dartmouth workshop names the field "artificial intelligence."
2. **1960s–1980s: Symbolic AI and expert systems.** Intelligence as logic: represent knowledge with symbols and rules, and reason by deduction. Successes in narrow domains; brittleness in the messy real world leads to "AI winters."
3. **1990s–2010s: Probabilistic and statistical AI.** Uncertainty becomes first-class: Bayesian inference, hidden Markov models, and support vector machines. Deep Blue defeats Kasparov (1997) — a triumph of *search* plus *evaluation heuristics*.
4. **2012–today: Deep learning and generative AI.** Neural networks trained on massive data dominate vision, speech, and language. AlphaGo (2016) fuses deep learning *and* search; large language models bring AI into daily writing, coding, and reasoning tasks.

Every wave is still with us. In this course you will implement ideas from **all four**.

---

## 3. Defining AI: Four Classic Perspectives

Textbooks organize definitions of AI along two axes — *thinking vs. acting* and *humanly vs. rationally*:

| | **Humanly** | **Rationally** |
|---|---|---|
| **Thinking** | Cognitive modeling: simulate human thought | "Laws of thought": correct logical inference |
| **Acting** | The Turing Test: behave indistinguishably from a person | **Rational agents: act to best achieve goals** |

We adopt the **rational agent** view: an **agent** perceives its **environment** through **sensors** and acts on it through **actuators**, choosing actions that maximize its expected **performance measure**.

Numbered micro-example — a vacuum-cleaner agent in a two-room world:

1. **Percepts:** which room it is in, and whether that room is dirty. *(What can it see?)*
2. **Actions:** `Left`, `Right`, `Suck`. *(What can it do?)*
3. **Performance measure:** +1 point per clean room per time step. *(What counts as doing well?)*
4. **A rational rule:** if dirty → `Suck`; else move to the other room. *(Given what it knows, no other rule earns more points.)*

Notice that rationality is **not** omniscience: the agent does the best it can *given its percepts*, not with perfect knowledge of the future. This idea returns when we study search (acting with a model), probability (acting under uncertainty), and reinforcement learning (acting while learning).

---

## 4. The Course Roadmap (Where We Are Going)

- **Weeks 1–2:** Agents and **state-space search** — solving problems by exploring possibilities.
- **Weeks 3–4:** **Optimization and genetic algorithms**, then **knowledge representation and logic**.
- **Weeks 5–6:** **Probability** and **data/feature thinking** — the bridge into machine learning, ending with **decision trees**.
- **Weeks 7–10:** **Machine learning core**: regression, clustering, model evaluation, SVMs, neural networks, PCA.
- **Weeks 10–13:** **Search and probabilistic reasoning, revisited with power tools**: minimax, A*, Bayesian inference, HMMs, reinforcement learning.
- **Weeks 12–15:** **Frontier topics**: GANs, explainable and responsible AI, and a wrap-up tour from sequence models to transformers and LLMs.

---

## 5. Your Toolkit: Python, Jupyter, and Colab

All of our activities use **Python 3** with **NumPy** and **Matplotlib**, delivered as **Jupyter notebooks** that you can run in the cloud via **Google Colab** — no installation required.

- A **notebook** is a sequence of **cells**: text cells (like this one) and code cells you can execute and modify.
- Look for the **Open In Colab** badges throughout our modules — click one and you are running real course code in your browser.
- Workflow habit: **read a cell → predict what it will print → run it → reconcile.** Prediction-before-execution is the fastest way to learn.

Try it right here — this page's code cells are runnable too:

```python
import math

# 1. Variables and f-strings
course = "CS477"
print(f"Welcome to {course}!")

# 2. Lists and loops: score a tiny "agent" that cleans 3 rooms
rooms = ["dirty", "clean", "dirty"]
score = sum(1 for r in rooms if r == "dirty")  # rooms it can clean
print("Rooms the agent can clean:", score)

# 3. Functions: Euclidean distance, our first "similarity" measure
def distance(p, q):
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p, q)))

print("Distance from (0,0) to (3,4):", distance((0, 0), (3, 4)))
```

**What to look for:** the third snippet computes $\sqrt{(3-0)^2 + (4-0)^2} = \sqrt{9+16} = \sqrt{25} = 5$. Distances like this quietly power much of machine learning (nearest neighbors, clustering, and more).

---

## 6. NumPy in Three Moves

NumPy gives us fast arrays — the data structure underneath every model we will build.

```python
import numpy as np

# Move 1: make arrays
x = np.array([1.0, 2.0, 3.0])
w = np.array([0.5, -1.0, 2.0])

# Move 2: elementwise math (no loops!)
print("x * w =", x * w)

# Move 3: the dot product — the single most important operation in ML
print("x . w =", x @ w)   # 1*0.5 + 2*(-1) + 3*2 = 4.5
```

Step by step, that dot product is:

1. Multiply pairwise: $1 \times 0.5 = 0.5$, $\;2 \times (-1) = -2$, $\;3 \times 2 = 6$. *(Each feature times its weight.)*
2. Add them up: $0.5 - 2 + 6 = 4.5$. *(One number summarizing the whole vector.)*

When we reach regression and neural networks, "make a prediction" will literally mean "take a dot product (plus a bias), maybe squash it." You already know the core operation.

---

## 7. Ethics from Day One

AI systems increasingly make or shape decisions about **credit, hiring, healthcare, policing, and information access**. Throughout the course we pair each technique with its responsibilities:

- Search and optimization: *what objective are we optimizing, and who chose it?*
- Probabilistic inference: *what happens when base rates differ across groups?*
- Machine learning: *where did the data come from, and whom does it represent?*
- Generative AI: *what are the costs — attribution, labor, energy — of scale?*

We return to these questions formally in the **Explainable and Responsible AI** module, but you should raise them in *every* module.

---

## Comprehension Quiz

[[MC]]
Under the rational-agent definition, an agent is intelligent when it...
- ( ) perfectly imitates human behavior.
- (x) selects actions expected to maximize its performance measure, given its percepts.
- ( ) always finds the true optimal action, using complete knowledge of the world.
- ( ) uses a neural network.

[[MC]]
Which pairing correctly matches a system to its dominant AI tradition?
- ( ) Spam filter → state-space search; GPS routing → learning from examples.
- (x) GPS routing → state-space search; spam filter → learning from examples.
- ( ) Thermostat → deep learning; GPS routing → expert systems.
- ( ) Spam filter → logic-based deduction; thermostat → reinforcement learning.

---

## Discussion Prompt

> Pick an AI system you used this week (autocomplete, recommendations, navigation, a chatbot). What are its percepts, actions, and performance measure — and who benefits when it "performs well"?

---

## Further Exploration

- Skim the opening chapter of *Machine Learning Systems* (see the syllabus readings) for the systems-engineering view of AI: models are a small part of a much larger pipeline.
- Browse Chris Tralie's Ursinus CS 477 materials at [https://ursinusai.github.io/F2025/](https://ursinusai.github.io/F2025/) and his [lecture video playlist](https://youtube.com/playlist?list=PLxGzv4uunL66am0572y8wLxVwQY67wZL_) for an alternate voice on the same topics.
