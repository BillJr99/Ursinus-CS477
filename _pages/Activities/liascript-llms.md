# From Sequence Models to Transformers, LLMs, and Retrieval-Augmented Generation
<!--
author:   William M. Mongan
language: en
narrator: US English Male
comment:  Render via https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-llms.md
import:   https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md
link:     https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-5
           https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# From Sequence Models to Transformers, LLMs, and Retrieval-Augmented Generation

Our course finale connects everything you have built — dot products, gradients, backpropagation, softmax, probability, even reinforcement learning — into the systems dominating AI today: **large language models (LLMs)**. The remarkable claim of this module: *you already know all the parts.* We will assemble them, in six steps, from "words as numbers" to a chatbot grounded in your own documents.

**The staircase we will climb:**

1. Words → vectors (**embeddings / word2vec**)
2. Vectors in order (**RNNs**, from the Neural Networks module)
3. Sequence in, sequence out (**seq2seq encoder–decoder**)
4. Look where it matters (**attention**)
5. Attention is all you need (**transformers**, GPT & BERT)
6. Make it useful and grounded (**RLHF** and **RAG**)

---

## Step 1 — Word Embeddings: Words as Vectors

A neural network eats numbers, not words. The naive fix — one-hot vectors — makes every pair of words equally distant ("cat" is as far from "kitten" as from "carburetor"). **Word embeddings** instead learn a dense vector per word such that *words used similarly get nearby vectors*.

**word2vec's trick, in micro-steps:**

1. Set up a tiny network: input = a word, output = a softmax over the vocabulary predicting its *neighbors* in real sentences. *(A fill-in-the-blank game.)*
2. Train with backpropagation on billions of sentences. *(Exactly the machinery from our neural networks module.)*
3. Throw the network away and **keep the learned weights** — each word's weight row *is* its embedding.
4. Geometry appears for free: famously, $\mathrm{vec}(\text{king}) - \mathrm{vec}(\text{man}) + \mathrm{vec}(\text{woman}) \approx \mathrm{vec}(\text{queen})$.

[![Word Embedding and Word2Vec, Clearly Explained!!! (StatQuest)](https://img.youtube.com/vi/viZrOnJclY0/0.jpg)](https://www.youtube.com/watch?v=viZrOnJclY0)

*Video summary (text equivalent):* StatQuest shows how a small neural network trained to predict neighboring words ends up assigning each word a vector of weights, why similar words receive similar vectors, and how negative sampling makes training tractable.

```python
import numpy as np

# Toy 4-word embedding space (2 dimensions for drawing; real ones use 300-12,000)
emb = {
    "king":  np.array([0.9, 0.8]),
    "queen": np.array([0.9, 0.2]),
    "man":   np.array([0.1, 0.8]),
    "woman": np.array([0.1, 0.2]),
}
analogy = emb["king"] - emb["man"] + emb["woman"]
sims = {w: float(v @ analogy / (np.linalg.norm(v) * np.linalg.norm(analogy)))
        for w, v in emb.items()}
print("king - man + woman is closest to:",
      max(sims, key=sims.get), " (cosine sims:", {k: round(s,2) for k,s in sims.items()}, ")")
```

---

## Step 2 — Recap: RNNs Read in Order

The **Neural Networks** module ended with RNNs and LSTMs: networks that carry a hidden state $h_t = f(W_x x_t + W_h h_{t-1} + b)$ from word to word, giving them memory. Their weakness: information must squeeze through one fixed-size state, so by word 100 the network has largely forgotten word 3 — and processing is inherently sequential (slow to train).

Hold that weakness in mind; steps 3–5 are the escape route.

---

## Step 3 — Seq2seq: An Encoder Talks to a Decoder

Translation maps a sequence to a *different-length* sequence. The **encoder–decoder** answer:

1. **Encoder** RNN reads the source sentence and compresses it into a final state — the "thought vector."
2. **Decoder** RNN starts from that state and emits the target sentence one token at a time, feeding each output back in as the next input.
3. Training maximizes the probability of the correct next token at every step — chain rule of probability, met in week 5:
   $$P(y_1,\dots,y_m \mid x) = \prod_{t=1}^{m} P(y_t \mid y_{<t}, x).$$

[![Sequence-to-Sequence (seq2seq) Encoder-Decoder Neural Networks, Clearly Explained!!! (StatQuest)](https://img.youtube.com/vi/L8HKweZIOmg/0.jpg)](https://www.youtube.com/watch?v=L8HKweZIOmg)

*Video summary (text equivalent):* StatQuest walks through translating English to Spanish with an LSTM encoder that summarizes the input and an LSTM decoder that unrolls the translation, highlighting the fixed-size bottleneck between them.

**The bottleneck problem:** an entire paragraph must fit into one vector. Long inputs degrade badly — motivating attention.

---

## Step 4 — Attention: Look Where It Matters

Instead of one summary vector, let the decoder *look back at every encoder position* each time it emits a word, weighting them by relevance:

1. **Score:** compare the decoder's current state (a **query** $q$) with each encoder state (**keys** $k_i$) — typically a dot product $s_i = q \cdot k_i$. *(Similarity via dot product — week 1's operation!)*
2. **Normalize:** $\alpha_i = \mathrm{softmax}(s_i)$, so the weights are positive and sum to 1. *(Softmax — from logistic regression and neural nets.)*
3. **Blend:** context $= \sum_i \alpha_i v_i$, a weighted average of the encoder **values**. *(Expectation — week 5.)*

Compactly, for matrices $Q, K, V$:

$$
\mathrm{Attention}(Q, K, V) = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}}\right) V
$$

The $\sqrt{d_k}$ merely keeps dot products from saturating the softmax.

[![Attention for Neural Networks, Clearly Explained!!! (StatQuest)](https://img.youtube.com/vi/PSs6nxngL6k/0.jpg)](https://www.youtube.com/watch?v=PSs6nxngL6k)

*Video summary (text equivalent):* StatQuest adds attention to the seq2seq model, showing how similarity scores between decoder and encoder states become softmax weights that let each output word draw directly on the most relevant input words, fixing the long-sentence bottleneck.

```python
import numpy as np

def softmax(x): e = np.exp(x - x.max()); return e / e.sum()

# 3 encoder positions ("the", "black", "cat"), d=4
K = V = np.array([[1., 0., 0., 1.],
                  [0., 1., 0., 1.],
                  [0., 0., 1., 1.]])
q = np.array([0.1, 0.2, 2.0, 1.0])       # decoder is about to say "gato" -> should attend to "cat"

scores  = K @ q / np.sqrt(4)
weights = softmax(scores)
print("attention weights over ['the','black','cat']:", np.round(weights, 2))
print("context vector:", np.round(weights @ V, 2))
```

**What to look for:** the query most resembles the third key, so ~70% of the attention lands on "cat" — the mechanism *learned nothing yet*, this is pure geometry. Training shapes the queries/keys so the right geometry emerges.

---

## Step 5 — Transformers: Attention Is All You Need

The 2017 transformer removes the RNN entirely and keeps only attention:

1. **Self-attention:** every token queries every other token in the same sentence — in parallel, no sequential bottleneck. Order is preserved by adding **positional encodings** to the embeddings.
2. **Multi-head:** run several attentions at once so one head can track syntax while another tracks coreference.
3. **Stack:** attention + feed-forward layers + residual connections + layer norm, repeated dozens of times.
4. **Scale:** because everything is matrix multiplication, GPUs devour it — enabling billions of parameters and the modern LLM era.

[![Transformer Neural Networks, ChatGPT's foundation, Clearly Explained!!! (StatQuest)](https://img.youtube.com/vi/zxQyTK8quyY/0.jpg)](https://www.youtube.com/watch?v=zxQyTK8quyY)

*Video summary (text equivalent):* StatQuest assembles a miniature transformer for English→Spanish translation from word embeddings, positional encoding, self-attention, and residual connections — each component reduced to arithmetic you can check by hand.

**Two body plans, one skeleton:**

| | **Decoder-only (GPT family)** | **Encoder-only (BERT family)** |
|---|---|---|
| Attention direction | causal — each token sees only the past | bidirectional — sees whole input |
| Pretraining game | predict the *next* token | fill in *masked* tokens |
| Superpower | generating text | understanding/embedding text |
| Role in this module | the chatbot | the retriever (Step 6b) |

[![Encoder-Only Transformers (like BERT) for RAG, Clearly Explained!!! (StatQuest)](https://img.youtube.com/vi/GDN649X_acE/0.jpg)](https://www.youtube.com/watch?v=GDN649X_acE)

*Video summary (text equivalent):* StatQuest explains how BERT-style models read a whole passage at once to produce context-aware embeddings, and why that makes them the natural engine for finding relevant documents in retrieval-augmented generation.

---

## Step 6a — RLHF: Teaching the Model to Be Helpful

A pretrained LLM only imitates its training text. **Reinforcement Learning from Human Feedback** aligns it with what users actually want:

1. **Supervised fine-tuning:** humans write good responses; the model imitates them.
2. **Reward model:** humans *rank* pairs of model responses; a second network learns to predict those preferences — a learned performance measure (week 1's vocabulary!).
3. **RL step:** the LLM becomes the *agent*, a response is an *action*, the reward model provides *reward*, and a policy-gradient method (PPO) updates the LLM — reinforcement learning (week 13) applied to language.

[![Reinforcement Learning with Human Feedback (RLHF), Clearly Explained!!! (StatQuest)](https://img.youtube.com/vi/qPN_XZcJf_s/0.jpg)](https://www.youtube.com/watch?v=qPN_XZcJf_s)

*Video summary (text equivalent):* StatQuest traces the three RLHF stages — supervised fine-tuning, preference-based reward modeling, and policy optimization — showing how chatbots like ChatGPT are steered from raw next-word predictors into helpful assistants.

---

## Step 6b — RAG: Grounding the Model in Your Documents

LLMs **hallucinate**: they generate plausible text, not verified facts, and they know nothing after their training cutoff or inside your private files. **Retrieval-Augmented Generation (RAG)** fixes both without retraining:

1. **Index:** split your documents into chunks; embed each chunk (BERT-style encoder, Step 5); store the vectors in a **vector database**.
2. **Retrieve:** embed the user's question; fetch the $k$ nearest chunks (cosine similarity — dot products again).
3. **Augment:** paste those chunks into the prompt as context.
4. **Generate:** the LLM answers *from the supplied context*, citably.

Two hands-on paths, one visual and one code-first — both make excellent final-project extensions:

[![Build a RAG Based LLM App in 20 Minutes! | Full Langflow Tutorial](https://img.youtube.com/vi/rz40ukZ3krQ/0.jpg)](https://www.youtube.com/watch?v=rz40ukZ3krQ)

*Video summary (text equivalent):* a drag-and-drop walkthrough of Langflow that wires document loading, chunking, embedding, a vector store, and an LLM into a working RAG chatbot with no code — useful for seeing the RAG dataflow as a literal diagram.

[![How to Build a Local AI Agent With Python (Ollama, LangChain & RAG)](https://img.youtube.com/vi/E4l91XKQSgw/0.jpg)](https://www.youtube.com/watch?v=E4l91XKQSgw)

*Video summary (text equivalent):* builds the same pipeline in Python — running an open-weights model locally with Ollama, embedding documents, retrieving with LangChain, and answering questions over private data entirely on your own machine (no API keys, no data leaving your laptop).

---

## The Whole Course in One Table

| LLM component | Where you learned the underlying idea |
|---|---|
| Embedding lookup, $QK^\top$ scores | Dot products (Intro), linear layers (Neural Nets) |
| Softmax over tokens | Logistic regression, Neural Nets |
| $P(y_t \mid y_{<t})$ factorization | Chain rule of probability (Prob. Foundations) |
| Training by gradient descent + backprop | Regression, Neural Nets |
| Attention as weighted expectation | Expectation (Prob. Foundations) |
| RLHF | Reinforcement Learning |
| Retrieval by nearest vectors | Distances & k-NN (Unsupervised Learning) |
| "Should we deploy it?" | Explainable & Responsible AI |

---

## Responsible Deployment Checklist

- **Hallucination:** generation is sampling from $P(\text{next token})$, not lookup — require RAG grounding or human review for factual claims.
- **Provenance & labor:** pretraining corpora and RLHF rankings embody the choices (and working conditions) of the people who made them.
- **Cost:** training frontier models consumes megawatt-hours; retrieval + small models often beat brute scale for domain tasks (see the *Machine Learning Systems* readings on efficient AI and ML operations).
- **Privacy:** the local-agent path above exists precisely so sensitive documents never leave your machine.

---

## Comprehension Quiz

[[MC]]
Why did attention displace the seq2seq bottleneck design?
- ( ) Attention has fewer parameters.
- (x) It lets each output step draw on *all* encoder positions via learned softmax weights, instead of one fixed-size summary vector.
- ( ) It removes the need for word embeddings.
- ( ) It guarantees factual outputs.

[[MC]]
In RLHF, the "reward" that fine-tunes the LLM comes from:
- ( ) the cross-entropy loss on next-token prediction.
- ( ) a rule-based grammar checker.
- (x) a learned model trained on human preference rankings of candidate responses.
- ( ) the vector database's similarity scores.

[[MC]]
A RAG system answers your question incorrectly, citing a retrieved chunk that is off-topic. Which staircase step failed?
- ( ) Step 1: the embeddings do not exist.
- (x) Step 6b's retrieval: the question and the relevant chunk were not near neighbors in embedding space.
- ( ) Step 4: softmax normalization.
- ( ) Step 6a: the reward model.

---

## Discussion Prompt

> You now know the full pipeline: pretraining, RLHF, retrieval. When an LLM asserts something false about a living person, which stage(s) bear responsibility — and which of the interventions from this module (better rewards, grounding, restriction to retrieval) would you require before deployment?

---

## Where to Go Next

- *Machine Learning for Engineers* Chapter 8 (Transformers) for the full mathematical treatment.
- The *Machine Learning Systems* book (syllabus readings) for serving, scaling, and monitoring these models in production.
- Chris Tralie's [Ursinus AI course materials](https://ursinusai.github.io/F2025/) and [lecture playlist](https://youtube.com/playlist?list=PLxGzv4uunL66am0572y8wLxVwQY67wZL_) for a complementary path through the same territory.
