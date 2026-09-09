---
layout: assignment
permalink: Assignments/GAN
title: "Assignment: Generative Adversarial Networks (GAN) — From Scratch and with Libraries"

info:
  points: 100
  goals:
    - Understand the adversarial training framework of GANs and their min–max objective.
    - Implement a simple GAN from scratch using PyTorch or TensorFlow for a 1D or 2D toy dataset.
    - Use a library-based GAN (DCGAN) on image data and evaluate generated samples.
    - Design and execute a creative GAN experiment for a novel dataset.
  purpose: "This assignment builds deep intuition for the adversarial game between generator and discriminator networks. You will implement GANs from scratch, explore convergence challenges, and apply them to creative data generation tasks."
  concepts:
    - Min–max optimization and adversarial dynamics
    - Generator and discriminator architectures
    - Mode collapse and convergence
    - Evaluation of generative quality (FID, visual inspection)
  tasks:
    - Implement a GAN from scratch for a toy dataset.
    - Train a DCGAN on MNIST or similar data using a deep learning library.
    - Design a creative generative task using the GAN framework.
  rubric:
    - weight: 30
      description: Implementation
      preemerging: Partial or unstable training loop.
      beginning: Functional GAN with visible learning progression.
      progressing: Well-structured code, convergence plots, and saved samples.
      proficient: Stable training, reproducible results, insightful commentary.
    - weight: 30
      description: Conceptual Understanding
      preemerging: Minimal discussion of losses or dynamics.
      beginning: Correct objective and update equations.
      progressing: Discusses stability, gradients, and trade-offs.
      proficient: Deep interpretation of loss surfaces, failure modes, and evaluation.
    - weight: 20
      description: Analysis and Creativity
      preemerging: Limited exploration.
      beginning: Visual evaluation of samples.
      progressing: Experimentation with architectures or hyperparameters.
      proficient: Creative adaptation of GANs to new contexts or data.
    - weight: 10
      description: Code Quality & Documentation
      preemerging: Sparse comments or poor organization.
      beginning: Basic comments and modular code.
      progressing: Structured, readable, with intermediate logs and plots.
      proficient: Excellent organization, reproducibility, and interpretability.
    - weight: 10
      description: Submission Completeness
      preemerging: Missing sections or results.
      beginning: Includes code and outputs.
      progressing: Includes clear report and reproducible results.
      proficient: Complete notebook, report, and discussion.

tags:
  - deep-learning
  - generative-models
  - adversarial-learning
  - neural-networks
---

# Overview

Generative Adversarial Networks (GANs) train two models — a **generator** $G$ and a **discriminator** $D$ — in a minimax game:
$$
\min_G \max_D V(D,G) = \mathbb{E}_{x\sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z\sim p_z}[\log(1 - D(G(z)))].
$$
The generator learns to produce samples that fool the discriminator, while the discriminator learns to distinguish real from fake.

---

## Stage 0 — Setup

Use PyTorch (preferred) or TensorFlow.

```python
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
torch.manual_seed(0)
```

---

## Stage 1 — GAN from Scratch on a Toy Dataset

We create a **1D Gaussian mixture** dataset and train a small fully-connected GAN to approximate it.

### 1.1 Data

```python
# Mixture of Gaussians
n = 1000
x_real = np.concatenate([np.random.normal(-2, 0.3, n//2),
                         np.random.normal(2, 0.3, n//2)])
x_real = x_real.reshape(-1, 1).astype(np.float32)
real_data = torch.tensor(x_real)
loader = DataLoader(TensorDataset(real_data), batch_size=64, shuffle=True)

def plot_samples(samples, title="Samples"):
    plt.hist(samples.detach().numpy(), bins=50, alpha=0.7)
    plt.title(title); plt.show()
```

### 1.2 Model Definitions

```python
class Generator(nn.Module):
    def __init__(self, z_dim=1, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(z_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, hidden=16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)
```

### 1.3 Training Loop

```python
z_dim = 1
G, D = Generator(z_dim), Discriminator()
opt_G = optim.Adam(G.parameters(), lr=1e-3)
opt_D = optim.Adam(D.parameters(), lr=1e-3)
criterion = nn.BCELoss()

for epoch in range(2000):
    for (x_batch,) in loader:
        bs = x_batch.size(0)
        # Real and fake labels
        y_real = torch.ones(bs, 1)
        y_fake = torch.zeros(bs, 1)
        # Train Discriminator
        z = torch.randn(bs, z_dim)
        x_fake = G(z).detach()
        D_loss = criterion(D(x_batch), y_real) + criterion(D(x_fake), y_fake)
        opt_D.zero_grad(); D_loss.backward(); opt_D.step()
        # Train Generator
        z = torch.randn(bs, z_dim)
        x_fake = G(z)
        G_loss = criterion(D(x_fake), y_real)
        opt_G.zero_grad(); G_loss.backward(); opt_G.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch}: D_loss={D_loss.item():.3f}, G_loss={G_loss.item():.3f}")
        z = torch.randn(500, z_dim)
        plot_samples(G(z).detach(), title=f"Epoch {epoch}")
```

**Checkpoint:** Observe how the generator distribution evolves from noise toward the bimodal data distribution.

---

## Stage 2 — Library GAN (DCGAN on MNIST)

Use PyTorch’s deep CNN-based GAN example to generate MNIST digits.

```python
from torchvision import datasets, transforms
from torchvision.utils import make_grid

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
loader = DataLoader(train_data, batch_size=128, shuffle=True)

# Define DCGAN generator and discriminator
class DCGenerator(nn.Module):
    def __init__(self, z_dim=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 64*4, 3, 1, 0, bias=False),
            nn.BatchNorm2d(64*4),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*4, 64*2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64*2),
            nn.ReLU(True),
            nn.ConvTranspose2d(64*2, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, z): return self.net(z)

class DCDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(),
            nn.Linear(128*7*7, 1),
            nn.Sigmoid()
        )
    def forward(self, x): return self.net(x)
```

Train this DCGAN for 10–20 epochs and visualize generated samples.

**Checkpoint:** Save generated images every few epochs to visualize improvement.

---

## Stage 3 — Scaffolded Creative Challenge: Domain-Specific GAN

Design and train a GAN for a **new creative purpose**. Choose one of the options below (or propose your own).

### Option A — 2D Shape Generator

Generate simple geometric shapes (circles, triangles, squares) as $$28\times28$$ binary images. Train a small GAN to reproduce the dataset distribution.

**Scaffold:**
```python
def make_shape_dataset(n=1000, size=28):
    imgs = np.zeros((n, size, size))
    for i in range(n):
        kind = np.random.choice(["circle", "square", "triangle"])
        if kind == "circle":
            rr, cc = np.ogrid[:size, :size]
            mask = (rr - size/2)**2 + (cc - size/2)**2 < (size/3)**2
        elif kind == "square":
            mask = np.zeros((size, size))
            mask[size//4:3*size//4, size//4:3*size//4] = 1
        else:  # triangle
            mask = np.tril(np.ones((size, size)))
        imgs[i][mask > 0.5] = 1
    return torch.tensor(imgs).unsqueeze(1).float()

shape_data = make_shape_dataset()
```

**Tasks:**
1. Implement generator and discriminator suitable for $$28\times28$$ grayscale images.
2. Train your GAN to reproduce the shape dataset.
3. Visualize real vs. generated samples.
4. Measure diversity (e.g., mode count, variety).

---

### Option B — Time-Series GAN (1D Synthetic Data)

Simulate 1D temporal patterns (e.g., sine waves, square waves, noise bursts) and train a GAN to model the temporal distribution.

```python
def make_wave_dataset(n=1000, T=100):
    X = []
    for _ in range(n):
        t = np.linspace(0, 1, T)
        mode = np.random.choice(["sine", "square", "noise"])
        if mode == "sine":
            x = np.sin(2*np.pi*(3+np.random.rand())*t)
        elif mode == "square":
            x = np.sign(np.sin(2*np.pi*(2+np.random.rand())*t))
        else:
            x = np.random.normal(0, 1, size=T)
        X.append(x)
    return torch.tensor(np.array(X)).float().unsqueeze(1)

waves = make_wave_dataset()
```

**Tasks:**
1. Use 1D convolutional GANs (Conv1d / ConvTranspose1d).
2. Visualize generated waveforms vs. real samples.
3. Analyze whether your generator captures multimodal behavior.

---

### Option C — (Design Your Own)

Propose a small dataset and domain (e.g., melodies, text embeddings, robot trajectories). Describe:
- The data representation.
- The generator/discriminator architecture.
- Your training objectives and hyperparameters.

---

# What to Submit

1. A notebook implementing the **from-scratch GAN**, **library-based DCGAN**, and **creative challenge**.  
2. Generated sample plots (and optionally saved images) across epochs.  
3. Discussion: adversarial loss curves, convergence behavior, and observed artifacts.  
4. Reflections: how stability and mode collapse issues arose and how you mitigated them.  
5. Reproducibility: random seeds, hyperparameters, and code comments.

---
