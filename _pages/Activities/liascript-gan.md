# Generative Adversarial Networks (GAN)
<!--
author:   William M. Mongan
language: en
narrator: US English Male
comment:  Render via https://liascript.github.io/course/?https://github.com/BillJr99/Ursinus-CS477/blob/gh-pages/_pages/Activities/liascript-gan.md
import:   https://raw.githubusercontent.com/liascript/CodeRunner/master/README.md
link:     https://cdn.jsdelivr.net/gh/BillJr99/Ursinus-Boilerplate-Assets@main/css/liascript-custom.css?v=2025-08-23-5
           https://fonts.googleapis.com/css2?family=Lexend+Deca&display=swap
-->

# Generative Adversarial Networks (GAN)

This module presents the theory, mathematics, and practical aspects of **Generative Adversarial Networks (GANs)**, focusing on the adversarial training framework between a **generator** and a **discriminator**.  

We will derive the core min–max formulation, explain stability challenges, and explore modern extensions.  
All concepts are tied to the provided notebook:

---

## Open Colab: GAN on MNIST (From Scratch)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/GAN_MNIST.ipynb)

---

# 1. Motivation: Learning to Generate Data

Traditional supervised learning learns $p(y\mid x)$ from labeled pairs.  
**Generative modeling** instead seeks to learn a model of $p(x)$ — the distribution of data itself.  

Examples include:
- Sampling new digits similar to MNIST examples.
- Synthesizing realistic images or sounds.
- Learning structured latent spaces for style transfer or interpolation.

GANs propose an adversarial training paradigm where two networks **compete**:  
- **Generator (G):** tries to produce data resembling the real distribution.  
- **Discriminator (D):** tries to distinguish real data from generated (fake) data.

The resulting equilibrium ideally corresponds to $p_G(x) = p_{data}(x)$.

---

# 2. Theoretical Framework

## 2.1 Min–Max Game

The **value function** of a GAN is defined as:

$$
V(G, D) = \mathbb{E}_{x \sim p_{data}(x)} [\log D(x)] + \mathbb{E}_{z \sim p_z(z)} [\log (1 - D(G(z)))].
$$

Training proceeds as:
$$
\min_G \max_D V(G, D).
$$

Here:
- $p_z$ is a simple latent prior (e.g., uniform or Gaussian).
- $G(z)$ maps latent noise to data space.
- $D(x)$ outputs the probability that $x$ is real.

---

## 2.2 Optimal Discriminator

For a fixed $G$, we can analytically derive the optimal $D^*$:

$$
D^*(x) = \frac{p_{data}(x)}{p_{data}(x) + p_G(x)}.
$$

Substituting $D^*$ into $V(G,D)$ yields:

$$
C(G) = V(G, D^*) = -\log(4) + 2 \cdot \text{JSD}(p_{data} \parallel p_G),
$$

where **JSD** is the Jensen–Shannon divergence.  
Thus, minimizing the generator loss drives $p_G \to p_{data}$.

---

## 2.3 Training Objective Variants

The original minimax generator loss $\log(1 - D(G(z)))$ can cause vanishing gradients.  
Practically, we train $G$ to **maximize** $\log D(G(z))$ (non-saturating trick).

Alternative formulations include:
- **Least-Squares GAN:** use $(D(x) - y)^2$ loss for stability.
- **Wasserstein GAN (WGAN):** replaces JSD with Earth Mover distance for smoother gradients.
- **WGAN-GP:** adds gradient penalty to enforce Lipschitz continuity.

---

# 3. Architecture Overview

## 3.1 Generator Network

Maps random vector $z \in \mathbb{R}^d$ to image space. Typically uses:
- Transposed convolutions (a.k.a. deconvolutions)
- Batch normalization
- ReLU or LeakyReLU activations
- Tanh output layer

Example generator snippet:

```python
import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self, z_dim=100, img_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(z_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, img_dim),
            nn.Tanh()
        )
    def forward(self, z):
        return self.model(z)
```

---

## 3.2 Discriminator Network

Binary classifier distinguishing real from fake samples.

```python
class Discriminator(nn.Module):
    def __init__(self, img_dim=784):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(img_dim, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    def forward(self, x):
        return self.model(x)
```

---

# 4. Training the GAN (Adversarial Optimization)

## 4.1 Alternating Updates

1. **Sample minibatch of real data** $x \sim p_{data}(x)$  
2. **Sample latent noise** $z \sim p_z(z)$  
3. **Update discriminator** via:
   $$
   L_D = -[\log D(x) + \log (1 - D(G(z)))]
   $$
4. **Update generator** via:
   $$
   L_G = -\log D(G(z))
   $$

These updates alternate — typically one $D$ update per $G$ update.

---

## 4.2 Code Excerpt — MNIST GAN Training Loop

```python
import torch.optim as optim

z_dim = 100
device = "cuda" if torch.cuda.is_available() else "cpu"
gen, disc = Generator(z_dim).to(device), Discriminator().to(device)
criterion = nn.BCELoss()
g_opt = optim.Adam(gen.parameters(), lr=2e-4, betas=(0.5, 0.999))
d_opt = optim.Adam(disc.parameters(), lr=2e-4, betas=(0.5, 0.999))

for epoch in range(num_epochs):
    for real, _ in dataloader:
        real = real.view(-1, 784).to(device)
        batch_size = real.size(0)

        # Train Discriminator
        z = torch.randn(batch_size, z_dim).to(device)
        fake = gen(z)
        d_loss = criterion(disc(real), torch.ones_like(disc(real))) +                  criterion(disc(fake.detach()), torch.zeros_like(disc(fake)))
        d_opt.zero_grad(); d_loss.backward(); d_opt.step()

        # Train Generator
        z = torch.randn(batch_size, z_dim).to(device)
        fake = gen(z)
        g_loss = criterion(disc(fake), torch.ones_like(disc(fake)))
        g_opt.zero_grad(); g_loss.backward(); g_opt.step()
```

---

# 5. Training Behavior and Stability

## 5.1 Common Issues
- **Mode collapse:** generator produces limited diversity.
- **Vanishing gradients:** discriminator too strong early on.
- **Oscillations:** adversarial objectives fail to converge.

## 5.2 Remedies
- Label smoothing for real/fake labels.
- Feature matching loss.
- Minibatch discrimination.
- Wasserstein distance and gradient penalty (WGAN-GP).
- Two-time-scale update rule (TTUR).

---

# 6. Evaluating GAN Performance

Quantitative metrics:
- **Inception Score (IS):** measures realism and diversity.
- **Fréchet Inception Distance (FID):** distance between real and generated feature statistics.
- **Precision/Recall for Generative Models:** disentangles fidelity and coverage.

Qualitative evaluation:
- Visual inspection of generated samples over epochs.
- Interpolation in latent space to check smoothness.

---

# 7. Applications and Extensions

- **Conditional GANs (cGAN):** conditioning on class labels or attributes.
- **CycleGAN:** unpaired image-to-image translation.
- **StyleGAN:** progressive growing, adaptive instance normalization.
- **Diffusion Models:** modern extensions inspired by GANs but using denoising score matching.

---

# 8. Discussion and Exercises

[[MC]]
Which of the following best describes the goal of the generator in a GAN?
- (x) Produce samples that the discriminator classifies as real.
- ( ) Estimate the posterior $p(y \mid x)$.
- ( ) Directly compute the likelihood $p(x)$.

**Exercises:**
1. Modify the latent dimension $z_{dim}$ and observe its effect on diversity.  
2. Replace fully-connected layers with convolutional ones (DCGAN).  
3. Compare training with BCE loss vs. least-squares GAN loss.  
4. Plot generator and discriminator losses per epoch to detect instability.

---

# 9. Open Colab Links 

- GAN on MNIST (From Scratch): [Open in Colab](https://colab.research.google.com/github/BillJr99/Ursinus-CS477/blob/gh-pages/files/notebooks/GAN_MNIST.ipynb)
