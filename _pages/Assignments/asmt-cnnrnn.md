---
layout: assignment
permalink: Assignments/CNN_RNN
title: "Assignment: Convolutional and Recurrent Neural Networks — From Scratch, with Libraries, and a Creative Challenge"

info:
  points: 100
  goals:
    - Understand the mathematical and computational foundations of CNNs and RNNs.
    - Implement basic convolutional and recurrent networks from scratch using PyTorch or NumPy.
    - Train CNNs and RNNs using library APIs on real-world datasets.
    - Solve a creative problem requiring temporal or spatial modeling with deep networks.
  purpose: "This assignment connects spatial feature extraction (CNNs) and temporal modeling (RNNs) to hands-on implementation. You will code core components manually, then use libraries for complex architectures, and apply them to a creative problem."
  concepts:
    - Convolution operations, padding, and receptive fields
    - Backpropagation through time (BPTT) and vanishing gradients
    - CNN and RNN architectures (Conv2D, LSTM, GRU)
    - Applications: image classification, time-series prediction, text modeling
  tasks:
    - Implement a CNN and RNN from scratch on toy datasets.
    - Use PyTorch or TensorFlow to train deep CNNs and RNNs on real data.
    - Complete a scaffolded creative problem integrating spatial and/or temporal learning.
  rubric:
    - weight: 30
      description: Implementation
      preemerging: Partial implementation or unstable training.
      beginning: Working forward pass, partial backward pass.
      progressing: Full working models and convergence.
      proficient: Stable, modular, and reproducible code with strong results.
    - weight: 30
      description: Conceptual Understanding
      preemerging: Minimal derivations or explanations.
      beginning: Correct convolution and recurrence equations.
      progressing: Interprets activations and sequence dynamics.
      proficient: Deep analysis of gradients, receptive fields, and memory.
    - weight: 20
      description: Analysis and Creativity
      preemerging: Limited evaluation or insight.
      beginning: Basic metrics and qualitative evaluation.
      progressing: Visualization of filters, hidden states, or saliency maps.
      proficient: Innovative application or analysis; interpretable insights.
    - weight: 10
      description: Code Quality & Documentation
      preemerging: Sparse comments or hard-coded logic.
      beginning: Basic organization.
      progressing: Modular and well-commented.
      proficient: Fully reproducible, readable, and configurable code.
    - weight: 10
      description: Submission Completeness
      preemerging: Missing results or plots.
      beginning: Includes code and metrics.
      progressing: Includes visualizations and clear explanation.
      proficient: Complete and well-structured notebook/report.

tags:
  - deep-learning
  - convolutional-networks
  - recurrent-networks
  - sequence-modeling
  - pytorch
---

# Overview

This assignment explores **Convolutional Neural Networks (CNNs)** and **Recurrent Neural Networks (RNNs)**. CNNs capture spatial patterns in images, while RNNs capture temporal or sequential dependencies. You will first implement simplified versions from scratch, then use modern deep learning libraries to train full models, and finally complete a creative, scaffolded project.

---

## Stage 0 — Setup

```python
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
torch.manual_seed(42)
```

---

## Stage 1 — Convolutional Neural Network (CNN) from Scratch

We implement a **2D convolution layer** and a **forward pass** for a simple CNN trained on a toy dataset.

### 1.1 Synthetic Image Dataset

```python
def make_toy_images(n=100, size=8):
    X, y = [], []
    for i in range(n):
        img = np.zeros((size, size))
        label = np.random.randint(0, 2)
        if label == 1:
            img[2:6, 2:6] = 1.0  # square pattern
        else:
            np.fill_diagonal(img, 1.0)  # diagonal line
        X.append(img + 0.1*np.random.randn(size, size))
        y.append(label)
    X = np.array(X).reshape(n, 1, size, size).astype(np.float32)
    y = np.array(y).astype(np.int64)
    return torch.tensor(X), torch.tensor(y)

X, y = make_toy_images(200)
train_loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)
```

### 1.2 Manual Convolution Layer (Forward Only)

```python
def conv2d_manual(X, W, b, stride=1, padding=0):
    n_filters, _, kH, kW = W.shape
    n, c, H, W_ = X.shape
    H_out = (H - kH + 2*padding)//stride + 1
    W_out = (W_ - kW + 2*padding)//stride + 1
    out = np.zeros((n, n_filters, H_out, W_out))
    X_pad = np.pad(X, ((0,0),(0,0),(padding,padding),(padding,padding)), mode='constant')
    for i in range(H_out):
        for j in range(W_out):
            region = X_pad[:, :, i*stride:i*stride+kH, j*stride:j*stride+kW]
            for f in range(n_filters):
                out[:, f, i, j] = np.sum(region * W[f], axis=(1,2,3)) + b[f]
    return out
```

**Checkpoint:** Visualize output activations for random filters.

---

## Stage 2 — CNN with PyTorch (on MNIST)

We train a CNN using library layers for image classification.

```python
from torchvision import datasets, transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_data = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
test_data = datasets.MNIST(root="./data", train=False, download=True, transform=transform)

train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.fc = nn.Sequential(
            nn.Linear(32*7*7, 64), nn.ReLU(),
            nn.Linear(64, 10)
        )
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

for epoch in range(3):
    for Xb, yb in train_loader:
        optimizer.zero_grad()
        out = model(Xb)
        loss = criterion(out, yb)
        loss.backward(); optimizer.step()
    print(f"Epoch {epoch+1}, loss={loss.item():.4f}")
```

**Checkpoint:** Evaluate on the test set and report accuracy.

---

## Stage 3 — Recurrent Neural Network (RNN) from Scratch (Sequence Prediction)

We will implement a **vanilla RNN cell** for sequence modeling (e.g., predicting the next symbol in a sequence).

```python
def rnn_step(x_t, h_prev, Wx, Wh, b):
    return np.tanh(np.dot(x_t, Wx) + np.dot(h_prev, Wh) + b)

Wx = np.random.randn(1, 16) * 0.1
Wh = np.random.randn(16, 16) * 0.1
b = np.zeros((1,16))
h = np.zeros((1,16))

seq = np.sin(np.linspace(0, 3*np.pi, 30)).reshape(-1,1)
hs = []
for x_t in seq:
    h = rnn_step(x_t, h, Wx, Wh, b)
    hs.append(h)
plt.plot([h[0,0] for h in hs]); plt.title("Hidden Unit Activation Over Time"); plt.show()
```

**Checkpoint:** Replace the sine input with another periodic pattern and observe hidden dynamics.

---

## Stage 4 — RNN with Libraries (Text Generation)

We train a small **character-level RNN** using `nn.RNN` or `nn.LSTM` to generate text.

```python
text = "hello world from cnn and rnn assignment"
chars = sorted(set(text))
stoi = {ch:i for i,ch in enumerate(chars)}
itos = {i:ch for ch,i in stoi.items()}

data = torch.tensor([stoi[ch] for ch in text], dtype=torch.long)
seq_len = 10
X = []; y = []
for i in range(len(data) - seq_len):
    X.append(data[i:i+seq_len])
    y.append(data[i+seq_len])
X, y = torch.stack(X), torch.tensor(y)
loader = DataLoader(TensorDataset(X, y), batch_size=16, shuffle=True)

class CharRNN(nn.Module):
    def __init__(self, vocab_size, hidden_size=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, hidden_size)
        self.rnn = nn.RNN(hidden_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
    def forward(self, x, h0=None):
        x = self.embed(x)
        out, h = self.rnn(x, h0)
        out = self.fc(out[:,-1,:])
        return out, h

model = CharRNN(len(chars))
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(50):
    for xb, yb in loader:
        optimizer.zero_grad()
        out, _ = model(xb)
        loss = criterion(out, yb)
        loss.backward(); optimizer.step()
    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss={loss.item():.3f}")
```

**Checkpoint:** Generate text by feeding the model’s predictions recursively.

---

## Stage 5 — Creative Mini-Project: **CNN–RNN Hybrid for Action Recognition (Scaffold)**

Design a **spatiotemporal classifier** combining CNNs and RNNs. Each sample is a short sequence of frames (e.g., moving dot patterns).

### 5.1 Data Scaffold

```python
def make_motion_dataset(n=200, T=5, size=16):
    X, y = [], []
    for i in range(n):
        cls = np.random.randint(0, 2)
        seq = np.zeros((T, 1, size, size))
        pos = np.random.randint(2, size-2)
        for t in range(T):
            if cls == 0:
                seq[t, 0, pos, 2+t] = 1.0
            else:
                seq[t, 0, 2+t, pos] = 1.0
        X.append(seq + 0.05*np.random.randn(*seq.shape))
        y.append(cls)
    return torch.tensor(np.array(X), dtype=torch.float32), torch.tensor(y)

X, y = make_motion_dataset()
```

### 5.2 Model Scaffold

```python
class CNN_RNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 8, 3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.rnn = nn.LSTM(8*(size//2)*(size//2), 32, batch_first=True)
        self.fc = nn.Linear(32, 2)

    def forward(self, x):
        B, T, C, H, W = x.shape
        x = x.view(B*T, C, H, W)
        feat = self.cnn(x).view(B, T, -1)
        out, _ = self.rnn(feat)
        return self.fc(out[:,-1,:])
```

**Tasks:**
1. Train the CNN–RNN model on the motion dataset for 10–20 epochs.  
2. Visualize learned filters or hidden trajectories.  
3. Experiment with sequence length $T$ and discuss trade-offs.  
4. (Stretch) Add attention or temporal pooling.

---

# What to Submit

1. Code and results for **CNN (scratch + library)** and **RNN (scratch + library)**.  
2. For the creative hybrid model: dataset code, training plots, and an analysis of temporal representations.  
3. Discussion of architectural choices, convergence, and challenges.  
4. Reproducibility: random seeds, model sizes, and runtime considerations.

---
