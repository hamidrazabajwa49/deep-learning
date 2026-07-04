# 02 — Core Components: Modules, Data Pipelines & Training Loops

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Status](https://img.shields.io/badge/status-complete-brightgreen)

**The three building blocks that turn tensors + autograd (from `01_foundations`) into a trainable model**: `nn.Module` (how a network is *defined*), `Dataset`/`DataLoader` (how data is *fed in*), and the training loop (how the model *learns*). This folder assembles all three into one working end-to-end classifier using a **local dataset (`data/data.csv`)**.

---

## 📦 What's Inside

| Notebook                                                       | Covers                                                                                                                  | Core Question It Answers                                          |
| -------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------- |
| [`03_nn_module.ipynb`](./03_nn_module.ipynb)                   | Three patterns for defining models with `nn.Module`: single layer, explicit multi-layer, `nn.Sequential`                | "How do I define a model's architecture in PyTorch?"              |
| [`04_dataset_dataloader.ipynb`](./04_dataset_dataloader.ipynb) | Custom `Dataset` class, `DataLoader` batching/shuffling, train/test splitting                                           | "How do I get raw data into a model, efficiently and in batches?" |
| [`05_training_pipeline.ipynb`](./05_training_pipeline.ipynb)   | Full pipeline on the Breast Cancer dataset: preprocessing → model → loop → evaluation (now loaded from `data/data.csv`) | "How do all the pieces fit together into one working system?"     |

---

## 🚀 Quickstart (60 seconds)

```bash
git clone https://github.com/hamidrazabajwa49/deep-learning.git
cd deep-learning/02_core_components

python -m venv venv && source venv/bin/activate   # optional but recommended
pip install torch torchinfo numpy pandas scikit-learn jupyter

jupyter notebook
```

Recommended reading order: `03 → 04 → 05`.
Notebook 5 reuses the exact `Dataset` pattern from notebook 4 and the `nn.Sequential` pattern from notebook 3 — it's the payoff notebook.

---

## 📁 Data

The dataset is now stored locally:

```
data/
└── data.csv   # Breast Cancer dataset
```

**Why this matters:**

* No online data fetch dependency 
* More realistic pipeline (file → preprocessing → model)
* Aligns with real-world ML workflows

In `05_training_pipeline.ipynb`, the dataset is loaded via:

```python
import pandas as pd
df = pd.read_csv("data/data.csv")
```

---

## 🧠 Concepts & Intuition

### 1. `nn.Module` — "A container that bundles parameters with computation"

**Core intuition:** A PyTorch model is a class that inherits from `nn.Module`. Any layer assigned as an attribute (`nn.Linear`, etc.) is automatically registered as a learnable parameter. Calling `model(x)` internally executes `forward(x)`.

| Subtopic                        | What it means                                  |
| ------------------------------- | ---------------------------------------------- |
| **Pattern 1 — Single layer**    | Minimal model (logistic regression equivalent) |
| **Pattern 2 — Explicit layers** | Full control via custom `forward()`            |
| **Pattern 3 — `nn.Sequential`** | Clean shorthand for linear stacks              |

**Key tradeoff:**

* `Sequential` → concise, but no branching
* Custom `forward()` → flexible, but more verbose

---

### 2. `Dataset` & `DataLoader` — "Separate *what data is* from *how it's served*"

**Core intuition:**

* `Dataset` → defines indexing logic (`__len__`, `__getitem__`)
* `DataLoader` → handles batching, shuffling, parallelism

| Component    | Responsibility              |
| ------------ | --------------------------- |
| `Dataset`    | Returns one sample `(x, y)` |
| `DataLoader` | Returns mini-batches        |

**Important insight:**
You can reuse the same `Dataset` with different batch sizes, shuffle strategies, or hardware setups — no code changes required.

---

### 3. Training Pipeline — "Iterative error minimization"

**Core loop:**

1. Forward pass → prediction
2. Compute loss → how wrong
3. Backward pass → gradients
4. Optimizer step → update weights
5. Reset gradients

```text
forward → loss → backward → step → zero_grad
```

| Component         | Role                                     |
| ----------------- | ---------------------------------------- |
| **Preprocessing** | Standardization ensures stable gradients |
| **Model**         | `nn.Sequential` binary classifier        |
| **Loss**          | `BCELoss` for binary classification      |
| **Optimizer**     | `SGD` updates parameters                 |
| **Evaluation**    | `model.eval()` + `torch.no_grad()`       |

**Critical detail:**
`optimizer.zero_grad()` must be called every iteration — gradients accumulate by default.

---

## 🗂️ Repository Structure

```
02_core_components/
├── data/
│   └── data.csv
├── 03_nn_module.ipynb
├── 04_dataset_dataloader.ipynb
├── 05_training_pipeline.ipynb
└──README.md
```

---

## 📊 Key Takeaways

| Idea                 | One-line summary                 |
| -------------------- | -------------------------------- |
| `nn.Module`          | Base abstraction for all models  |
| `nn.Sequential`      | Fast way to stack layers         |
| `Dataset`            | Defines data access logic        |
| `DataLoader`         | Handles batching and shuffling   |
| Training loop        | Core learning mechanism          |
| `train()` / `eval()` | Controls model behavior          |
| `no_grad()`          | Disables gradients for inference |

---

**Bottom line:**
This folder transitions you from *understanding tensors and gradients* → *building a complete, real-world training pipeline using structured data from disk*.

---
