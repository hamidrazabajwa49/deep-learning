# deep-learning

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch)
![Status](https://img.shields.io/badge/status-active-brightgreen)

> PyTorch fundamentals, built from the ground up — tensor mechanics and reverse-mode autodiff, verified against hand-derived gradients notebook by notebook.

---

## 🧠 Why This Exists

- **Ground-truth verification** — every autograd result is cross-checked against a manually derived gradient, not just trusted blindly (see `01_foundations/02_autograd.ipynb`, binary classifier example).
- **Full tensor surface area** — creation, dtypes, broadcasting, reductions, linear algebra (`matmul`, `det`, `inverse`), and reshaping in one reference notebook.
- **Reverse-mode AD from first principles** — scalar gradients → chain rule → vector/mean gradients, building intuition for how `.backward()` actually walks the computation graph.
- **NumPy interop** covered explicitly (`torch.from_numpy`, zero-copy `.numpy()`), since most real pipelines cross this boundary constantly.

---

## 🚀 Quickstart (60 seconds)

```bash
git clone https://github.com/hamidrazabajwa49/deep-learning.git
cd deep-learning
python -m venv venv && source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook 01_foundations/01_tensors.ipynb
```

Then open `01_foundations/02_autograd.ipynb` next.

---

## 🔬 Under the Hood

**Stack:** PyTorch, NumPy, Jupyter.

**Learning path / dependency flow:**

```mermaid
flowchart LR
    A[01_foundations/01_tensors.ipynb] -->|tensor ops, dtypes, matmul| B[01_foundations/02_autograd.ipynb]
    B --> C[Scalar gradients: y = x²]
    C --> D[Chain rule: z = sin(x²)]
    D --> E[Manual BCE gradient derivation]
    E --> F[Autograd BCE gradient — cross-check]
    F --> G[Vector gradients: mean(x²)]
```

**Notebook 1 — `01_foundations/01_tensors.ipynb`**
| Section | Covers |
|---|---|
| Tensor Creation | `empty`, `zeros`, `ones`, `rand`, `arange`, `linspace`, `eye`, `full` |
| Shapes & dtypes | `.shape`, `.dtype`, `zeros_like`, `ones_like` |
| Math Ops | scalar ops, element-wise ops, reductions (`sum`, `mean`, `std`, `argmax`) |
| Linear Algebra | `matmul`, `dot`, `transpose`, `det`, `inverse` |
| Algebraic Functions | `log`, `exp`, `sqrt`, `sigmoid`, `softmax`, `relu` |
| Reshaping | `reshape`, `flatten`, `permute`, `squeeze`/`unsqueeze` |
| In-place Ops | `add_`, `relu_` (trailing underscore convention) |
| NumPy Interop | `from_numpy`, `.numpy()` |

**Notebook 2 — `01_foundations/02_autograd.ipynb`**
| Section | Covers |
|---|---|
| Scalar Gradients | `requires_grad=True`, `.backward()`, `.grad` on `y = x²` |
| Chain Rule | `z = sin(x²)` → verifies `dz/dx = 2x·cos(x²)` |
| Manual vs Autograd | Binary cross-entropy loss: hand-derived `dL/dw`, `dL/db` vs. autograd, side by side |
| Vector Gradients | `.backward()` on `mean(x²)` across a vector input |

---
