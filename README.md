<<<<<<< HEAD
# Deep Learning with PyTorch

Progressive series covering PyTorch fundamentals through CNNs on Fashion MNIST.

## Structure

```
Deep_Learning/
├── 01_foundations/
│   ├── 01_tensors.ipynb            # Tensor creation, ops, NumPy interop
│   └── 02_autograd.ipynb           # Automatic differentiation, manual vs autograd gradients
│
├── 02_core_components/
│   ├── 03_nn_module.ipynb          # nn.Module patterns (logistic, hidden layers, Sequential)
│   ├── 04_dataset_dataloader.ipynb # Custom Dataset & DataLoader with batching
│   └── 05_training_pipeline.ipynb  # Full binary classification pipeline (Breast Cancer)
│
├── 03_ann/
│   ├── 06_ann_fashion_mnist.ipynb  # ANN on Fashion MNIST with GPU support, LR scheduler, per-class report
│   └── 07_ann_optuna.ipynb         # Hyperparameter search with Optuna, retrain best config
│
└── 04_cnn/
    └── 08_cnn_fashion_mnist.ipynb  # CNN on Fashion MNIST, two conv blocks, per-class report
```

## Data

| File | Used in |
|------|---------|
| `fashion_mnist.csv` | Notebooks 06, 07, 08 |
| Breast Cancer (auto-downloaded) | Notebook 05 |

Place `fashion_mnist.csv` in the same directory as the notebook, or update the `pd.read_csv` path.

## Requirements

```
torch torchvision torchinfo optuna scikit-learn pandas matplotlib
```

```bash
pip install torch torchvision torchinfo optuna scikit-learn pandas matplotlib
```

## Progression

1. **Tensors** → understand the core data structure  
2. **Autograd** → understand gradient computation  
3. **nn.Module** → build models with PyTorch's module system  
4. **Dataset / DataLoader** → efficient data pipelines  
5. **Training Pipeline** → end-to-end training loop pattern  
6. **ANN Fashion MNIST** → apply all the above to a real dataset  
7. **Optuna** → automate hyperparameter search  
8. **CNN Fashion MNIST** → add spatial feature extraction  
=======
# deep-learning
>>>>>>> 305671d99050f823b919bffec8e8b69088d40551
