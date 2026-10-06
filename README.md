# Vanilla GCN (Graph Convolutional Networks)

A clean, modular, and dependency-light implementation of **Vanilla Graph Convolutional Networks (GCN)** in PyTorch from scratch, based on the foundational paper:

> **[Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907)**  
> *Thomas N. Kipf, Max Welling (ICLR 2017)*

---

## 📐 Mathematical Formulation

The Graph Convolution layer implements the localized first-order spectral graph convolution:

$$H^{(l+1)} = \sigma\left(\tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}} H^{(l)} W^{(l)}\right)$$

Where:
- $\tilde{A} = A + I_N$: Adjacency matrix with self-loops added.
- $\tilde{D}_{ii} = \sum_{j} \tilde{A}_{ij}$: Degree matrix of $\tilde{A}$.
- $\hat{A} = \tilde{D}^{-\frac{1}{2}} \tilde{A} \tilde{D}^{-\frac{1}{2}}$: Symmetric normalized adjacency matrix.
- $W^{(l)} \in \mathbb{R}^{d_l \times d_{l+1}}$: Trainable layer weight matrix.
- $\sigma(\cdot)$: Non-linear activation function (e.g., $\text{ReLU}$).
- $H^{(0)} = X \in \mathbb{R}^{N \times F}$: Input node feature matrix.

---

## 📁 Repository Structure

```text
vanilla-gcn/
├── gcn/
│   ├── __init__.py      # Package export
│   ├── layers.py        # GraphConvolution layer (PyTorch sparse/dense support)
│   ├── models.py        # 2-layer Vanilla GCN classifier
│   └── utils.py         # Adjacency normalization, dataset loader, accuracy metrics
├── data/                # Data storage directory (auto-downloaded)
├── train.py             # Training and evaluation pipeline
├── visualize.py         # 2D t-SNE latent representation visualizer
├── requirements.txt     # Python dependencies
├── .gitignore           # Git ignore patterns
└── README.md            # Project documentation
```

---

## 🚀 Quickstart

### 1. Installation

Clone your repository and install dependencies:

```bash
git clone https://github.com/<your-username>/vanilla-gcn.git
cd vanilla-gcn
pip install -r requirements.txt
```

### 2. Train the Model on Cora

Run node classification on the Cora citation benchmark:

```bash
python train.py --epochs 200 --lr 0.01 --weight_decay 5e-4 --hidden 16 --dropout 0.5
```

Sample output:
```text
Loaded Cora dataset:
  Nodes: 2,708
  Features: 1,433
  Classes: 7
  Edges: 5,429

Epoch: 0010 | Train Loss: 1.8341 | Train Acc: 0.6286 | Val Loss: 1.8522 | Val Acc: 0.5640
Epoch: 0050 | Train Loss: 0.7230 | Train Acc: 0.9286 | Val Loss: 0.9841 | Val Acc: 0.7680
Epoch: 0100 | Train Loss: 0.3541 | Train Acc: 0.9786 | Val Loss: 0.7320 | Val Acc: 0.8040
Epoch: 0200 | Train Loss: 0.1872 | Train Acc: 0.9929 | Val Loss: 0.7011 | Val Acc: 0.8160

Optimization Finished!
Test Set Results: Loss: 0.7142 | Accuracy: 0.8150 (81.50%)
```

### 3. Visualize Learned Embeddings

Visualize the node embeddings learned by the GCN using t-SNE:

```bash
python visualize.py --output gcn_tsne.png
```

---

## ⚙️ Hyperparameters & Options

| Argument | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--epochs` | `int` | `200` | Number of training epochs |
| `--lr` | `float` | `0.01` | Learning rate (Adam optimizer) |
| `--weight_decay` | `float` | `5e-4` | L2 regularization on Layer 1 weights |
| `--hidden` | `int` | `16` | Number of hidden units |
| `--dropout` | `float` | `0.5` | Dropout rate applied to features and hidden states |
| `--seed` | `int` | `42` | Random seed for reproducibility |
| `--no_cuda` | `flag` | `False` | Disable CUDA / GPU acceleration |

---

## 📚 References

- Kipf, T. N., & Welling, M. (2017). [Semi-Supervised Classification with Graph Convolutional Networks](https://arxiv.org/abs/1609.02907). *ICLR 2017*.
- Kipf's original TensorFlow implementation: [tkipf/gcn](https://github.com/tkipf/gcn)

---

## 📜 License

MIT License. Feel free to use and adapt this code for research and projects!
