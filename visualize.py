import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE

from gcn.models import GCN
from gcn.utils import load_data


def get_args():
    parser = argparse.ArgumentParser(description="Visualize GCN Embeddings with t-SNE")
    parser.add_argument("--checkpoint", type=str, default="gcn_model.pt", help="Path to trained model checkpoint.")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name (default: cora).")
    parser.add_argument("--data_dir", type=str, default="./data/cora/", help="Path to dataset directory.")
    parser.add_argument("--hidden", type=int, default=16, help="Hidden dimension used during training.")
    parser.add_argument("--output", type=str, default="gcn_tsne.png", help="Output plot filename.")
    return parser.parse_args()


def plot_tsne(embeddings, labels, output_path):
    print("Computing t-SNE 2D projection...")
    tsne = TSNE(n_components=2, perplexity=30, random_state=42)
    embeddings_2d = tsne.fit_transform(embeddings)

    plt.figure(figsize=(10, 8), dpi=300)
    unique_labels = np.unique(labels)
    colors = plt.cm.get_cmap("tab10", len(unique_labels))

    for idx, c in enumerate(unique_labels):
        mask = labels == c
        plt.scatter(
            embeddings_2d[mask, 0],
            embeddings_2d[mask, 1],
            c=[colors(idx)],
            label=f"Class {c}",
            alpha=0.8,
            s=25,
            edgecolors="none",
        )

    plt.title("t-SNE Visualization of Vanilla GCN Node Embeddings", fontsize=14, fontweight="bold")
    plt.xlabel("t-SNE Dimension 1", fontsize=11)
    plt.ylabel("t-SNE Dimension 2", fontsize=11)
    plt.legend(loc="best", framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
    print(f"Saved t-SNE visualization to: {output_path}")


def main():
    args = get_args()

    # Load data
    adj, features, labels, _, _, _ = load_data(path=args.data_dir, dataset=args.dataset)
    nfeat = features.shape[1]
    nclass = labels.max().item() + 1

    model = GCN(nfeat=nfeat, nhid=args.hidden, nclass=nclass)

    if os.path.exists(args.checkpoint):
        print(f"Loading checkpoint from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found. Using untrained model weights.")

    model.eval()
    with torch.no_grad():
        embeddings = model.get_embeddings(features, adj).cpu().numpy()

    labels_np = labels.cpu().numpy()
    plot_tsne(embeddings, labels_np, args.output)


if __name__ == "__main__":
    main()
