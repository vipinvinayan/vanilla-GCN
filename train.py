import argparse
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from gcn.models import GCN
from gcn.utils import load_data, accuracy


def get_args():
    parser = argparse.ArgumentParser(description="Vanilla GCN Training")
    parser.add_argument("--epochs", type=int, default=200, help="Number of epochs to train (default: 200).")
    parser.add_argument("--lr", type=float, default=0.01, help="Initial learning rate (default: 0.01).")
    parser.add_argument("--weight_decay", type=float, default=5e-4, help="Weight decay (L2 loss on parameters) (default: 5e-4).")
    parser.add_argument("--hidden", type=int, default=16, help="Number of hidden units (default: 16).")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate (1 - keep probability) (default: 0.5).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42).")
    parser.add_argument("--dataset", type=str, default="cora", help="Dataset name (default: cora).")
    parser.add_argument("--data_dir", type=str, default="./data/cora/", help="Directory to store dataset.")
    parser.add_argument("--save_path", type=str, default="gcn_model.pt", help="Path to save best model checkpoint.")
    parser.add_argument("--no_cuda", action="store_true", default=False, help="Disables CUDA training.")
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def train_epoch(epoch: int, model, optimizer, features, adj, labels, idx_train, idx_val):
    t_start = time.time()
    
    # Training step
    model.train()
    optimizer.zero_grad()
    output = model(features, adj)
    loss_train = F.nll_loss(output[idx_train], labels[idx_train])
    acc_train = accuracy(output[idx_train], labels[idx_train])
    loss_train.backward()
    optimizer.step()

    # Validation step (without dropout)
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_val = F.nll_loss(output[idx_val], labels[idx_val])
        acc_val = accuracy(output[idx_val], labels[idx_val])

    duration = time.time() - t_start

    if (epoch + 1) % 10 == 0 or epoch == 0:
        print(
            f"Epoch: {epoch+1:04d} | "
            f"Train Loss: {loss_train.item():.4f} | "
            f"Train Acc: {acc_train.item():.4f} | "
            f"Val Loss: {loss_val.item():.4f} | "
            f"Val Acc: {acc_val.item():.4f} | "
            f"Time: {duration:.4f}s"
        )
    return loss_val.item(), acc_val.item()


def evaluate(model, features, adj, labels, idx_test):
    model.eval()
    with torch.no_grad():
        output = model(features, adj)
        loss_test = F.nll_loss(output[idx_test], labels[idx_test])
        acc_test = accuracy(output[idx_test], labels[idx_test])
    return loss_test.item(), acc_test.item()


def main():
    args = get_args()
    set_seed(args.seed)

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print(f"Using device: {device}")

    # Load dataset
    adj, features, labels, idx_train, idx_val, idx_test = load_data(
        path=args.data_dir, dataset=args.dataset
    )

    print(f"Dataset stats: {features.shape[0]} nodes, {features.shape[1]} features, {labels.max().item() + 1} classes")
    print(f"Splits -> Train: {len(idx_train)}, Val: {len(idx_val)}, Test: {len(idx_test)}")

    # Move tensors to target device
    features = features.to(device)
    adj = adj.to(device)
    labels = labels.to(device)
    idx_train = idx_train.to(device)
    idx_val = idx_val.to(device)
    idx_test = idx_test.to(device)

    # Initialize model
    model = GCN(
        nfeat=features.shape[1],
        nhid=args.hidden,
        nclass=labels.max().item() + 1,
        dropout=args.dropout,
    ).to(device)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    print("\n--- Starting Training ---")
    best_val_loss = float("inf")
    best_val_acc = 0.0

    for epoch in range(args.epochs):
        val_loss, val_acc = train_epoch(
            epoch, model, optimizer, features, adj, labels, idx_train, idx_val
        )
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            torch.save(model.state_dict(), args.save_path)

    print("\n--- Optimization Finished ---")
    print(f"Best Validation Loss: {best_val_loss:.4f} | Best Validation Acc: {best_val_acc:.4f}")

    # Load best checkpoint and evaluate on test set
    model.load_state_dict(torch.load(args.save_path))
    test_loss, test_acc = evaluate(model, features, adj, labels, idx_test)
    print(f"\nFinal Test Results:")
    print(f"  Test Loss: {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc * 100:.2f}%)")


if __name__ == "__main__":
    main()
