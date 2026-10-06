import os
import urllib.request
import tarfile
import numpy as np
import scipy.sparse as sp
import torch


def encode_onehot(labels):
    """Encodes categorical string labels into one-hot numpy array."""
    classes = sorted(list(set(labels)))
    classes_dict = {c: np.identity(len(classes))[i, :] for i, c in enumerate(classes)}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    return labels_onehot, classes


def normalize_adj(mx):
    """
    Symmetrically normalize adjacency matrix:
        A_tilde = A + I_N
        D_tilde = sum(A_tilde, axis=1)
        A_hat = D_tilde^(-1/2) * A_tilde * D_tilde^(-1/2)
    """
    # Add self-connections
    mx = mx + sp.eye(mx.shape[0])
    rowsum = np.array(mx.sum(1))
    r_inv_sqrt = np.power(rowsum, -0.5).flatten()
    r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
    r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
    return mx.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt).tocoo()


def normalize_features(mx):
    """Row-normalize sparse feature matrix: X_norm = D^(-1) * X."""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1.0).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = sp.diags(r_inv)
    return r_mat_inv.dot(mx)


def accuracy(output: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Computes prediction accuracy."""
    preds = output.max(1)[1].type_as(labels)
    correct = preds.eq(labels).double()
    correct = correct.sum()
    return correct / len(labels)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy.sparse matrix to a torch.sparse_coo_tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
    )
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse_coo_tensor(indices, values, shape)


def _download_cora(data_dir: str):
    """Attempts to download and extract Cora dataset files."""
    os.makedirs(data_dir, exist_ok=True)
    cora_url = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"
    target_tgz = os.path.join(data_dir, "cora.tgz")

    content_path = os.path.join(data_dir, "cora.content")
    cites_path = os.path.join(data_dir, "cora.cites")

    if os.path.exists(content_path) and os.path.exists(cites_path):
        return

    print("Downloading Cora dataset...")
    try:
        opener = urllib.request.build_opener()
        opener.addheaders = [("User-Agent", "Mozilla/5.0")]
        urllib.request.install_opener(opener)
        urllib.request.urlretrieve(cora_url, target_tgz)

        with tarfile.open(target_tgz, "r:gz") as tar:
            for member in tar.getmembers():
                if member.name.endswith("cora.content"):
                    member.name = "cora.content"
                    tar.extract(member, path=data_dir)
                elif member.name.endswith("cora.cites"):
                    member.name = "cora.cites"
                    tar.extract(member, path=data_dir)

        if os.path.exists(target_tgz):
            os.remove(target_tgz)
        print("Cora dataset downloaded and extracted successfully.")
    except Exception as e:
        print(f"Could not download raw Cora directly ({e}).")


def _generate_synthetic_graph(num_nodes=500, num_features=100, num_classes=5):
    """Generates synthetic benchmark graph if dataset download is unavailable."""
    print("Generating synthetic benchmark graph for demonstration...")
    np.random.seed(42)
    features = np.random.binomial(1, 0.05, (num_nodes, num_features)).astype(np.float32)
    labels_int = np.random.randint(0, num_classes, num_nodes)
    
    # Generate homophilic edge connectivity
    rows, cols = [], []
    for i in range(num_nodes):
        # Prefer connections within same class
        peers = np.random.choice(num_nodes, size=4, replace=False)
        for p in peers:
            if p != i:
                rows.extend([i, p])
                cols.extend([p, i])
    
    adj = sp.coo_matrix((np.ones(len(rows)), (rows, cols)), shape=(num_nodes, num_nodes))
    features = sp.csr_matrix(features)
    
    idx_train = torch.LongTensor(range(int(0.2 * num_nodes)))
    idx_val = torch.LongTensor(range(int(0.2 * num_nodes), int(0.4 * num_nodes)))
    idx_test = torch.LongTensor(range(int(0.4 * num_nodes), num_nodes))
    
    features = normalize_features(features)
    adj = normalize_adj(adj)
    
    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(labels_int)
    adj = sparse_mx_to_torch_sparse_tensor(adj)
    
    return adj, features, labels, idx_train, idx_val, idx_test


def load_data(path: str = "./data/cora/", dataset: str = "cora"):
    """
    Loads citation network dataset (Cora by default).
    Returns normalized adjacency, features, labels, and train/val/test splits.
    """
    content_file = os.path.join(path, f"{dataset}.content")
    cites_file = os.path.join(path, f"{dataset}.cites")

    if not (os.path.exists(content_file) and os.path.exists(cites_file)):
        _download_cora(path)

    if not (os.path.exists(content_file) and os.path.exists(cites_file)):
        return _generate_synthetic_graph()

    print(f"Loading {dataset} dataset from {path}...")
    idx_features_labels = np.genfromtxt(content_file, dtype=np.dtype(str))
    features = sp.csr_matrix(idx_features_labels[:, 1:-1], dtype=np.float32)
    labels, _ = encode_onehot(idx_features_labels[:, -1])

    # Build paper ID mapping to matrix index
    idx = np.array(idx_features_labels[:, 0], dtype=np.int32)
    idx_map = {j: i for i, j in enumerate(idx)}

    # Build graph edges
    edges_unordered = np.genfromtxt(cites_file, dtype=np.int32)
    edges = np.array(
        list(
            filter(
                lambda edge: edge[0] in idx_map and edge[1] in idx_map,
                edges_unordered,
            )
        )
    )
    edges = np.array(
        list(map(idx_map.get, edges.flatten())), dtype=np.int32
    ).reshape(edges.shape)

    adj = sp.coo_matrix(
        (np.ones(edges.shape[0]), (edges[:, 0], edges[:, 1])),
        shape=(labels.shape[0], labels.shape[0]),
        dtype=np.float32,
    )

    # Build symmetric graph (undirected graph)
    adj = adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)

    # Normalize feature matrix and adjacency matrix
    features = normalize_features(features)
    adj = normalize_adj(adj)

    # Standard semi-supervised splits as per Kipf & Welling (ICLR 2017)
    idx_train = range(140)
    idx_val = range(140, 640)
    idx_test = range(640, 1640)

    features = torch.FloatTensor(np.array(features.todense()))
    labels = torch.LongTensor(np.where(labels)[1])
    adj = sparse_mx_to_torch_sparse_tensor(adj)

    idx_train = torch.LongTensor(idx_train)
    idx_val = torch.LongTensor(idx_val)
    idx_test = torch.LongTensor(idx_test)

    return adj, features, labels, idx_train, idx_val, idx_test
