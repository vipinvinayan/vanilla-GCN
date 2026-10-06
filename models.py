import torch
import torch.nn as nn
import torch.nn.functional as F
from .layers import GraphConvolution


class GCN(nn.Module):
    """
    Two-layer Vanilla Graph Convolutional Network (GCN) architecture for
    semi-supervised node classification, as described by Kipf & Welling (ICLR 2017).

    Architecture:
        Input Features
             │
          Dropout
             │
        GraphConv 1 (in_features -> hidden_dim)
             │
           ReLU
             │
          Dropout
             │
        GraphConv 2 (hidden_dim -> n_classes)
             │
         LogSoftmax
    """

    def __init__(self, nfeat: int, nhid: int, nclass: int, dropout: float = 0.5):
        """
        Args:
            nfeat (int): Dimensionality of input node features.
            nhid (int): Dimensionality of hidden representation.
            nclass (int): Number of target classes.
            dropout (float): Dropout probability (applied to features and hidden activations).
        """
        super(GCN, self).__init__()

        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, nclass)
        self.dropout = dropout

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x (torch.Tensor): Feature matrix of shape (N, nfeat).
            adj (torch.Tensor): Normalized adjacency matrix of shape (N, N).

        Returns:
            torch.Tensor: Log-probabilities of shape (N, nclass).
        """
        # First Graph Convolutional Layer
        x = F.dropout(x, self.dropout, training=self.training)
        x = F.relu(self.gc1(x, adj))

        # Second Graph Convolutional Layer
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)

        return F.log_softmax(x, dim=1)

    def get_embeddings(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Extract the intermediate hidden node representations (e.g. for t-SNE visualization).

        Args:
            x (torch.Tensor): Feature matrix of shape (N, nfeat).
            adj (torch.Tensor): Normalized adjacency matrix of shape (N, N).

        Returns:
            torch.Tensor: Hidden representations of shape (N, nhid).
        """
        self.eval()
        with torch.no_grad():
            x = F.relu(self.gc1(x, adj))
        return x
