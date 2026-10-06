import math
import torch
import torch.nn as nn
from torch.nn.parameter import Parameter


class GraphConvolution(nn.Module):
    """
    Simple Graph Convolutional Layer (Vanilla GCN layer),
    as described in:
    "Semi-Supervised Classification with Graph Convolutional Networks"
    (Kipf & Welling, ICLR 2017).

    Implements:
        Z = A_hat * X * W + b
    where A_hat is the symmetric normalized adjacency matrix D^(-1/2) * A_tilde * D^(-1/2).
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize weights with Glorot / Xavier uniform distribution."""
        stdv = 1.0 / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input_features: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for the graph convolution layer.

        Args:
            input_features (torch.Tensor): Node feature matrix of shape (N, in_features).
            adj (torch.Tensor): Normalized adjacency matrix of shape (N, N)
                               (can be dense or sparse COO tensor).

        Returns:
            torch.Tensor: Convolved output node features of shape (N, out_features).
        """
        # Linear transformation: X * W (shape: [N, out_features])
        support = torch.mm(input_features, self.weight)

        # Graph aggregation: A_hat * (X * W)
        if adj.is_sparse:
            output = torch.spmm(adj, support)
        else:
            output = torch.mm(adj, support)

        # Add bias if present
        if self.bias is not None:
            output = output + self.bias

        return output

    def __repr__(self):
        return f"{self.__class__.__name__}(in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None})"
