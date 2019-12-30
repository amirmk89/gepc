import torch
from torch import nn as nn


class ClusteringLayer(nn.Module):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.
    Partially ported from: https://github.com/XifengGuo/DCEC/ and https://github.com/michaal94/torch_DCEC/
    # Example
    ```
        cl = ClusteringLayer(n_clusters=10)
    ```
    # Arguments
        n_clusters: number of clusters.
        input_dim: size of input data with shape `(n_samples, n_features)`
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """
    def __init__(self, n_clusters, input_dim, weights=None, alpha=1.0, **kwargs):
        super(ClusteringLayer, self).__init__(**kwargs)
        if weights is not None:
            assert weights.shape[1] == input_dim
        self.n_clusters = n_clusters
        self.input_dim = input_dim  # (n_samples, n_features)
        self.alpha = alpha
        self.initial_weights = weights
        self.clusters = nn.Parameter(torch.Tensor(n_clusters, int(input_dim)))
        self.clusters = nn.init.xavier_uniform_(self.clusters)
        if self.initial_weights is not None:
            self.initial_weights = torch.from_numpy(self.initial_weights)
            self.clusters = nn.Parameter(self.initial_weights)
            del self.initial_weights
        self.input_dim = self.clusters.size(1)

    def forward(self, x):
        """ student t-distribution, as same as used in t-SNE algorithm.
                 q_ij = 1/(1+dist(x_i, u_j)^2), then normalize it.
        Arguments:
            x: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q_denom = (x.unsqueeze(1) - self.clusters)**2
        q_denom = q_denom.sum(dim=2)
        q_denom /= self.alpha
        q_denom += 1.0
        q = 1.0 / q_denom
        q = q ** ((self.alpha + 1.0) / 2.0)
        q = q.t() / q.sum(dim=1)  # Div shapes [20, 1024] / [1024]
        q = q.t()
        return q



