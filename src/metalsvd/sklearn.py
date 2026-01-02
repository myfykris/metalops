from sklearn.base import BaseEstimator, TransformerMixin
import torch
import metalsvd
import numpy as np

class MetalTruncatedSVD(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=2, n_iter=10, random_state=None, tol=0.0):
        self.n_components = n_components
        self.n_iter = n_iter
        self.random_state = random_state
        self.tol = tol
        self.components_ = None
        self.explained_variance_ = None
        self.explained_variance_ratio_ = None
        self.singular_values_ = None

    def fit(self, X, y=None):
        self.fit_transform(X)
        return self

    def fit_transform(self, X, y=None):
        X_tensor = torch.from_numpy(X).float().to("mps")
        
        # Use Randomized SVD for speed if Large, or Block SVD if small?
        # MetalSVD handles dispatch.
        if min(X.shape) > 1024: # Heuristic
            U, S, V = metalsvd.randomized_svd(X_tensor, k=self.n_components, n_iter=self.n_iter)
        else:
            U, S, V = metalsvd.svd(X_tensor)
            # Truncate
            k = self.n_components
            U = U[:, :k]
            S = S[:k]
            V = V[:, :k]

        self.components_ = V.t().cpu().numpy()
        self.singular_values_ = S.cpu().numpy()
        
        # Explained variance (approx)
        var = (S ** 2) / (X.shape[0] - 1)
        self.explained_variance_ = var.cpu().numpy()
        
        # Calculate total variance? (Expensive to compute full SVD)
        # Use X variance.
        total_var = torch.var(X_tensor, dim=0, unbiased=False).sum()
        self.explained_variance_ratio_ = self.explained_variance_ / total_var.item()

        return (U @ torch.diag(S)).cpu().numpy()

    def transform(self, X):
         X_tensor = torch.from_numpy(X).float().to("mps")
         # Projection: X * V
         V = torch.from_numpy(self.components_).float().to("mps")
         return (X_tensor @ V.t()).cpu().numpy()

    def inverse_transform(self, X):
        X_tensor = torch.from_numpy(X).float().to("mps")
        V = torch.from_numpy(self.components_).float().to("mps")
        return (X_tensor @ V).cpu().numpy()
