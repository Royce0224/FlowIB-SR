"""
HSIC (Hilbert-Schmidt Independence Criterion) Module for Information Bottleneck
Contains: Standard HSIC, Linear HSIC (O(N) complexity), InfoNCE alternatives
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class HSICLoss(nn.Module):
    """
    Standard HSIC Loss with Gaussian RBF kernel.
    
    Complexity: O(N^2) - suitable for small batch sizes
    
    HSIC measures statistical dependence between two random variables.
    Minimizing HSIC(z_t, z_0) encourages the denoised representation
    to be independent of the original noisy social features.
    """
    
    def __init__(self, sigma: float = 1.0):
        super().__init__()
        self.sigma = sigma
    
    def gaussian_kernel(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute Gaussian RBF kernel matrix K(X, Y).
        
        K(x, y) = exp(-||x - y||^2 / (2 * sigma^2))
        
        Args:
            X: [n, d] first set of vectors
            Y: [m, d] second set of vectors
            
        Returns:
            K: [n, m] kernel matrix
        """
        # Compute squared Euclidean distances
        XX = torch.sum(X * X, dim=1, keepdim=True)  # [n, 1]
        YY = torch.sum(Y * Y, dim=1, keepdim=True)  # [m, 1]
        XY = torch.mm(X, Y.t())  # [n, m]
        
        dist_sq = XX + YY.t() - 2 * XY  # [n, m]
        
        # Apply Gaussian kernel
        K = torch.exp(-dist_sq / (2 * self.sigma ** 2))
        return K
    
    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute HSIC between X and Y.
        
        HSIC(X, Y) = (1/(n-1)^2) * tr(K_X H K_Y H)
        
        where H = I - (1/n) * 1 * 1^T is the centering matrix.
        
        Args:
            X: [batch_size, dim_x] first variable (e.g., z_t)
            Y: [batch_size, dim_y] second variable (e.g., z_0)
            
        Returns:
            hsic: scalar HSIC value
        """
        n = X.size(0)
        
        if n < 2:
            return torch.tensor(0.0, device=X.device)
        
        # Compute kernel matrices
        K_X = self.gaussian_kernel(X, X)  # [n, n]
        K_Y = self.gaussian_kernel(Y, Y)  # [n, n]
        
        # Centering matrix H = I - (1/n) * 1 * 1^T
        H = torch.eye(n, device=X.device) - torch.ones(n, n, device=X.device) / n
        
        # Centered kernels
        K_X_c = torch.mm(torch.mm(H, K_X), H)
        K_Y_c = torch.mm(torch.mm(H, K_Y), H)
        
        # HSIC = tr(K_X_c @ K_Y_c) / (n-1)^2
        hsic = torch.trace(torch.mm(K_X_c, K_Y_c)) / ((n - 1) ** 2)
        
        return hsic


class LinearHSIC(nn.Module):
    """
    Linear HSIC with O(N) complexity.
    
    Uses feature covariance approximation instead of kernel matrices.
    Much more efficient for large batch sizes while maintaining
    similar optimization properties.
    
    Reference: "A Kernel Two-Sample Test" (Gretton et al., 2012)
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(self, z_t: torch.Tensor, z_0: torch.Tensor) -> torch.Tensor:
        """
        Compute linear HSIC using feature covariance.
        
        Approximates HSIC by measuring the Frobenius norm of the
        cross-covariance matrix between centered features.
        
        Args:
            z_t: [batch_size, dim] denoised/intermediate embeddings
            z_0: [batch_size, dim] original noisy embeddings
            
        Returns:
            hsic: scalar linear HSIC value
        """
        n = z_t.size(0)
        
        if n < 2:
            return torch.tensor(0.0, device=z_t.device)
        
        # Center the features
        z_t_c = z_t - z_t.mean(dim=0, keepdim=True)
        z_0_c = z_0 - z_0.mean(dim=0, keepdim=True)
        
        # Compute cross-covariance matrix
        # Cov(z_t, z_0) = (1/(n-1)) * z_t_c^T @ z_0_c
        covariance = torch.matmul(z_t_c.t(), z_0_c) / (n - 1)
        
        # Frobenius norm of covariance measures dependence
        hsic = torch.norm(covariance, p='fro')
        
        return hsic


class FastHSIC(nn.Module):
    """
    Fast HSIC with random Fourier features approximation.
    
    Complexity: O(N * D) where D is the number of random features.
    Provides a good trade-off between accuracy and efficiency.
    """
    
    def __init__(self, input_dim: int, n_features: int = 256, sigma: float = 1.0):
        super().__init__()
        self.input_dim = input_dim
        self.n_features = n_features
        self.sigma = sigma
        
        # Random Fourier features (fixed after initialization)
        self.register_buffer('W', torch.randn(input_dim, n_features) / sigma)
        self.register_buffer('b', torch.rand(n_features) * 2 * torch.pi)
    
    def random_features(self, X: torch.Tensor) -> torch.Tensor:
        """
        Compute random Fourier features for Gaussian kernel approximation.
        
        φ(x) = sqrt(2/D) * cos(W^T x + b)
        
        Args:
            X: [n, d] input features
            
        Returns:
            phi: [n, D] random features
        """
        # Linear projection
        proj = torch.mm(X, self.W) + self.b  # [n, D]
        
        # Cosine features with normalization
        phi = torch.cos(proj) * (2.0 / self.n_features) ** 0.5
        
        return phi
    
    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        """
        Compute HSIC using random Fourier features.
        
        Args:
            X: [batch_size, dim] first variable
            Y: [batch_size, dim] second variable
            
        Returns:
            hsic: scalar HSIC approximation
        """
        n = X.size(0)
        
        if n < 2:
            return torch.tensor(0.0, device=X.device)
        
        # Compute random features
        phi_X = self.random_features(X)  # [n, D]
        phi_Y = self.random_features(Y)  # [n, D]
        
        # Center the features
        phi_X_c = phi_X - phi_X.mean(dim=0, keepdim=True)
        phi_Y_c = phi_Y - phi_Y.mean(dim=0, keepdim=True)
        
        # HSIC ≈ ||Cov(φ_X, φ_Y)||_F^2
        cov = torch.mm(phi_X_c.t(), phi_Y_c) / (n - 1)
        hsic = torch.sum(cov ** 2)
        
        return hsic


class InfoNCELoss(nn.Module):
    """
    InfoNCE Loss as an alternative to HSIC for information bottleneck.
    
    Maximizes mutual information between z_t and z_1 (target)
    while implicitly minimizing information about z_0 (source).
    
    This is a contrastive approach that can be more stable than HSIC
    in some scenarios.
    """
    
    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, z_t: torch.Tensor, z_pos: torch.Tensor, 
                z_neg: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            z_t: [batch_size, dim] anchor embeddings
            z_pos: [batch_size, dim] positive embeddings (same user, different view)
            z_neg: [batch_size, dim] or None, negative embeddings
                   If None, uses other samples in batch as negatives
            
        Returns:
            loss: scalar InfoNCE loss
        """
        # Normalize embeddings
        z_t = F.normalize(z_t, p=2, dim=1)
        z_pos = F.normalize(z_pos, p=2, dim=1)
        
        # Positive similarity
        pos_sim = torch.sum(z_t * z_pos, dim=1) / self.temperature  # [batch_size]
        
        if z_neg is None:
            # Use all other samples as negatives (in-batch negatives)
            sim_matrix = torch.mm(z_t, z_t.t()) / self.temperature  # [batch_size, batch_size]
            
            # Mask out self-similarity
            mask = torch.eye(z_t.size(0), device=z_t.device).bool()
            sim_matrix = sim_matrix.masked_fill(mask, float('-inf'))
            
            # Log-sum-exp over negatives
            neg_sim = torch.logsumexp(sim_matrix, dim=1)  # [batch_size]
        else:
            z_neg = F.normalize(z_neg, p=2, dim=1)
            neg_sim = torch.sum(z_t * z_neg, dim=1) / self.temperature
        
        # InfoNCE loss: -log(exp(pos) / (exp(pos) + sum(exp(neg))))
        loss = -torch.mean(pos_sim - torch.logsumexp(
            torch.stack([pos_sim, neg_sim], dim=1), dim=1
        ))
        
        return loss


class CombinedIBLoss(nn.Module):
    """
    Combined Information Bottleneck Loss.
    
    Combines multiple IB objectives:
    1. HSIC(z_t, z_0): Minimize dependence on noisy source
    2. Alignment(z_t, z_1): Maximize alignment with clean target
    
    Total IB Loss = λ_hsic * HSIC(z_t, z_0) - λ_align * Alignment(z_t, z_1)
    """
    
    def __init__(self, use_linear_hsic: bool = True, 
                 lambda_hsic: float = 1.0, lambda_align: float = 0.1):
        super().__init__()
        
        if use_linear_hsic:
            self.hsic = LinearHSIC()
        else:
            self.hsic = HSICLoss()
        
        self.lambda_hsic = lambda_hsic
        self.lambda_align = lambda_align
    
    def alignment_loss(self, z_t: torch.Tensor, z_1: torch.Tensor) -> torch.Tensor:
        """
        Compute alignment loss between z_t and target z_1.
        
        Uses negative cosine similarity.
        """
        z_t_norm = F.normalize(z_t, p=2, dim=1)
        z_1_norm = F.normalize(z_1, p=2, dim=1)
        
        # Negative cosine similarity (we want to maximize alignment)
        alignment = -torch.mean(torch.sum(z_t_norm * z_1_norm, dim=1))
        
        return alignment
    
    def forward(self, z_t: torch.Tensor, z_0: torch.Tensor, 
                z_1: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute combined IB loss.
        
        Args:
            z_t: [batch_size, dim] intermediate/denoised embeddings
            z_0: [batch_size, dim] source (noisy) embeddings
            z_1: [batch_size, dim] target (clean) embeddings (optional)
            
        Returns:
            loss: scalar combined IB loss
        """
        # HSIC loss: minimize dependence on source
        hsic_loss = self.hsic(z_t, z_0)
        
        total_loss = self.lambda_hsic * hsic_loss
        
        # Alignment loss: maximize alignment with target
        if z_1 is not None and self.lambda_align > 0:
            align_loss = self.alignment_loss(z_t, z_1)
            total_loss = total_loss + self.lambda_align * align_loss
        
        return total_loss


def fast_hsic(z_t: torch.Tensor, z_0: torch.Tensor) -> torch.Tensor:
    """
    Functional interface for linear HSIC.
    
    GPU-optimized linear HSIC with O(N) complexity.
    
    Args:
        z_t: [batch_size, dim] denoised embeddings
        z_0: [batch_size, dim] original noisy embeddings
        
    Returns:
        hsic: scalar HSIC value
    """
    n = z_t.size(0)
    
    if n < 2:
        return torch.tensor(0.0, device=z_t.device)
    
    # Center the features
    z_t_c = z_t - z_t.mean(dim=0, keepdim=True)
    z_0_c = z_0 - z_0.mean(dim=0, keepdim=True)
    
    # Compute cross-covariance and its Frobenius norm
    covariance = torch.matmul(z_t_c.t(), z_0_c) / (n - 1)
    
    return torch.norm(covariance, p='fro')


def create_ib_loss(loss_type: str = 'linear_hsic', **kwargs) -> nn.Module:
    """
    Factory function to create IB loss module.
    
    Args:
        loss_type: 'hsic', 'linear_hsic', 'fast_hsic', 'infonce', 'combined'
        **kwargs: additional arguments for the loss module
        
    Returns:
        loss_module: IB loss module
    """
    if loss_type == 'hsic':
        return HSICLoss(sigma=kwargs.get('sigma', 1.0))
    elif loss_type == 'linear_hsic':
        return LinearHSIC()
    elif loss_type == 'fast_hsic':
        return FastHSIC(
            input_dim=kwargs.get('input_dim', 64),
            n_features=kwargs.get('n_features', 256),
            sigma=kwargs.get('sigma', 1.0)
        )
    elif loss_type == 'infonce':
        return InfoNCELoss(temperature=kwargs.get('temperature', 0.1))
    elif loss_type == 'combined':
        return CombinedIBLoss(
            use_linear_hsic=kwargs.get('use_linear_hsic', True),
            lambda_hsic=kwargs.get('lambda_hsic', 1.0),
            lambda_align=kwargs.get('lambda_align', 0.1)
        )
    else:
        raise ValueError(f"Unknown IB loss type: {loss_type}")
