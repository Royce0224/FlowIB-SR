"""
Flow Matching Module for Flow-IB Social Recommendation
Contains: VelocityNet, ODE Solvers (Euler, RK4), FlowMatcher
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class VelocityNet(nn.Module):
    """
    Velocity Network v_θ(z_t, t) for Flow Matching.
    
    Predicts the velocity field that transforms noisy social embeddings
    to clean preference embeddings along a learned trajectory.
    
    Architecture:
        - Time embedding via sinusoidal encoding
        - Residual MLP with GroupNorm (more stable than LayerNorm for small batches)
        - SiLU activation (smooth, non-monotonic)
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 256, n_layers: int = 3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # Time embedding: sinusoidal + MLP
        self.time_embed = nn.Sequential(
            SinusoidalPositionEmbedding(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Input projection: z_t -> hidden
        self.input_proj = nn.Linear(embedding_dim, hidden_dim)
        
        # Residual blocks with GroupNorm
        self.blocks = nn.ModuleList([
            ResidualBlock(hidden_dim) for _ in range(n_layers)
        ])
        
        # Output projection: hidden -> embedding_dim
        self.output_proj = nn.Sequential(
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Predict velocity at state z_t and time t.
        
        Args:
            z_t: [batch_size, embedding_dim] current state
            t: [batch_size, 1] or [batch_size] time in [0, 1]
            
        Returns:
            v: [batch_size, embedding_dim] predicted velocity
        """
        # Ensure t has correct shape
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        
        # Time embedding
        t_emb = self.time_embed(t.squeeze(-1))  # [batch_size, hidden_dim]
        
        # Input projection
        h = self.input_proj(z_t)  # [batch_size, hidden_dim]
        
        # Add time embedding and pass through residual blocks
        h = h + t_emb
        for block in self.blocks:
            h = block(h)
        
        # Output projection
        v = self.output_proj(h)
        
        return v


class SinusoidalPositionEmbedding(nn.Module):
    """Sinusoidal position embedding for time encoding."""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
    
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t: [batch_size] time values in [0, 1]
            
        Returns:
            emb: [batch_size, dim] sinusoidal embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Frequency scaling
        emb = torch.log(torch.tensor(10000.0)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Scale time and compute sin/cos
        emb = t.unsqueeze(-1) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        
        return emb


class ResidualBlock(nn.Module):
    """Residual block with GroupNorm and SiLU activation."""
    
    def __init__(self, hidden_dim: int, groups: int = 8):
        super().__init__()
        self.block = nn.Sequential(
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(groups, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class VelocityNetSimple(nn.Module):
    """
    Simplified Velocity Network (concatenation-based).
    Faster but slightly less expressive than the full VelocityNet.
    
    Input: [z_t, t] concatenated
    Output: velocity v
    """
    
    def __init__(self, embedding_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embedding_dim + 1, hidden_dim),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, z_t: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z_t: [batch_size, embedding_dim]
            t: [batch_size, 1] or [batch_size]
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        x = torch.cat([z_t, t], dim=-1)
        return self.net(x)


class FlowMatcher(nn.Module):
    """
    Flow Matching trainer and sampler.
    
    Implements:
        - Linear interpolation path: z_t = (1-t)*z_0 + t*z_1
        - Conditional Flow Matching loss
        - ODE solvers for inference (Euler, RK4)
    """
    
    def __init__(self, velocity_net: nn.Module, sigma_min: float = 0.0):
        super().__init__()
        self.velocity_net = velocity_net
        self.sigma_min = sigma_min  # Minimum noise for stability
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample time uniformly from [0, 1]."""
        return torch.rand(batch_size, 1, device=device)
    
    def interpolate(self, z0: torch.Tensor, z1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Linear interpolation between z0 and z1.
        
        z_t = (1 - t) * z0 + t * z1
        
        Args:
            z0: [batch_size, dim] source (noisy social)
            z1: [batch_size, dim] target (clean preference)
            t: [batch_size, 1] time
            
        Returns:
            z_t: [batch_size, dim] interpolated state
        """
        return (1 - t) * z0 + t * z1
    
    def compute_target_velocity(self, z0: torch.Tensor, z1: torch.Tensor) -> torch.Tensor:
        """
        Compute target velocity for linear path.
        
        For linear interpolation: v* = z1 - z0 (constant velocity)
        """
        return z1 - z0
    
    def compute_flow_loss(self, z0: torch.Tensor, z1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute Flow Matching loss.
        
        Loss = E_t ||v_θ(z_t, t) - (z1 - z0)||^2
        
        Args:
            z0: [batch_size, dim] source embeddings (noisy)
            z1: [batch_size, dim] target embeddings (clean)
            
        Returns:
            loss: scalar flow matching loss
            z_t: [batch_size, dim] sampled intermediate state (for IB loss)
        """
        batch_size = z0.size(0)
        device = z0.device
        
        # Sample time
        t = self.sample_time(batch_size, device)
        
        # Interpolate
        z_t = self.interpolate(z0, z1, t)
        
        # Target velocity (constant for linear path)
        target_v = self.compute_target_velocity(z0, z1)
        
        # Predicted velocity
        pred_v = self.velocity_net(z_t, t)
        
        # MSE loss
        loss = F.mse_loss(pred_v, target_v)
        
        return loss, z_t
    
    @torch.no_grad()
    def euler_step(self, z: torch.Tensor, t: torch.Tensor, dt: float) -> torch.Tensor:
        """Single Euler integration step."""
        v = self.velocity_net(z, t)
        return z + v * dt
    
    @torch.no_grad()
    def rk4_step(self, z: torch.Tensor, t: torch.Tensor, dt: float) -> torch.Tensor:
        """Single RK4 (Runge-Kutta 4th order) integration step."""
        k1 = self.velocity_net(z, t)
        k2 = self.velocity_net(z + 0.5 * dt * k1, t + 0.5 * dt)
        k3 = self.velocity_net(z + 0.5 * dt * k2, t + 0.5 * dt)
        k4 = self.velocity_net(z + dt * k3, t + dt)
        return z + (dt / 6.0) * (k1 + 2*k2 + 2*k3 + k4)
    
    @torch.no_grad()
    def solve_ode(self, z0: torch.Tensor, n_steps: int = 10, 
                  method: str = 'euler') -> torch.Tensor:
        """
        Solve ODE from t=0 to t=1 to denoise embeddings.
        
        Args:
            z0: [batch_size, dim] initial state (noisy social embeddings)
            n_steps: number of integration steps
            method: 'euler' or 'rk4'
            
        Returns:
            z1: [batch_size, dim] final state (denoised embeddings)
        """
        dt = 1.0 / n_steps
        z = z0
        
        step_fn = self.euler_step if method == 'euler' else self.rk4_step
        
        for step in range(n_steps):
            t = torch.full((z.size(0), 1), step * dt, device=z.device)
            z = step_fn(z, t, dt)
        
        return z
    
    @torch.no_grad()
    def denoise(self, z0: torch.Tensor, n_steps: int = 10, 
                method: str = 'euler') -> torch.Tensor:
        """Alias for solve_ode for clarity."""
        return self.solve_ode(z0, n_steps, method)
    
    def forward_train(self, z0: torch.Tensor, z1: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training forward pass."""
        return self.compute_flow_loss(z0, z1)
    
    def forward_inference(self, z0: torch.Tensor, n_steps: int = 10) -> torch.Tensor:
        """Inference forward pass."""
        return self.denoise(z0, n_steps)


class AdaptiveFlowMatcher(FlowMatcher):
    """
    Adaptive Flow Matcher with importance sampling for time.
    
    Samples more time steps near t=0 and t=1 where the flow
    is typically harder to learn.
    """
    
    def __init__(self, velocity_net: nn.Module, sigma_min: float = 0.0, 
                 beta: float = 2.0):
        super().__init__(velocity_net, sigma_min)
        self.beta = beta  # Controls importance sampling strength
    
    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """
        Sample time with beta distribution for importance sampling.
        Beta(beta, beta) concentrates samples near 0.5 when beta > 1,
        and near 0 and 1 when beta < 1.
        """
        # Use beta distribution with alpha=beta for symmetric sampling
        dist = torch.distributions.Beta(self.beta, self.beta)
        t = dist.sample((batch_size, 1)).to(device)
        return t


def create_velocity_net(embedding_dim: int = 64, hidden_dim: int = 256, 
                        simple: bool = False) -> nn.Module:
    """Factory function to create velocity network."""
    if simple:
        return VelocityNetSimple(embedding_dim, hidden_dim)
    return VelocityNet(embedding_dim, hidden_dim)


def create_flow_matcher(embedding_dim: int = 64, hidden_dim: int = 256,
                        simple: bool = False, adaptive: bool = False) -> FlowMatcher:
    """Factory function to create flow matcher."""
    velocity_net = create_velocity_net(embedding_dim, hidden_dim, simple)
    
    if adaptive:
        return AdaptiveFlowMatcher(velocity_net)
    return FlowMatcher(velocity_net)
