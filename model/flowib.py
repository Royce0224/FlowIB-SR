"""
Flow-IB Social Recommendation Model
Combines Flow Matching (RecFlow ICML 2025) with Information Bottleneck (GBSR KDD 2024)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import LGConv


class LightGCNEncoder(nn.Module):
    """LightGCN-based encoder for social and interaction graphs"""
    
    def __init__(self, n_users, n_items, embedding_dim, n_layers=3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # LightGCN convolution layers
        self.convs = nn.ModuleList([LGConv() for _ in range(n_layers)])
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, edge_index, edge_weight=None):
        """
        Args:
            edge_index: [2, num_edges] edge indices
            edge_weight: [num_edges] edge weights
        Returns:
            user_emb: [n_users, embedding_dim]
            item_emb: [n_items, embedding_dim]
        """
        # Initial embeddings
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        # Multi-layer propagation
        all_emb = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            all_emb.append(x)
        
        # Average pooling
        final_emb = torch.stack(all_emb, dim=0).mean(dim=0)
        
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        
        return user_emb, item_emb


class SocialEncoder(nn.Module):
    """Social network encoder (noisy branch)"""
    
    def __init__(self, n_users, embedding_dim, n_layers=2):
        super().__init__()
        self.n_users = n_users
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.convs = nn.ModuleList([LGConv() for _ in range(n_layers)])
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.xavier_uniform_(self.user_embedding.weight)
    
    def forward(self, edge_index, edge_weight=None):
        """
        Args:
            edge_index: [2, num_edges] social network edges
        Returns:
            social_emb: [n_users, embedding_dim] noisy social embeddings
        """
        x = self.user_embedding.weight
        
        all_emb = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            all_emb.append(x)
        
        social_emb = torch.stack(all_emb, dim=0).mean(dim=0)
        return social_emb


class FlowVelocityNetwork(nn.Module):
    """Flow matching velocity network v_θ(z_t, t)"""
    
    def __init__(self, embedding_dim, hidden_dim=256):
        super().__init__()
        self.embedding_dim = embedding_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Velocity network
        self.velocity_net = nn.Sequential(
            nn.Linear(embedding_dim + hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, embedding_dim)
        )
    
    def forward(self, z_t, t):
        """
        Args:
            z_t: [batch_size, embedding_dim] current state
            t: [batch_size, 1] time
        Returns:
            v: [batch_size, embedding_dim] velocity
        """
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Concatenate and predict velocity
        x = torch.cat([z_t, t_emb], dim=-1)
        v = self.velocity_net(x)
        
        return v


class HSICLoss(nn.Module):
    """HSIC (Hilbert-Schmidt Independence Criterion) for IB constraint"""
    
    def __init__(self, sigma=1.0):
        super().__init__()
        self.sigma = sigma
    
    def gaussian_kernel(self, X, Y):
        """Compute Gaussian kernel matrix"""
        # X: [n, d], Y: [m, d]
        XX = torch.sum(X * X, dim=1, keepdim=True)  # [n, 1]
        YY = torch.sum(Y * Y, dim=1, keepdim=True)  # [m, 1]
        XY = torch.mm(X, Y.t())  # [n, m]
        
        dist = XX + YY.t() - 2 * XY
        K = torch.exp(-dist / (2 * self.sigma ** 2))
        return K
    
    def forward(self, X, Y):
        """
        Compute HSIC between X and Y
        Args:
            X: [batch_size, dim_x]
            Y: [batch_size, dim_y]
        Returns:
            hsic: scalar
        """
        n = X.size(0)
        
        # Compute kernel matrices
        K = self.gaussian_kernel(X, X)
        L = self.gaussian_kernel(Y, Y)
        
        # Center the kernel matrices
        H = torch.eye(n, device=X.device) - torch.ones(n, n, device=X.device) / n
        K_c = torch.mm(torch.mm(H, K), H)
        L_c = torch.mm(torch.mm(H, L), H)
        
        # Compute HSIC
        hsic = torch.trace(torch.mm(K_c, L_c)) / ((n - 1) ** 2)
        
        return hsic


class FlowIBModel(nn.Module):
    """
    Flow-IB Social Recommendation Model
    
    Architecture:
    1. Initial Encoder: Social (noisy) + Preference (clean) branches
    2. Flow Velocity Network: Learns denoising trajectory
    3. IB Controller: HSIC-based regularization
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64, 
                 n_layers_interact=3, n_layers_social=2,
                 hidden_dim=256, sigma=1.0):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # 1. Initial Encoders
        # Preference branch (clean target)
        self.preference_encoder = LightGCNEncoder(
            n_users, n_items, embedding_dim, n_layers_interact
        )
        
        # Social branch (noisy source)
        self.social_encoder = SocialEncoder(
            n_users, embedding_dim, n_layers_social
        )
        
        # 2. Flow Velocity Network
        self.velocity_net = FlowVelocityNetwork(embedding_dim, hidden_dim)
        
        # 3. IB Controller
        self.hsic_loss = HSICLoss(sigma)
    
    def get_preference_embeddings(self, interact_edge_index):
        """Get clean preference embeddings (target)"""
        user_pref, item_emb = self.preference_encoder(interact_edge_index)
        return user_pref, item_emb
    
    def get_social_embeddings(self, social_edge_index):
        """Get noisy social embeddings (source)"""
        social_emb = self.social_encoder(social_edge_index)
        return social_emb
    
    def sample_flow_path(self, z0, z1, batch_size):
        """
        Sample points along the flow path: z_t = (1-t)*z0 + t*z1
        Args:
            z0: [batch_size, dim] noisy social embeddings
            z1: [batch_size, dim] clean preference embeddings
        Returns:
            t: [batch_size, 1] sampled time
            z_t: [batch_size, dim] interpolated state
            target_v: [batch_size, dim] target velocity (z1 - z0)
        """
        # Sample time uniformly
        t = torch.rand(batch_size, 1, device=z0.device)
        
        # Linear interpolation
        z_t = (1 - t) * z0 + t * z1
        
        # Target velocity (derivative of linear path)
        target_v = z1 - z0
        
        return t, z_t, target_v
    
    def compute_flow_loss(self, z0, z1, batch_size):
        """
        Flow Matching Loss: ||v_θ(z_t, t) - (z1 - z0)||^2
        """
        t, z_t, target_v = self.sample_flow_path(z0, z1, batch_size)
        
        # Predict velocity
        pred_v = self.velocity_net(z_t, t)
        
        # MSE loss
        flow_loss = F.mse_loss(pred_v, target_v)
        
        return flow_loss, z_t, t
    
    def compute_ib_loss(self, z_t, social_emb):
        """
        Information Bottleneck Loss: HSIC(z_t, social_graph)
        Minimize mutual information between denoised embedding and original social structure
        """
        ib_loss = self.hsic_loss(z_t, social_emb)
        return ib_loss
    
    def compute_bpr_loss(self, user_emb, item_emb, users, pos_items, neg_items):
        """
        BPR Loss for recommendation task
        """
        user_emb_batch = user_emb[users]
        pos_emb = item_emb[pos_items]
        neg_emb = item_emb[neg_items]
        
        pos_scores = torch.sum(user_emb_batch * pos_emb, dim=1)
        neg_scores = torch.sum(user_emb_batch * neg_emb, dim=1)
        
        bpr_loss = -torch.mean(F.logsigmoid(pos_scores - neg_scores))
        
        return bpr_loss
    
    def denoise(self, z0, n_steps=10):
        """
        Inference: Denoise social embeddings using learned flow
        Args:
            z0: [batch_size, dim] noisy social embeddings
            n_steps: number of ODE steps
        Returns:
            z1: [batch_size, dim] denoised embeddings
        """
        z_t = z0
        dt = 1.0 / n_steps
        
        for step in range(n_steps):
            t = torch.ones(z_t.size(0), 1, device=z_t.device) * (step * dt)
            v = self.velocity_net(z_t, t)
            z_t = z_t + v * dt
        
        return z_t
    
    def forward(self, users, pos_items, neg_items, 
                interact_edge_index, social_edge_index, mode='train'):
        """
        Forward pass
        Args:
            users: [batch_size] user indices
            pos_items: [batch_size] positive item indices
            neg_items: [batch_size] negative item indices
            interact_edge_index: [2, num_edges] interaction graph
            social_edge_index: [2, num_edges] social graph
            mode: 'train' or 'eval'
        """
        # 1. Get embeddings
        user_pref, item_emb = self.get_preference_embeddings(interact_edge_index)
        social_emb = self.get_social_embeddings(social_edge_index)
        
        if mode == 'train':
            batch_size = users.size(0)
            # Training: compute all losses
            z0 = social_emb[users]  # noisy
            z1 = user_pref[users]   # clean target
            
            # Flow Matching Loss
            flow_loss, z_t, t = self.compute_flow_loss(z0, z1, batch_size)
            
            # Information Bottleneck Loss
            ib_loss = self.compute_ib_loss(z_t, z0)
            
            # BPR Loss (use denoised embeddings)
            denoised_user_emb = self.denoise(social_emb, n_steps=5)
            bpr_loss = self.compute_bpr_loss(denoised_user_emb, item_emb, 
                                            users, pos_items, neg_items)
            
            return {
                'flow_loss': flow_loss,
                'ib_loss': ib_loss,
                'bpr_loss': bpr_loss
            }
        
        else:
            # Evaluation: denoise and return embeddings
            denoised_user_emb = self.denoise(social_emb, n_steps=10)
            return denoised_user_emb, item_emb
    
    def predict(self, users, interact_edge_index, social_edge_index):
        """
        Predict scores for all items for given users
        """
        with torch.no_grad():
            user_emb, item_emb = self.forward(
                users, None, None, 
                interact_edge_index, social_edge_index, 
                mode='eval'
            )
            scores = torch.mm(user_emb[users], item_emb.t())
        return scores
