"""
Encoders Module for Flow-IB Social Recommendation
Contains: PreferenceEncoder (LightGCN), SocialEncoder
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import LGConv


class PreferenceEncoder(nn.Module):
    """
    LightGCN-based Preference Encoder for user-item interaction graph.
    This is the "clean target" branch that learns from collaborative filtering signals.
    
    Architecture:
        - User/Item embeddings
        - Multi-layer LightGCN propagation
        - Layer-wise mean pooling
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64, n_layers: int = 3):
        super().__init__()
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Learnable embeddings
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        self.item_embedding = nn.Embedding(n_items, embedding_dim)
        
        # LightGCN convolution layers (parameter-free)
        self.convs = nn.ModuleList([LGConv() for _ in range(n_layers)])
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization for stable training"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.item_embedding.weight)
    
    def forward(self, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        """
        Forward pass through LightGCN.
        
        Args:
            edge_index: [2, num_edges] bipartite graph edges (user-item)
            edge_weight: [num_edges] optional edge weights
            
        Returns:
            user_emb: [n_users, embedding_dim] user embeddings
            item_emb: [n_items, embedding_dim] item embeddings
        """
        # Concatenate user and item embeddings
        x = torch.cat([self.user_embedding.weight, self.item_embedding.weight], dim=0)
        
        # Multi-layer propagation with residual connections
        all_emb = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            all_emb.append(x)
        
        # Layer-wise mean pooling (LightGCN aggregation)
        final_emb = torch.stack(all_emb, dim=0).mean(dim=0)
        
        # Split user and item embeddings
        user_emb = final_emb[:self.n_users]
        item_emb = final_emb[self.n_users:]
        
        return user_emb, item_emb
    
    def get_user_embeddings(self, edge_index: torch.Tensor, users: torch.Tensor = None):
        """Get user embeddings only (for efficiency)"""
        user_emb, _ = self.forward(edge_index)
        if users is not None:
            return user_emb[users]
        return user_emb
    
    def compute_bpr_loss(self, user_emb: torch.Tensor, item_emb: torch.Tensor,
                         users: torch.Tensor, pos_items: torch.Tensor, neg_items: torch.Tensor):
        """
        Compute BPR loss for preference learning.
        
        Args:
            user_emb: [n_users, dim] all user embeddings
            item_emb: [n_items, dim] all item embeddings
            users: [batch_size] user indices
            pos_items: [batch_size] positive item indices
            neg_items: [batch_size] negative item indices
            
        Returns:
            bpr_loss: scalar BPR loss
            reg_loss: scalar L2 regularization loss
        """
        user_e = user_emb[users]
        pos_e = item_emb[pos_items]
        neg_e = item_emb[neg_items]
        
        # BPR loss
        pos_scores = (user_e * pos_e).sum(dim=1)
        neg_scores = (user_e * neg_e).sum(dim=1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # L2 regularization on embeddings
        reg_loss = (
            self.user_embedding.weight[users].norm(2).pow(2) +
            self.item_embedding.weight[pos_items].norm(2).pow(2) +
            self.item_embedding.weight[neg_items].norm(2).pow(2)
        ) / users.size(0)
        
        return bpr_loss, reg_loss


class SocialEncoder(nn.Module):
    """
    Social Network Encoder for user-user social graph.
    This is the "noisy source" branch that captures social influence.
    
    Architecture:
        - User embeddings (separate from preference)
        - Multi-layer LightGCN propagation on social graph
        - Layer-wise mean pooling
    """
    
    def __init__(self, n_users: int, embedding_dim: int = 64, n_layers: int = 2):
        super().__init__()
        self.n_users = n_users
        self.embedding_dim = embedding_dim
        self.n_layers = n_layers
        
        # Learnable user embeddings for social network
        self.user_embedding = nn.Embedding(n_users, embedding_dim)
        
        # LightGCN convolution layers
        self.convs = nn.ModuleList([LGConv() for _ in range(n_layers)])
        
        self._init_weights()
    
    def _init_weights(self):
        """Xavier initialization"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
    
    def forward(self, edge_index: torch.Tensor, edge_weight: torch.Tensor = None):
        """
        Forward pass through social GCN.
        
        Args:
            edge_index: [2, num_edges] social network edges (user-user)
            edge_weight: [num_edges] optional edge weights
            
        Returns:
            social_emb: [n_users, embedding_dim] social embeddings (noisy)
        """
        x = self.user_embedding.weight
        
        # Multi-layer propagation
        all_emb = [x]
        for conv in self.convs:
            x = conv(x, edge_index, edge_weight)
            all_emb.append(x)
        
        # Layer-wise mean pooling
        social_emb = torch.stack(all_emb, dim=0).mean(dim=0)
        
        return social_emb
    
    def get_user_embeddings(self, edge_index: torch.Tensor, users: torch.Tensor = None):
        """Get user embeddings for specific users"""
        social_emb = self.forward(edge_index)
        if users is not None:
            return social_emb[users]
        return social_emb


class DualEncoder(nn.Module):
    """
    Combined Dual Encoder that wraps both Preference and Social encoders.
    Useful for Stage 1 pre-training and inference.
    """
    
    def __init__(self, n_users: int, n_items: int, embedding_dim: int = 64,
                 n_layers_interact: int = 3, n_layers_social: int = 2):
        super().__init__()
        
        self.preference_encoder = PreferenceEncoder(
            n_users, n_items, embedding_dim, n_layers_interact
        )
        self.social_encoder = SocialEncoder(
            n_users, embedding_dim, n_layers_social
        )
    
    def forward(self, interact_edge_index: torch.Tensor, social_edge_index: torch.Tensor):
        """
        Get embeddings from both encoders.
        
        Returns:
            user_pref: [n_users, dim] preference embeddings (clean)
            item_emb: [n_items, dim] item embeddings
            social_emb: [n_users, dim] social embeddings (noisy)
        """
        user_pref, item_emb = self.preference_encoder(interact_edge_index)
        social_emb = self.social_encoder(social_edge_index)
        
        return user_pref, item_emb, social_emb
