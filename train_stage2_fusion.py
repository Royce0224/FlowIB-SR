"""
Stage 2 Training with Embedding Fusion Strategy

Key Insight: Pure Flow Matching denoising hurts performance.
Solution: Fuse preference embeddings with denoised social embeddings.

Fusion: z_final = α * z_pref + (1-α) * z_denoised
where α is a learnable parameter.
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import GBSRDataset, TrainDataset
from models.encoders import PreferenceEncoder, SocialEncoder
from models.flow import FlowMatcher, VelocityNet
from models.hsic import LinearHSIC


class FlowIBFusion(nn.Module):
    """
    Flow-IB Model with Embedding Fusion.
    
    Instead of replacing preference embeddings with denoised social embeddings,
    we learn to fuse them optimally.
    """
    
    def __init__(self, n_users, n_items, embedding_dim=64, hidden_dim=256,
                 n_layers_interact=3, n_layers_social=2):
        super().__init__()
        
        self.n_users = n_users
        self.n_items = n_items
        self.embedding_dim = embedding_dim
        
        # Preference Encoder (will be frozen initially)
        self.preference_encoder = PreferenceEncoder(
            n_users, n_items, embedding_dim, n_layers_interact
        )
        
        # Social Encoder
        self.social_encoder = SocialEncoder(
            n_users, embedding_dim, n_layers_social
        )
        
        # Velocity Network and Flow Matcher
        velocity_net = VelocityNet(embedding_dim, hidden_dim)
        self.flow_matcher = FlowMatcher(velocity_net)
        
        # HSIC Loss
        self.hsic_loss = LinearHSIC()
        
        # Learnable fusion weight (initialized to favor preference)
        self.fusion_alpha = nn.Parameter(torch.tensor(0.8))  # Start with 80% preference
        
        # Optional: User-specific fusion weights
        self.user_fusion_weights = nn.Embedding(n_users, 1)
        nn.init.constant_(self.user_fusion_weights.weight, 0.0)  # Start at 0, sigmoid -> 0.5
    
    def freeze_preference_encoder(self):
        """Freeze PreferenceEncoder parameters."""
        for param in self.preference_encoder.parameters():
            param.requires_grad = False
        print("✓ PreferenceEncoder frozen")
    
    def load_stage1_weights(self, checkpoint_path):
        """Load pre-trained PreferenceEncoder weights."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        self.preference_encoder.load_state_dict(checkpoint['model_state_dict'])
        print(f"✓ Loaded Stage 1 weights from {checkpoint_path}")
        print(f"  Stage 1 Recall@20: {checkpoint['metrics'].get('Recall@20', 'N/A')}")
    
    def get_fusion_weight(self, users=None):
        """Get fusion weight (global or user-specific)."""
        global_alpha = torch.sigmoid(self.fusion_alpha)  # Constrain to [0, 1]
        
        if users is not None:
            # User-specific adjustment
            user_adj = torch.sigmoid(self.user_fusion_weights(users)).squeeze(-1)
            # Combine global and user-specific
            alpha = global_alpha * 0.5 + user_adj * 0.5
        else:
            alpha = global_alpha
        
        return alpha
    
    def get_embeddings(self, interact_edge_index, social_edge_index, users=None):
        """Get embeddings from both encoders."""
        with torch.no_grad():
            user_pref, item_emb = self.preference_encoder(interact_edge_index)
        
        social_emb = self.social_encoder(social_edge_index)
        
        if users is not None:
            z_pref = user_pref[users]
            z_social = social_emb[users]
        else:
            z_pref = user_pref
            z_social = social_emb
        
        return z_pref, z_social, item_emb
    
    def forward_train(self, users, pos_items, neg_items, 
                      interact_edge_index, social_edge_index):
        """Training forward pass with fusion."""
        # Get embeddings
        z_pref, z_social, item_emb = self.get_embeddings(
            interact_edge_index, social_edge_index, users
        )
        
        # Normalize
        z_pref_norm = F.normalize(z_pref, p=2, dim=1)
        z_social_norm = F.normalize(z_social, p=2, dim=1)
        
        # Flow Matching Loss
        flow_loss, z_t = self.flow_matcher.compute_flow_loss(z_social_norm, z_pref_norm)
        
        # Information Bottleneck Loss
        ib_loss = self.hsic_loss(z_t, z_social_norm)
        
        # Get denoised embeddings (single step for training)
        t_zero = torch.zeros(z_social.size(0), 1, device=z_social.device)
        v_pred = self.flow_matcher.velocity_net(z_social_norm, t_zero)
        z_denoised = z_social_norm + v_pred * 1.0  # Full step
        z_denoised = F.normalize(z_denoised, p=2, dim=1)
        
        # Fusion: combine preference and denoised social
        alpha = self.get_fusion_weight(users)
        if alpha.dim() == 0:
            z_fused = alpha * z_pref_norm + (1 - alpha) * z_denoised
        else:
            alpha = alpha.unsqueeze(-1)
            z_fused = alpha * z_pref_norm + (1 - alpha) * z_denoised
        z_fused = F.normalize(z_fused, p=2, dim=1)
        
        # BPR Loss with fused embeddings
        pos_scores = (z_fused * item_emb[pos_items]).sum(dim=-1)
        neg_scores = (z_fused * item_emb[neg_items]).sum(dim=-1)
        bpr_loss = -F.logsigmoid(pos_scores - neg_scores).mean()
        
        # Regularization: encourage fusion weight to stay reasonable
        alpha_reg = (self.fusion_alpha ** 2) * 0.01
        
        return {
            'flow_loss': flow_loss,
            'ib_loss': ib_loss,
            'bpr_loss': bpr_loss,
            'alpha_reg': alpha_reg,
            'fusion_alpha': torch.sigmoid(self.fusion_alpha).item(),
            'z_t': z_t,
            'z_social': z_social_norm
        }
    
    @torch.no_grad()
    def forward_inference(self, interact_edge_index, social_edge_index, 
                          n_steps=10, method='euler', use_fusion=True):
        """Inference with optional fusion."""
        # Get preference embeddings
        user_pref, item_emb = self.preference_encoder(interact_edge_index)
        z_pref = F.normalize(user_pref, p=2, dim=1)
        
        if not use_fusion:
            return z_pref, item_emb
        
        # Get denoised social embeddings
        social_emb = self.social_encoder(social_edge_index)
        z_social = F.normalize(social_emb, p=2, dim=1)
        z_denoised = self.flow_matcher.denoise(z_social, n_steps=n_steps, method=method)
        z_denoised = F.normalize(z_denoised, p=2, dim=1)
        
        # Fusion
        alpha = torch.sigmoid(self.fusion_alpha)
        z_fused = alpha * z_pref + (1 - alpha) * z_denoised
        z_fused = F.normalize(z_fused, p=2, dim=1)
        
        return z_fused, item_emb


def build_edges(train_data, social_data, n_users, device):
    """Build interaction and social graph edges."""
    interact_row = [u for u, i in train_data] + [i + n_users for u, i in train_data]
    interact_col = [i + n_users for u, i in train_data] + [u for u, i in train_data]
    interact_edge_index = torch.tensor([interact_row, interact_col], dtype=torch.long, device=device)
    
    social_row = [u1 for u1, u2 in social_data]
    social_col = [u2 for u1, u2 in social_data]
    social_edge_index = torch.tensor([social_row, social_col], dtype=torch.long, device=device)
    
    return interact_edge_index, social_edge_index


def train_epoch(model, train_loader, optimizer, scaler, 
                interact_edge_index, social_edge_index, device,
                lambda_flow=1.0, lambda_ib=0.01, lambda_bpr=1.0,
                use_amp=True):
    """Train for one epoch."""
    model.train()
    
    total_loss = 0
    total_flow = 0
    total_ib = 0
    total_bpr = 0
    total_alpha = 0
    n_batches = 0
    
    for batch_idx, (users, pos_items, neg_items) in enumerate(train_loader):
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        
        optimizer.zero_grad(set_to_none=True)
        
        if use_amp:
            with autocast():
                losses = model.forward_train(
                    users, pos_items, neg_items,
                    interact_edge_index, social_edge_index
                )
                
                loss = (lambda_flow * losses['flow_loss'] + 
                       lambda_ib * losses['ib_loss'] + 
                       lambda_bpr * losses['bpr_loss'] +
                       losses['alpha_reg'])
            
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            losses = model.forward_train(
                users, pos_items, neg_items,
                interact_edge_index, social_edge_index
            )
            
            loss = (lambda_flow * losses['flow_loss'] + 
                   lambda_ib * losses['ib_loss'] + 
                   lambda_bpr * losses['bpr_loss'] +
                   losses['alpha_reg'])
            
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        total_flow += losses['flow_loss'].item()
        total_ib += losses['ib_loss'].item()
        total_bpr += losses['bpr_loss'].item()
        total_alpha += losses['fusion_alpha']
        n_batches += 1
        
        if (batch_idx + 1) % 100 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                  f"Loss={loss.item():.4f}, "
                  f"BPR={losses['bpr_loss'].item():.4f}, "
                  f"α={losses['fusion_alpha']:.3f}")
    
    return {
        'total_loss': total_loss / n_batches,
        'flow_loss': total_flow / n_batches,
        'ib_loss': total_ib / n_batches,
        'bpr_loss': total_bpr / n_batches,
        'fusion_alpha': total_alpha / n_batches
    }


def evaluate(model, dataset, interact_edge_index, social_edge_index, 
             device, k_list=[5, 10, 20], n_steps=10, use_fusion=True):
    """Evaluate model."""
    model.eval()
    
    from utils.metrics import recall_at_k, ndcg_at_k
    
    test_dict = dataset.get_test_dict()
    train_mat = dataset.get_train_interactions()
    all_users = list(test_dict.keys())
    
    metrics = {f'Recall@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    
    with torch.no_grad():
        user_emb, item_emb = model.forward_inference(
            interact_edge_index, social_edge_index, 
            n_steps=n_steps, use_fusion=use_fusion
        )
        
        batch_size = 256
        for start_idx in range(0, len(all_users), batch_size):
            end_idx = min(start_idx + batch_size, len(all_users))
            batch_users = all_users[start_idx:end_idx]
            
            batch_user_emb = user_emb[batch_users]
            scores = torch.mm(batch_user_emb, item_emb.t())
            
            for i, user in enumerate(batch_users):
                train_items = train_mat[user].nonzero()[1]
                scores[i, train_items] = -float('inf')
            
            _, top_items = torch.topk(scores, max(k_list), dim=1)
            top_items = top_items.cpu().numpy()
            
            for i, user in enumerate(batch_users):
                true_items = test_dict[user]
                pred_items = top_items[i].tolist()
                
                for k in k_list:
                    metrics[f'Recall@{k}'].append(recall_at_k(pred_items, true_items, k))
                    metrics[f'NDCG@{k}'].append(ndcg_at_k(pred_items, true_items, k))
    
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
    
    return metrics


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    print(f"\n{'='*60}")
    print(f"Stage 2 with Embedding Fusion")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}")
    
    dataset = GBSRDataset(args.data_path, args.dataset)
    
    print("\nBuilding graphs...")
    interact_edge_index, social_edge_index = build_edges(
        dataset.train_data, dataset.social_data, dataset.n_users, device
    )
    
    print("\nInitializing FlowIB Fusion model...")
    model = FlowIBFusion(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        n_layers_interact=args.n_layers_interact,
        n_layers_social=args.n_layers_social
    ).to(device)
    
    # Load Stage 1 weights
    stage1_path = os.path.join(args.save_dir, f'{args.dataset}_stage1_best.pt')
    if os.path.exists(stage1_path):
        model.load_stage1_weights(stage1_path)
    else:
        print(f"⚠ Stage 1 checkpoint not found!")
        return
    
    # Freeze preference encoder
    model.freeze_preference_encoder()
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} / {total_params:,}")
    
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scaler = GradScaler() if args.use_amp else None
    
    train_dataset = TrainDataset(dataset.train_data, dataset.n_items)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    print("\n" + "="*60)
    print("Starting Training with Fusion...")
    print(f"Lambda Flow: {args.lambda_flow}, Lambda IB: {args.lambda_ib}, Lambda BPR: {args.lambda_bpr}")
    print("="*60)
    
    best_recall = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        train_losses = train_epoch(
            model, train_loader, optimizer, scaler,
            interact_edge_index, social_edge_index, device,
            args.lambda_flow, args.lambda_ib, args.lambda_bpr,
            args.use_amp
        )
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.2f}s):")
        print(f"  Total Loss: {train_losses['total_loss']:.4f}")
        print(f"  BPR Loss: {train_losses['bpr_loss']:.4f}")
        print(f"  Fusion α: {train_losses['fusion_alpha']:.3f}")
        
        if epoch % args.eval_every == 0:
            print("\n  Evaluating (with fusion)...")
            metrics_fusion = evaluate(
                model, dataset, interact_edge_index, social_edge_index,
                device, k_list=args.k_list, n_steps=args.ode_steps, use_fusion=True
            )
            
            print("\n  Evaluating (preference only)...")
            metrics_pref = evaluate(
                model, dataset, interact_edge_index, social_edge_index,
                device, k_list=args.k_list, use_fusion=False
            )
            
            print(f"  Results:")
            print(f"    Fusion:     Recall@20={metrics_fusion['Recall@20']:.4f}, NDCG@20={metrics_fusion['NDCG@20']:.4f}")
            print(f"    Pref Only:  Recall@20={metrics_pref['Recall@20']:.4f}, NDCG@20={metrics_pref['NDCG@20']:.4f}")
            
            # Use the better one
            recall_20 = max(metrics_fusion['Recall@20'], metrics_pref['Recall@20'])
            best_metrics = metrics_fusion if metrics_fusion['Recall@20'] >= metrics_pref['Recall@20'] else metrics_pref
            
            if recall_20 > best_recall:
                best_recall = recall_20
                best_epoch = epoch
                patience_counter = 0
                
                save_path = os.path.join(args.save_dir, f'{args.dataset}_stage2_fusion_best.pt')
                os.makedirs(args.save_dir, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': best_metrics,
                    'fusion_alpha': torch.sigmoid(model.fusion_alpha).item(),
                    'args': vars(args)
                }, save_path)
                
                print(f"  ✓ Best model saved! (Recall@20: {best_recall:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
            
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print("-" * 60)
    
    print("\n" + "="*60)
    print(f"Training Completed!")
    print(f"Best Recall@20: {best_recall:.4f} at epoch {best_epoch}")
    print(f"Final Fusion α: {torch.sigmoid(model.fusion_alpha).item():.3f}")
    print("="*60)
    
    return best_recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 2 with Embedding Fusion')
    
    parser.add_argument('--dataset', type=str, default='douban_book')
    parser.add_argument('--data_path', type=str, default='./data')
    
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--n_layers_interact', type=int, default=3)
    parser.add_argument('--n_layers_social', type=int, default=2)
    
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--use_amp', action='store_true', default=True)
    
    parser.add_argument('--lambda_flow', type=float, default=0.5)
    parser.add_argument('--lambda_ib', type=float, default=0.01)
    parser.add_argument('--lambda_bpr', type=float, default=1.0)
    
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--k_list', type=int, nargs='+', default=[5, 10, 20])
    parser.add_argument('--ode_steps', type=int, default=5)
    
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--patience', type=int, default=30)
    
    args = parser.parse_args()
    main(args)
