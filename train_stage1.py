"""
Stage 1 Training: Pre-train PreferenceEncoder (LightGCN)

This stage trains the preference encoder on the interaction graph alone,
establishing a "clean target" embedding space for Flow Matching.

Goal: Achieve baseline Recall without social information.
Output: Saved encoder weights as the "North Star" coordinates.
"""
import os
import sys
import argparse
import time
import numpy as np
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils.data_loader import GBSRDataset, TrainDataset
from models.encoders import PreferenceEncoder
from utils.metrics import evaluate_encoder


def build_interact_edges(train_data, n_users, device):
    """Build bidirectional user-item interaction edges."""
    # User -> Item edges (offset item indices by n_users)
    row = [u for u, i in train_data] + [i + n_users for u, i in train_data]
    col = [i + n_users for u, i in train_data] + [u for u, i in train_data]
    
    edge_index = torch.tensor([row, col], dtype=torch.long, device=device)
    return edge_index


def train_epoch(model, train_loader, optimizer, edge_index, device, reg_weight=1e-5):
    """Train for one epoch with BPR loss."""
    model.train()
    
    total_loss = 0
    total_bpr = 0
    total_reg = 0
    n_batches = 0
    
    for batch_idx, (users, pos_items, neg_items) in enumerate(train_loader):
        users = users.to(device)
        pos_items = pos_items.to(device)
        neg_items = neg_items.to(device)
        
        # Forward pass
        user_emb, item_emb = model(edge_index)
        
        # Compute BPR loss
        bpr_loss, reg_loss = model.compute_bpr_loss(
            user_emb, item_emb, users, pos_items, neg_items
        )
        
        loss = bpr_loss + reg_weight * reg_loss
        
        # Backward pass
        optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_bpr += bpr_loss.item()
        total_reg += reg_loss.item()
        n_batches += 1
        
        if (batch_idx + 1) % 200 == 0:
            print(f"  Batch {batch_idx + 1}/{len(train_loader)}: "
                  f"Loss={loss.item():.4f}, BPR={bpr_loss.item():.4f}")
    
    return {
        'total_loss': total_loss / n_batches,
        'bpr_loss': total_bpr / n_batches,
        'reg_loss': total_reg / n_batches
    }


def main(args):
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print(f"\n{'='*60}")
    print(f"Stage 1: Pre-training PreferenceEncoder")
    print(f"Dataset: {args.dataset}")
    print(f"{'='*60}")
    
    dataset = GBSRDataset(args.data_path, args.dataset)
    
    # Build interaction graph
    print("\nBuilding interaction graph...")
    edge_index = build_interact_edges(
        dataset.train_data, dataset.n_users, device
    )
    print(f"Interaction edges: {edge_index.size(1)}")
    
    # Create model
    print("\nInitializing PreferenceEncoder (LightGCN)...")
    model = PreferenceEncoder(
        n_users=dataset.n_users,
        n_items=dataset.n_items,
        embedding_dim=args.embedding_dim,
        n_layers=args.n_layers
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {n_params:,}")
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Data loader
    train_dataset = TrainDataset(dataset.train_data, dataset.n_items)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Training loop
    print("\n" + "="*60)
    print("Starting Stage 1 Training...")
    print("="*60)
    
    best_recall = 0
    best_epoch = 0
    patience_counter = 0
    
    for epoch in range(1, args.epochs + 1):
        start_time = time.time()
        
        # Train
        train_losses = train_epoch(
            model, train_loader, optimizer, edge_index, device, args.reg_weight
        )
        
        epoch_time = time.time() - start_time
        
        print(f"\nEpoch {epoch}/{args.epochs} ({epoch_time:.2f}s):")
        print(f"  Loss: {train_losses['total_loss']:.4f}")
        print(f"  BPR: {train_losses['bpr_loss']:.4f}")
        print(f"  Reg: {train_losses['reg_loss']:.6f}")
        
        # Evaluate
        if epoch % args.eval_every == 0:
            print("\n  Evaluating...")
            eval_start = time.time()
            
            metrics = evaluate_encoder(
                model, dataset, edge_index, device, k_list=args.k_list
            )
            
            eval_time = time.time() - eval_start
            
            print(f"  Evaluation ({eval_time:.2f}s):")
            for k in args.k_list:
                print(f"    Recall@{k}: {metrics[f'Recall@{k}']:.4f}, "
                      f"NDCG@{k}: {metrics[f'NDCG@{k}']:.4f}")
            
            # Update scheduler
            recall_20 = metrics['Recall@20']
            scheduler.step(recall_20)
            
            # Save best model
            if recall_20 > best_recall:
                best_recall = recall_20
                best_epoch = epoch
                patience_counter = 0
                
                # Save checkpoint
                save_path = os.path.join(args.save_dir, f'{args.dataset}_stage1_best.pt')
                os.makedirs(args.save_dir, exist_ok=True)
                
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metrics': metrics,
                    'args': vars(args)
                }, save_path)
                
                print(f"  ✓ Best model saved! (Recall@20: {best_recall:.4f})")
            else:
                patience_counter += 1
                print(f"  No improvement. Patience: {patience_counter}/{args.patience}")
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
        
        print("-" * 60)
    
    # Final summary
    print("\n" + "="*60)
    print(f"Stage 1 Training Completed!")
    print(f"Best Recall@20: {best_recall:.4f} at epoch {best_epoch}")
    print(f"Model saved to: {os.path.join(args.save_dir, f'{args.dataset}_stage1_best.pt')}")
    print("="*60)
    
    return best_recall


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Stage 1: Pre-train PreferenceEncoder')
    
    # Dataset
    parser.add_argument('--dataset', type=str, default='douban_book',
                       choices=['douban_book', 'epinions', 'yelp'])
    parser.add_argument('--data_path', type=str, default='./data')
    
    # Model
    parser.add_argument('--embedding_dim', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=3)
    
    # Training
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=4096)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--reg_weight', type=float, default=1e-5)
    
    # Evaluation
    parser.add_argument('--eval_every', type=int, default=5)
    parser.add_argument('--k_list', type=int, nargs='+', default=[5, 10, 20])
    
    # Other
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_dir', type=str, default='./checkpoints')
    parser.add_argument('--patience', type=int, default=30)
    
    args = parser.parse_args()
    main(args)
