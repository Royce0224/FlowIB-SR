"""
Evaluation Metrics for Recommendation
"""
import numpy as np
import torch


def recall_at_k(pred_items, true_items, k):
    """
    Recall@K: proportion of relevant items in top-k
    """
    if len(true_items) == 0:
        return 0.0
    
    hits = len(set(pred_items[:k]) & set(true_items))
    return hits / len(true_items)


def ndcg_at_k(pred_items, true_items, k):
    """
    NDCG@K: Normalized Discounted Cumulative Gain
    """
    if len(true_items) == 0:
        return 0.0
    
    dcg = 0.0
    for i, item in enumerate(pred_items[:k]):
        if item in true_items:
            dcg += 1.0 / np.log2(i + 2)
    
    idcg = sum([1.0 / np.log2(i + 2) for i in range(min(len(true_items), k))])
    
    return dcg / idcg if idcg > 0 else 0.0


def precision_at_k(pred_items, true_items, k):
    """
    Precision@K: proportion of top-k that are relevant
    """
    if k == 0:
        return 0.0
    
    hits = len(set(pred_items[:k]) & set(true_items))
    return hits / k


def evaluate_model(model, dataset, interact_edge_index, social_edge_index, 
                   device, k_list=[5, 10, 20]):
    """
    Evaluate model on test set
    
    Args:
        model: FlowIB model
        dataset: GBSRDataset
        interact_edge_index: interaction graph edges
        social_edge_index: social graph edges
        device: torch device
        k_list: list of k values for metrics
    
    Returns:
        metrics: dict of evaluation metrics
    """
    model.eval()
    
    test_dict = dataset.get_test_dict()
    train_mat = dataset.get_train_interactions()
    
    all_users = list(test_dict.keys())
    
    metrics = {f'Recall@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    metrics.update({f'Precision@{k}': [] for k in k_list})
    
    with torch.no_grad():
        # Get all embeddings
        user_emb, item_emb = model.forward(
            None, None, None,
            interact_edge_index, social_edge_index,
            mode='eval'
        )
        
        # Evaluate in batches
        batch_size = 256
        for start_idx in range(0, len(all_users), batch_size):
            end_idx = min(start_idx + batch_size, len(all_users))
            batch_users = all_users[start_idx:end_idx]
            
            # Compute scores
            batch_user_emb = user_emb[batch_users]
            scores = torch.mm(batch_user_emb, item_emb.t())  # [batch, n_items]
            
            # Mask training items
            for i, user in enumerate(batch_users):
                train_items = train_mat[user].nonzero()[1]
                scores[i, train_items] = -float('inf')
            
            # Get top-k items
            _, top_items = torch.topk(scores, max(k_list), dim=1)
            top_items = top_items.cpu().numpy()
            
            # Compute metrics for each user
            for i, user in enumerate(batch_users):
                true_items = test_dict[user]
                pred_items = top_items[i].tolist()
                
                for k in k_list:
                    metrics[f'Recall@{k}'].append(recall_at_k(pred_items, true_items, k))
                    metrics[f'NDCG@{k}'].append(ndcg_at_k(pred_items, true_items, k))
                    metrics[f'Precision@{k}'].append(precision_at_k(pred_items, true_items, k))
    
    # Average metrics
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
    
    return metrics


def evaluate_encoder(model, dataset, edge_index, device, k_list=[5, 10, 20], batch_size=512):
    """
    Evaluate encoder model on test set
    
    Args:
        model: Encoder model (e.g., PreferenceEncoder)
        dataset: GBSRDataset
        edge_index: interaction graph edges
        device: torch device
        k_list: list of k values for metrics
        batch_size: batch size for evaluation
    
    Returns:
        metrics: dict of evaluation metrics
    """
    model.eval()
    
    test_dict = dataset.get_test_dict()
    train_mat = dataset.get_train_interactions()
    
    all_users = list(test_dict.keys())
    
    metrics = {f'Recall@{k}': [] for k in k_list}
    metrics.update({f'NDCG@{k}': [] for k in k_list})
    metrics.update({f'Precision@{k}': [] for k in k_list})
    
    with torch.no_grad():
        # Get all embeddings
        user_emb, item_emb = model(edge_index)
        
        # Evaluate in batches
        for start_idx in range(0, len(all_users), batch_size):
            end_idx = min(start_idx + batch_size, len(all_users))
            batch_users = all_users[start_idx:end_idx]
            
            # Compute scores
            batch_user_emb = user_emb[batch_users]
            scores = torch.mm(batch_user_emb, item_emb.t())  # [batch, n_items]
            
            # Mask training items
            for i, user in enumerate(batch_users):
                train_items = train_mat[user].nonzero()[1]
                scores[i, train_items] = -float('inf')
            
            # Get top-k items
            _, top_items = torch.topk(scores, max(k_list), dim=1)
            top_items = top_items.cpu().numpy()
            
            # Compute metrics for each user
            for i, user in enumerate(batch_users):
                true_items = test_dict[user]
                pred_items = top_items[i].tolist()
                
                for k in k_list:
                    metrics[f'Recall@{k}'].append(recall_at_k(pred_items, true_items, k))
                    metrics[f'NDCG@{k}'].append(ndcg_at_k(pred_items, true_items, k))
                    metrics[f'Precision@{k}'].append(precision_at_k(pred_items, true_items, k))
    
    # Average metrics
    for key in metrics:
        metrics[key] = np.mean(metrics[key])
    
    return metrics
