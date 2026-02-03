"""
Data Loader for GBSR Datasets
Supports: douban_book, epinions, yelp
"""
import os
import numpy as np
import scipy.sparse as sp
import torch
from torch.utils.data import Dataset, DataLoader


class GBSRDataset:
    """Load and process GBSR dataset"""
    
    def __init__(self, data_path, dataset_name):
        self.data_path = data_path
        self.dataset_name = dataset_name
        self.load_data()
        
    def load_data(self):
        """Load train/test data and social network"""
        data_dir = os.path.join(self.data_path, self.dataset_name)
        
        # Load training data
        train_file = os.path.join(data_dir, 'train.txt')
        self.train_data = self._load_rating_file(train_file)
        
        # Load test data
        test_file = os.path.join(data_dir, 'test.txt')
        self.test_data = self._load_rating_file(test_file)
        
        # Load social network
        social_file = os.path.join(data_dir, 'trust.txt')
        self.social_data = self._load_social_file(social_file)
        
        # Get statistics
        self.n_users = max(max([u for u, _ in self.train_data]), 
                          max([u for u, _ in self.test_data])) + 1
        self.n_items = max(max([i for _, i in self.train_data]), 
                          max([i for _, i in self.test_data])) + 1
        
        print(f"Dataset: {self.dataset_name}")
        print(f"Users: {self.n_users}, Items: {self.n_items}")
        print(f"Train interactions: {len(self.train_data)}")
        print(f"Test interactions: {len(self.test_data)}")
        print(f"Social links: {len(self.social_data)}")
        
    def _load_rating_file(self, filename):
        """Load user-item interactions"""
        data = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user = int(parts[0])
                    item = int(parts[1])
                    data.append((user, item))
        return data
    
    def _load_social_file(self, filename):
        """Load social network"""
        data = []
        with open(filename, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    user1 = int(parts[0])
                    user2 = int(parts[1])
                    data.append((user1, user2))
        return data
    
    def get_train_interactions(self):
        """Get training user-item matrix"""
        row = [u for u, i in self.train_data]
        col = [i for u, i in self.train_data]
        data = np.ones(len(row))
        return sp.csr_matrix((data, (row, col)), shape=(self.n_users, self.n_items))
    
    def get_social_matrix(self):
        """Get social adjacency matrix"""
        row = [u1 for u1, u2 in self.social_data]
        col = [u2 for u1, u2 in self.social_data]
        data = np.ones(len(row))
        social_mat = sp.csr_matrix((data, (row, col)), shape=(self.n_users, self.n_users))
        # Make symmetric
        social_mat = social_mat + social_mat.T
        social_mat = (social_mat > 0).astype(float)
        return social_mat
    
    def get_test_dict(self):
        """Get test data as dictionary"""
        test_dict = {}
        for user, item in self.test_data:
            if user not in test_dict:
                test_dict[user] = []
            test_dict[user].append(item)
        return test_dict


class TrainDataset(Dataset):
    """Training dataset with negative sampling"""
    
    def __init__(self, train_data, n_items, neg_ratio=1):
        self.train_data = train_data
        self.n_items = n_items
        self.neg_ratio = neg_ratio
        
        # Build user positive items dict
        self.user_pos_items = {}
        for user, item in train_data:
            if user not in self.user_pos_items:
                self.user_pos_items[user] = set()
            self.user_pos_items[user].add(item)
        
        self.users = list(self.user_pos_items.keys())
        
    def __len__(self):
        return len(self.train_data) * (1 + self.neg_ratio)
    
    def __getitem__(self, idx):
        # Sample a positive pair
        user_idx = idx % len(self.train_data)
        user, pos_item = self.train_data[user_idx]
        
        # Sample negative items
        neg_item = np.random.randint(0, self.n_items)
        while neg_item in self.user_pos_items[user]:
            neg_item = np.random.randint(0, self.n_items)
        
        return user, pos_item, neg_item


def create_dataloaders(dataset, batch_size=1024, num_workers=4):
    """Create train and test dataloaders"""
    train_dataset = TrainDataset(dataset.train_data, dataset.n_items)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    return train_loader
