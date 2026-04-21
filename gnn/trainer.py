"""
gnn/trainer.py
--------------
Training loop with early stopping and model checkpointing.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score
import numpy as np
import copy
from typing import Tuple

class GNNTrainer:
    def __init__(
        self,
        model: nn.Module,
        lr: float = 0.005,
        weight_decay: float = 1e-4,
        patience: int = 15,
        max_epochs: int = 150
    ):
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.patience = patience
        self.max_epochs = max_epochs
        self.best_model_state = None
        self.best_val_auc = 0.0
        self.epochs_no_improve = 0

    def compute_pos_weight(self, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Determines pos_weight = num_neg / num_pos for BCE loss."""
        y_train = y[mask]
        num_pos = torch.sum(y_train).item()
        num_neg = len(y_train) - num_pos
        return torch.tensor([num_neg / (num_pos + 1e-9)])

    def train_epoch(self, data) -> float:
        self.model.train()
        self.optimizer.zero_grad()
        
        logits = self.model(data.x, data.edge_index)
        
        # Only compute loss on TRAIN mask
        loss = self.criterion(logits[data.mask], data.y[data.mask])
        
        loss.backward()
        self.optimizer.step()
        return loss.item()

    @torch.no_grad()
    def evaluate(self, data) -> float:
        self.model.eval()
        logits = self.model(data.x, data.edge_index)
        probs = torch.sigmoid(logits)
        
        y_true = data.y[data.mask].cpu().numpy()
        y_score = probs[data.mask].cpu().numpy()
        
        if len(np.unique(y_true)) < 2:
            return 0.5
        return roc_auc_score(y_true, y_score)

    def train(self, data_train, data_val):
        pos_weight = self.compute_pos_weight(data_train.y, data_train.mask)
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
        print(f"Starting training (pos_weight={pos_weight.item():.2f})...")
        
        for epoch in range(1, self.max_epochs + 1):
            loss = self.train_epoch(data_train)
            val_auc = self.evaluate(data_val)
            
            if epoch % 5 == 0 or epoch == 1:
                print(f"Epoch {epoch:03d} | Train Loss: {loss:.4f} | Val AUC: {val_auc:.4f}")
            
            # Early Stopping check
            if val_auc > self.best_val_auc:
                self.best_val_auc = val_auc
                self.best_model_state = copy.deepcopy(self.model.state_dict())
                self.epochs_no_improve = 0
            else:
                self.epochs_no_improve += 1
                
            if self.epochs_no_improve >= self.patience:
                print(f"Early stopping at epoch {epoch}. Best Val AUC: {self.best_val_auc:.4f}")
                break
                
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            
    @torch.no_grad()
    def get_probs(self, data) -> np.ndarray:
        self.model.eval()
        logits = self.model(data.x, data.edge_index)
        return torch.sigmoid(logits).cpu().numpy()
