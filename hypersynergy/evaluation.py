import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import time
from .losses import GraphFocalLoss

class ModelEvaluator:
    """
    ModelEvaluator: Manages the training and cross-validation lifecycle.
    
    Updated to include CosineAnnealingLR and AdamW weight decay as specified 
    in the v82_Final Colab notes for optimal 400-epoch convergence.
    """
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ModelEvaluator] Initialized on device: {self.device}")

    def execute_model_training(self, model_factory, name, dataset, epochs=400, batch_size=256):
        """
        Executes a 5-Fold Cross-Validation training run with scheduling.
        dataset columns: [0: formula_idx, 1: herb_idx, 2: label]
        """
        X = dataset[:, :2] 
        y = dataset[:, 2]
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = []
        
        print(f"\n[Execution] Starting 5-Fold CV for {name}...")
        start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  > Fold {fold+1}/5 training...")
            
            # Prepare Data Loaders
            train_loader = DataLoader(
                TensorDataset(torch.LongTensor(X[train_idx]), torch.FloatTensor(y[train_idx])), 
                batch_size=batch_size, shuffle=True
            )
            val_loader = DataLoader(
                TensorDataset(torch.LongTensor(X[val_idx]), torch.FloatTensor(y[val_idx])), 
                batch_size=batch_size
            )

            model = model_factory().to(self.device)
            
            # Optimizer and Scheduler configuration from v82 Final Colab
            # High weight decay is essential to penalize Euclidean collisions 
            # while the manifold routing holds structural integrity.
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            criterion = GraphFocalLoss(alpha=1.5, gamma=4.0)
            
            for epoch in range(epochs):
                model.train()
                for inputs, targets in train_loader:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    
                    # model expects (herb_indices, formula_indices)
                    # dataset columns: 0 = Formula, 1 = Herb
                    logits = model(inputs[:, 1], inputs[:, 0]) 
                    
                    loss = criterion(logits, targets)
                    loss.backward()
                    optimizer.step()
                
                # Step the scheduler at the end of each epoch to follow the cosine curve
                scheduler.step()

            # Final evaluation for the fold
            model.eval()
            all_preds, all_targets = [], []
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.to(self.device)
                    # model expects (herb_indices, formula_indices)
                    logits = model(inputs[:, 1], inputs[:, 0]) 
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(probs)
                    all_targets.extend(targets.numpy())

            preds_binary = (np.array(all_preds) > 0.5).astype(int)
            metrics = {
                'acc': accuracy_score(all_targets, preds_binary),
                'f1': f1_score(all_targets, preds_binary, zero_division=0),
                'auc': roc_auc_score(all_targets, all_preds) if len(np.unique(all_targets)) > 1 else 0.5
            }
            fold_metrics.append(metrics)
            print(f"    Fold {fold+1} Results: Acc: {metrics['acc']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")

        # Aggregate metrics across all folds
        avg_metrics = {k: np.mean([f[k] for f in fold_metrics]) for k in ['acc', 'f1', 'auc']}
        std_acc = np.std([f['acc'] for f in fold_metrics])
        
        total_time = (time.time() - start_time) / 60
        print(f"\n[Final Results] {name}")
        print(f"  - Avg Accuracy: {avg_metrics['acc']:.4f} ± {std_acc:.4f}")
        print(f"  - Avg F1-Score: {avg_metrics['f1']:.4f}")
        print(f"  - Avg ROC-AUC:  {avg_metrics['auc']:.4f}")
        print(f"  - Total Execution: {total_time:.2f} mins")

        return avg_metrics, std_acc, fold_metrics, all_preds, f"{name.replace(' ', '_')}_results.pth"
