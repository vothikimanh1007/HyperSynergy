import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import time
from .losses import GraphFocalLoss

class ModelEvaluator:
    """
    ModelEvaluator: Manages the training and cross-validation lifecycle.
    
    Revised to use Full-Batch Training as per the v82_Final Colab logic.
    Full-batch optimization is critical for the stability of Riemannian 
    manifolds on small-scale hypergraph benchmarks like DoTatLoi-714.
    """
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ModelEvaluator] Initialized on device: {self.device}")

    def execute_model_training(self, model_factory, name, dataset, epochs=400, batch_size=None):
        """
        Executes a 5-Fold Cross-Validation training run using Full-Batch optimization.
        
        Args:
            model_factory: Lambda returning a fresh model instance.
            name: Name of the model (for logging).
            dataset: Array [formula_idx, herb_idx, label].
            epochs: Training duration.
            batch_size: Ignored in favor of Full-Batch logic for v82 reproducibility.
        """
        X = dataset[:, :2] 
        y = dataset[:, 2]
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = []
        
        print(f"\n[Execution] Starting 5-Fold CV for {name}...")
        start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  > Fold {fold+1}/5 training (Full Batch)...")
            
            # Prepare Full-Batch Tensors
            X_train = torch.LongTensor(X[train_idx]).to(self.device)
            y_train = torch.FloatTensor(y[train_idx]).to(self.device)
            X_val = torch.LongTensor(X[val_idx]).to(self.device)
            y_val = torch.FloatTensor(y[val_idx]).to(self.device)

            model = model_factory().to(self.device)
            
            # AdamW + CosineAnnealingLR as per Colab v82
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = GraphFocalLoss(alpha=1.5, gamma=4.0)
            
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                # model(herb_indices, formula_indices)
                # X_train[:, 1] is Herb, X_train[:, 0] is Formula
                logits = model(X_train[:, 1], X_train[:, 0]) 
                
                loss = criterion(logits, y_train)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Evaluation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val[:, 1], X_val[:, 0])
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_targets = y_val.cpu().numpy()

            preds_binary = (val_probs > 0.5).astype(int)
            metrics = {
                'acc': accuracy_score(val_targets, preds_binary),
                'f1': f1_score(val_targets, preds_binary, zero_division=0),
                'auc': roc_auc_score(val_targets, val_probs) if len(np.unique(val_targets)) > 1 else 0.5
            }
            fold_metrics.append(metrics)
            print(f"    Fold {fold+1} Results: Acc: {metrics['acc']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")

        # Aggregate Metrics
        avg_metrics = {k: np.mean([f[k] for f in fold_metrics]) for k in ['acc', 'f1', 'auc']}
        std_acc = np.std([f['acc'] for f in fold_metrics])
        
        total_time = (time.time() - start_time) / 60
        print(f"\n[Final Results] {name}")
        print(f"  - Avg Accuracy: {avg_metrics['acc']:.4f} ± {std_acc:.4f}")
        print(f"  - Avg F1-Score: {avg_metrics['f1']:.4f}")
        print(f"  - Avg ROC-AUC:  {avg_metrics['auc']:.4f}")
        print(f"  - Total Execution: {total_time:.2f} mins")

        return avg_metrics, std_acc, fold_metrics, val_probs, f"{name.replace(' ', '_')}_results.pth"
