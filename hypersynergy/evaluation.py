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
    
    Updated to match the v82_Final Colab logic:
    1. Full-Batch Training for Riemannian stability.
    2. Global Prediction Accumulation across all 5 folds (necessary for Fig 8 & 9).
    3. AdamW Weight Decay (1e-1) to force Euclidean collapse vs. Manifold separation.
    """
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ModelEvaluator] Initialized on device: {self.device}")

    def execute_model_training(self, model_factory, name, dataset, epochs=400, batch_size=None):
        """
        Executes a 5-Fold Cross-Validation training run.
        
        Args:
            model_factory: Lambda returning a fresh model instance.
            name: Name of the model.
            dataset: Array [formula_idx, herb_idx, label].
            epochs: Training duration.
        """
        X = dataset[:, :2] 
        y = dataset[:, 2]
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        metrics = {'acc': [], 'f1': [], 'auc': []}
        # Global lists to store results from all folds for aggregate metrics (Fig 8/9)
        global_y_true = []
        global_y_probs = []
        
        print(f"\n[Execution] Starting 5-Fold CV for {name}...")
        start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  > Fold {fold+1}/5 training (Full Batch)...")
            
            # Prepare Tensors
            X_train = torch.LongTensor(X[train_idx]).to(self.device)
            y_train = torch.FloatTensor(y[train_idx]).to(self.device)
            X_val = torch.LongTensor(X[val_idx]).to(self.device)
            y_val = torch.FloatTensor(y[val_idx]).to(self.device)

            model = model_factory().to(self.device)
            
            # Use specific weight_decay from Colab to penalize Euclidean collisions
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            
            # Ensure GraphFocalLoss uses the 1.5 pos_weight as specified in the paper
            criterion = GraphFocalLoss(alpha=1.5, gamma=4.0)
            
            # Training Loop
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                # model expects (herb_indices, formula_indices)
                logits = model(X_train[:, 1], X_train[:, 0]) 
                
                loss = criterion(logits, y_train)
                loss.backward()
                optimizer.step()
                scheduler.step()

            # Post-Training Fold Evaluation
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val[:, 1], X_val[:, 0])
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_targets = y_val.cpu().numpy()

            preds_binary = (val_probs > 0.5).astype(int)
            
            # Record metrics for this fold
            metrics['acc'].append(accuracy_score(val_targets, preds_binary))
            metrics['f1'].append(f1_score(val_targets, preds_binary, zero_division=0))
            metrics['auc'].append(roc_auc_score(val_targets, val_probs) if len(np.unique(val_targets)) > 1 else 0.5)
            
            # Accumulate for global reporting
            global_y_true.extend(val_targets)
            global_y_probs.extend(val_probs)
            
            print(f"    Fold {fold+1} Results: Acc: {metrics['acc'][-1]:.4f}, F1: {metrics['f1'][-1]:.4f}, AUC: {metrics['auc'][-1]:.4f}")

        # Aggregate Results
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        std_acc = np.std(metrics['acc'])
        
        total_time = (time.time() - start_time) / 60
        print(f"\n[Final Results] {name}")
        print(f"  - Avg Accuracy: {avg_metrics['acc']:.4f} ± {std_acc:.4f}")
        print(f"  - Avg F1-Score: {avg_metrics['f1']:.4f}")
        print(f"  - Avg ROC-AUC:  {avg_metrics['auc']:.4f}")
        print(f"  - Total Execution: {total_time:.2f} mins")

        # Return global predictions for the benchmark script to generate Fig 8/9
        return avg_metrics, std_acc, metrics, (np.array(global_y_true), np.array(global_y_probs)), f"{name.replace(' ', '_')}_results.pth"
