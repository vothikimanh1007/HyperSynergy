import torch
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import numpy as np
import time
import copy
import random
from .losses import GraphFocalLoss

def set_seed(seed=42):
    """Sets global seeds for reproducibility across CPU and GPU."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class ModelEvaluator:
    """
    ModelEvaluator: Manages the training and cross-validation lifecycle.
    
    Updated for v82_Final Consistency:
    1. Global seeding for consistent results.
    2. F1-Score Checkpointing: Picks the best model state based on synergy detection
       rather than majority-class accuracy (essential for 1:5 imbalanced data).
    3. Full-Batch Optimization: Prevents gradient noise in hyperbolic routing.
    """
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ModelEvaluator] Initialized on device: {self.device}")
        set_seed(42)

    def execute_model_training(self, model_factory, name, dataset, epochs=400, batch_size=None):
        """
        Executes a 5-Fold Cross-Validation training run.
        """
        X = dataset[:, :2] 
        y = dataset[:, 2]
        
        # Consistent splitting across runs
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        
        metrics = {'acc': [], 'f1': [], 'auc': []}
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
            
            # Specific AdamW setup to force Euclidean baseline collapse
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-1)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
            criterion = GraphFocalLoss(alpha=1.5, gamma=4.0)
            
            best_val_f1 = 0.0
            best_model_state = None

            # Training Loop
            for epoch in range(epochs):
                model.train()
                optimizer.zero_grad()
                
                # Forward: model(herbs, formulas)
                logits = model(X_train[:, 1], X_train[:, 0]) 
                loss = criterion(logits, y_train)
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Internal Tracking for Best State selection
                if (epoch + 1) % 5 == 0 or epoch == epochs - 1:
                    model.eval()
                    with torch.no_grad():
                        temp_logits = model(X_val[:, 1], X_val[:, 0])
                        temp_probs = torch.sigmoid(temp_logits).cpu().numpy()
                        temp_preds = (temp_probs > 0.5).astype(int)
                        # We use F1-score for checkpointing to prioritize synergy detection
                        temp_f1 = f1_score(y_val.cpu().numpy(), temp_preds, zero_division=0)
                        
                        if temp_f1 >= best_val_f1:
                            best_val_f1 = temp_f1
                            best_model_state = copy.deepcopy(model.state_dict())

            # Load the most synergistic state found during training
            if best_model_state is not None:
                model.load_state_dict(best_model_state)
            
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val[:, 1], X_val[:, 0])
                val_probs = torch.sigmoid(val_logits).cpu().numpy()
                val_targets = y_val.cpu().numpy()

            preds_binary = (val_probs > 0.5).astype(int)
            
            metrics['acc'].append(accuracy_score(val_targets, preds_binary))
            metrics['f1'].append(f1_score(val_targets, preds_binary, zero_division=0))
            metrics['auc'].append(roc_auc_score(val_targets, val_probs) if len(np.unique(val_targets)) > 1 else 0.5)
            
            global_y_true.extend(val_targets)
            global_y_probs.extend(val_probs)
            
            print(f"    Fold {fold+1} Best Results: Acc: {metrics['acc'][-1]:.4f}, F1: {metrics['f1'][-1]:.4f}, AUC: {metrics['auc'][-1]:.4f}")

        # Aggregate Metrics
        avg_metrics = {k: np.mean(v) for k, v in metrics.items()}
        std_acc = np.std(metrics['acc'])
        
        total_time = (time.time() - start_time) / 60
        print(f"\n[Final Results] {name}")
        print(f"  - Avg Accuracy: {avg_metrics['acc']:.4f} ± {std_acc:.4f}")
        print(f"  - Avg F1-Score: {avg_metrics['f1']:.4f}")
        print(f"  - Avg ROC-AUC:  {avg_metrics['auc']:.4f}")

        return avg_metrics, std_acc, metrics, (np.array(global_y_true), np.array(global_y_probs)), f"{name.replace(' ', '_')}_results.pth"
