import torch
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import time
from .losses import GraphFocalLoss

class ModelEvaluator:
    """
    ModelEvaluator: Manages the training and cross-validation lifecycle.
    
    Includes automated Stratified 5-Fold splitting, focal-weighted training loops,
    and standardized metric reporting for the HyperSynergy framework.
    """
    def __init__(self, device=None):
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[ModelEvaluator] Initialized on device: {self.device}")

    def execute_model_training(self, model_factory, name, dataset, epochs=400, batch_size=256):
        """
        Executes a 5-Fold Cross-Validation training run.
        
        Args:
            model_factory (callable): A function that returns a new model instance.
            name (str): Name of the model configuration (for logging).
            dataset (np.array): The shuffled [formula_idx, herb_idx, label] dataset.
            epochs (int): Number of training epochs per fold.
            batch_size (int): Size of training batches.
            
        Returns:
            dict: Mean metrics and standard deviations.
        """
        X = dataset[:, :2] # Formula and Herb indices
        y = dataset[:, 2]  # Labels
        
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        fold_metrics = []
        
        print(f"\n[Execution] Starting 5-Fold CV for {name}...")
        start_time = time.time()

        for fold, (train_idx, val_idx) in enumerate(skf.split(X, y)):
            print(f"  > Fold {fold+1}/5 training...")
            
            # 1. Prepare Data Loaders
            train_data = DataLoader(
                TensorDataset(torch.LongTensor(X[train_idx]), torch.FloatTensor(y[train_idx])), 
                batch_size=batch_size, shuffle=True
            )
            val_data = DataLoader(
                TensorDataset(torch.LongTensor(X[val_idx]), torch.FloatTensor(y[val_idx])), 
                batch_size=batch_size
            )

            # 2. Initialize Model, Optimizer, and Loss
            model = model_factory().to(self.device)
            optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-1)
            criterion = GraphFocalLoss(alpha=1.5, gamma=4.0)
            
            # 3. Training Loop
            for epoch in range(epochs):
                model.train()
                for inputs, targets in train_data:
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    optimizer.zero_grad()
                    logits = model(inputs[:, 0], inputs[:, 1])
                    loss = criterion(logits, targets)
                    loss.backward()
                    optimizer.step()

            # 4. Evaluation
            model.eval()
            all_preds = []
            all_targets = []
            with torch.no_grad():
                for inputs, targets in val_data:
                    inputs = inputs.to(self.device)
                    logits = model(inputs[:, 0], inputs[:, 1])
                    probs = torch.sigmoid(logits).cpu().numpy()
                    all_preds.extend(probs)
                    all_targets.extend(targets.numpy())

            # Calculate Metrics for this Fold
            preds_binary = (np.array(all_preds) > 0.5).astype(int)
            metrics = {
                'acc': accuracy_score(all_targets, preds_binary),
                'f1': f1_score(all_targets, preds_binary),
                'auc': roc_auc_score(all_targets, all_preds)
            }
            fold_metrics.append(metrics)
            print(f"    Fold {fold+1} Results: Acc: {metrics['acc']:.4f}, F1: {metrics['f1']:.4f}, AUC: {metrics['auc']:.4f}")

        # 5. Aggregate Results
        avg_metrics = {k: np.mean([f[k] for f in fold_metrics]) for k in ['acc', 'f1', 'auc']}
        std_acc = np.std([f['acc'] for f in fold_metrics])
        
        total_time = (time.time() - start_time) / 60
        print(f"\n[Final Results] {name}")
        print(f"  - Avg Accuracy: {avg_metrics['acc']:.4f} ± {std_acc:.4f}")
        print(f"  - Avg F1-Score: {avg_metrics['f1']:.4f}")
        print(f"  - Avg ROC-AUC:  {avg_metrics['auc']:.4f}")
        print(f"  - Execution Time: {total_time:.2f} mins")

        return avg_metrics, std_acc, fold_metrics, all_preds, f"{name}_results.pth"
