"""
Example Script: MATG Synergy Prediction on Simulated Data
Intended for use in Google Colab or as a rapid-start tutorial.
"""

import torch
from hypersynergy.data_loader import DoTatLoiBenchmark
from hypersynergy.models import MATG_Model, GraphFocalLoss

# 1. Initialize Simulation Data
# In a real scenario, use: DoTatLoiBenchmark.load_and_build_graph(data_dir="data/raw")
# For this example, we use the internal mock generator from your v82 logic.
print(">>> Initializing HyperG-TCM Framework Simulation...")
dataset, vtm_feats, tcm_feats, form_feats, num_f, num_h, k_neg, _ = \
    DoTatLoiBenchmark.load_and_build_graph(data_dir="NON_EXISTENT") # Trigger fallback

# 2. Configure Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = MATG_Model(
    num_h, num_f, vtm_feats, tcm_feats, form_feats, 
    mode='proposed', 
    embed_dim=12
).to(device)

# 3. Setup Training (v82 Optimized)
optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-1)
criterion = GraphFocalLoss(gamma=4.0, pos_weight=1.5)

# 4. Minimal Training Loop
print(f"Training on {device}...")
model.train()
for epoch in range(1, 11): # Short run for example
    optimizer.zero_grad()
    
    # Batch inputs
    f_idx = torch.LongTensor(dataset[:100, 0]).to(device)
    h_idx = torch.LongTensor(dataset[:100, 1]).to(device)
    labels = torch.FloatTensor(dataset[:100, 2]).to(device)
    
    logits = model(h_idx, f_idx)
    loss = criterion(logits, labels)
    
    loss.backward()
    optimizer.step()
    
    if epoch % 2 == 0:
        print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")

print("\n>>> Example Complete. Model is ready for SynergyPredictor.predict() calls.")
