"""
HyperSynergy: Custom Inference & Domain Adaptation Script.
This script demonstrates how to load pre-trained weights (.pth) to predict 
synergy probabilities on custom or simulated datasets.
"""

import torch
import numpy as np
import os
import sys

# Ensure the core library is accessible if running from the examples folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hypersynergy_core import MATG_Model, SynergyPredictor

def run_custom_inference(weights_path="weights/Proposed_MATG_Ours_v82_Final_MATG_Best.pth"):
    """
    Loads pre-trained MATG weights and performs inference on custom domain data.
    """
    print(f"\n[HyperSynergy] Initializing Inference Engine...")
    print(f"[*] Target Weights: {weights_path}")
    
    # 1. Environment & Device Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Running on device: {device}")

    # 2. Domain Parameters
    # To use the pre-trained v82 weights, the architecture must match the training dimensions.
    num_entities = 714  # Total number of unique nodes (e.g., Herbs, Drugs, or Ingredients)
    num_groups = 150    # Total number of hyperedges (e.g., Formulas, Recipes, or Protocols)
    feat_dim = 22       # Feature vector dimension used during v82 training
    
    # 3. Simulate or Load Custom Features
    # In a real scenario, you would load these from your CSV/aligned PMEA pipeline.
    # Here we simulate custom features for a new domain.
    print(f"[*] Simulating custom features (Dimension: {feat_dim})...")
    custom_vtm_feats = np.random.randn(num_entities, feat_dim)
    custom_tcm_feats = np.random.randn(num_entities, feat_dim)
    custom_form_feats = np.random.randn(num_groups, feat_dim)
    
    # 4. Initialize Architecture
    model = MATG_Model(
        num_nodes=num_entities, 
        num_hyperedges=num_groups, 
        vtm_feats=custom_vtm_feats, 
        tcm_feats=custom_tcm_feats, 
        form_feats=custom_form_feats,
        embed_dim=12
    )
    
    # 5. Load Pre-trained State
    if os.path.exists(weights_path):
        try:
            # map_location ensures compatibility between GPU training and CPU inference
            state_dict = torch.load(weights_path, map_location=device)
            model.load_state_dict(state_dict)
            print("[+] Weights successfully mapped to MATG architecture.")
        except Exception as e:
            print(f"[!] Error loading state_dict: {e}")
            print("[!] Proceeding with randomly initialized weights for demonstration.")
    else:
        print(f"[?] Weights not found at {weights_path}. Using random initialization.")

    # 6. Execute Predictions
    predictor = SynergyPredictor(model, device=device)
    
    # Example Query: Check synergy for 3 specific entity-group pairs
    # These indices refer to the rows in your custom feature matrices
    test_entities = [16, 22, 470] # Example high-degree hubs from the paper
    test_groups = [0, 5, 10]      # Candidate formula/synergy contexts
    
    probs = predictor.predict(test_entities, test_groups)
    
    # 7. Output Results
    print("\n" + "="*60)
    print(f"{'ENTITY INDEX':<15} | {'GROUP INDEX':<15} | {'PROBABILITY':<12} | {'STATUS'}")
    print("-" * 60)
    for i, p in enumerate(probs):
        # Qualitative thresholds based on the paper's calibrated focal loss results
        status = "HIGH SYNERGY" if p > 0.7 else ("POTENTIAL" if p > 0.4 else "LOW PROBABILITY")
        print(f"{test_entities[i]:<15} | {test_groups[i]:<15} | {p:12.4f} | {status}")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_custom_inference()
