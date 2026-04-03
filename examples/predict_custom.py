"""
HyperSynergy: Custom Inference & Domain Adaptation Script.
Usage: python predict_custom.py --weights weights/Proposed_MATG_Ours_v82_Final_MATG_Best.pth
"""

import torch
import numpy as np
import argparse
import os
import sys

# Ensure the parent directory is in the path so we can import hypersynergy_core
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hypersynergy_core import MATG_Model, SynergyPredictor

def run_inference(weights_path):
    print(f"\n[HyperSynergy] Starting Inference Engine...")
    print(f"[*] Loading weights from: {weights_path}")

    # 1. Hardware & Environment Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"[*] Target Device: {device}")

    # 2. Domain Parameters (Aligned with v82 paper benchmark)
    # Note: These dimensions must match the pre-trained state_dict
    NUM_NODES = 714  
    NUM_GROUPS = 150 
    FEAT_DIM = 22     
    EMBED_DIM = 12

    # 3. Data Mocking/Loading
    # In a real-world application, replace these with your actual aligned features
    # (e.g., from your PMEA alignment pipeline output)
    print(f"[*] Generating mock feature space for domain adaptation...")
    vtm_feats = np.random.randn(NUM_NODES, FEAT_DIM)
    tcm_feats = np.random.randn(NUM_NODES, FEAT_DIM)
    form_feats = np.random.randn(NUM_GROUPS, FEAT_DIM)

    # 4. Model Initialization
    model = MATG_Model(
        num_nodes=NUM_NODES,
        num_hyperedges=NUM_GROUPS,
        vtm_feats=vtm_feats,
        tcm_feats=tcm_feats,
        form_feats=form_feats,
        embed_dim=EMBED_DIM
    )

    # 5. Loading Pre-trained Weights
    if os.path.exists(weights_path):
        try:
            # Using map_location='cpu' for cross-device compatibility
            checkpoint = torch.load(weights_path, map_location=device)
            model.load_state_dict(checkpoint)
            print("[+] Successfully mapped pre-trained weights to MATG manifold.")
        except Exception as e:
            print(f"[!] Warning: Could not load weights ({e}).")
            print("[!] Running with random initialization for demonstration purposes.")
    else:
        print(f"[!] Error: Weights file not found at {weights_path}")
        return

    # 6. Prediction Logic
    predictor = SynergyPredictor(model, device=device)

    # Example Query Indices (e.g., Herbs 16, 22, 470 vs Formula 5)
    test_herbs = [16, 22, 470, 0, 152]
    test_formulas = [5, 5, 5, 10, 10]

    print(f"[*] Executing synergy link prediction for {len(test_herbs)} pairs...")
    probabilities = predictor.predict(test_herbs, test_formulas)

    # 7. Formatting Output for Research Presentation
    print("\n" + "="*65)
    print(f"{'Herb Index':<12} | {'Formula Index':<15} | {'Synergy Prob':<15} | {'Status'}")
    print("-" * 65)
    
    for h, f, p in zip(test_herbs, test_formulas, probabilities):
        # Thresholds based on Calibrated Graph Focal Loss (v82)
        if p > 0.75:
            status = "High (Synergistic)"
        elif p > 0.45:
            status = "Moderate (Potential)"
        else:
            status = "Low (Baseline)"
            
        print(f"{h:<12} | {f:<15} | {p:15.4f} | {status}")
    
    print("="*65)
    print("[HyperSynergy] Inference Task Complete.\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HyperSynergy Custom Predictor")
    parser.add_argument(
        "--weights", 
        type=str, 
        default="weights/Proposed_MATG_Ours_v82_Final_MATG_Best.pth",
        help="Path to the pre-trained .pth file"
    )
    args = parser.parse_args()
    
    run_inference(args.weights)
