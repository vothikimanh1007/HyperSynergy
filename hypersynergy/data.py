import torch
import numpy as np
import random
import os
from hypersynergy.data_loader import DoTatLoiBenchmark
from hypersynergy.models import MATG_Model
from hypersynergy.evaluation import ModelEvaluator
from hypersynergy.explainers import NeuMapperExplainer

def set_seed(seed=42):
    """
    Sets the seed for all relevant libraries to ensure reproducibility across 
    CPU and GPU environments.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    
    # Force deterministic behavior in CuDNN to prevent floating-point variance
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set a fixed value for the Python hash seed
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"[Reproducibility] Global seed set to: {seed}")

def run_hypersynergy_pipeline():
    """
    Main execution script for the HyperSynergy Framework.
    1. Sets global seed for perfect reproducibility.
    2. Loads the DoTatLoi-714 Benchmark.
    3. Trains the Proposed MATG Model using 5-Fold Cross-Validation.
    4. Runs an Ablation Baseline (GCN) for comparison.
    5. Generates NeuMapper TDA Visualizations.
    """
    
    # Step 0: Set seed before any data loading or model initialization
    set_seed(42)
    
    # --- Step 1: Data Preparation ---
    (dataset, vtm_feats, tcm_feats, form_feats, 
     num_form, num_herbs, k_neg, inverse_herb_map, 
     registry_df, tcm_df, mapped_formulas, mapped_herbs) = DoTatLoiBenchmark.load_and_build_graph()

    # --- Step 2: Initialize Evaluator ---
    evaluator = ModelEvaluator()

    # --- Step 3: Train Proposed MATG Model ---
    def matg_factory():
        return MATG_Model(
            num_nodes=num_herbs,
            num_hyperedges=num_form,
            vtm_feats=vtm_feats,
            tcm_feats=tcm_feats,
            formula_feats=form_feats,
            mode='proposed',
            embed_dim=12
        )

    matg_results, matg_std, _, _, _ = evaluator.execute_model_training(
        model_factory=matg_factory,
        name="Proposed_MATG",
        dataset=dataset,
        epochs=400,
        batch_size=256
    )

    # --- Step 4: Train Baseline (GCN) for Comparison ---
    def gcn_factory():
        return MATG_Model(
            num_nodes=num_herbs,
            num_hyperedges=num_form,
            vtm_feats=vtm_feats,
            tcm_feats=tcm_feats,
            formula_feats=form_feats,
            mode='gcn',
            embed_dim=12
        )

    gcn_results, gcn_std, _, _, _ = evaluator.execute_model_training(
        model_factory=gcn_factory,
        name="GCN_Baseline",
        dataset=dataset,
        epochs=400,
        batch_size=256
    )

    # --- Step 5: Explainability (NeuMapper TDA) ---
    print("\n[XAI] Generating Topological Data Analysis (TDA) Map...")
    explainer = NeuMapperExplainer(resolution=20, overlap=0.3)
    explainer.generate_topology(
        vtm_features=vtm_feats,
        mapped_formulas=mapped_formulas,
        mapped_herbs=mapped_herbs,
        save_path='fig7_neumapper_topology.png'
    )

    # --- Step 6: Final Summary Comparison ---
    print("\n" + "="*40)
    print("HYPERSYNERGY BENCHMARK SUMMARY")
    print("="*40)
    print(f"{'Model':<15} | {'Accuracy':<12} | {'F1-Score':<10}")
    print("-" * 40)
    print(f"{'Proposed MATG':<15} | {matg_results['acc']:.4f} ± {matg_std:.4f} | {matg_results['f1']:.4f}")
    print(f"{'GCN Baseline':<15} | {gcn_results['acc']:.4f} ± {gcn_std:.4f} | {gcn_results['f1']:.4f}")
    print("="*40)
    print("Results and Figure 7 saved to local directory.")

if __name__ == "__main__":
    run_hypersynergy_pipeline()
