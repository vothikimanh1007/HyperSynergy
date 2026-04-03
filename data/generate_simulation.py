import pandas as pd
import numpy as np
import os

def create_domain_simulation(
    domain_name="Herbal",
    entity_name="Herb",
    group_name="Formula",
    num_entities=714,
    num_groups=150,
    feat_dim=22,
    sparsity=0.95,
    target_path="data/raw"
):
    """
    Creates a simulated CSV suite for any domain knowledge.
    
    Args:
        domain_name: e.g., 'Pharma', 'Nutritional', 'Social'
        entity_name: e.g., 'Drug', 'Ingredient', 'User'
        group_name: e.g., 'Combination', 'Recipe', 'Community'
        num_entities: Total nodes (V)
        num_groups: Total hyperedges (E)
        feat_dim: Feature vector size (must be 22 for current MATG v82 config)
        sparsity: Ratio of negative to positive incidence
    """
    if not os.path.exists(target_path):
        os.makedirs(target_path)
    
    print(f"\n>>> Generating {domain_name} Domain Simulation...")

    # 1. Incidence Matrix (Connections)
    # Simulating standard positive interactions
    num_pos_links = int(num_entities * num_groups * (1 - sparsity))
    pos_links = pd.DataFrame({
        'ID_Group': [f"G_{np.random.randint(0, num_groups)}" for _ in range(num_pos_links)],
        'ID_Entity': [f"E_{np.random.randint(0, num_entities)}" for _ in range(num_pos_links)]
    })
    
    # Standardizing column names for HyperSynergy compatibility
    # Note: Library expects ID_BaiThuoc and ID_ViThuoc for legacy support
    pos_links.columns = ['ID_BaiThuoc', 'ID_ViThuoc'] 
    pos_links.to_csv(f"{target_path}/CongThuc_updated.csv", index=False)

    # 2. Entity Features (Semantic Knowledge)
    # We simulate semantic vectors (e.g., chemical properties, nutritional values)
    entities = pd.DataFrame({
        'ID_ViThuoc': [f"E_{i}" for i in range(num_entities)],
        'Entity_Name': [f"{entity_name}_{i}" for i in range(num_entities)],
        'Semantic_Feature_Vector': [str(list(np.random.randn(feat_dim))) for _ in range(num_entities)]
    })
    entities.to_csv(f"{target_path}/DoTatLoi_714_Enriched.csv", index=False)

    # 3. Global Registry Simulation (For PMEA alignment testing)
    # This simulates a "Global Database" for cross-cultural alignment
    global_reg = pd.DataFrame(np.random.randn(num_entities, feat_dim))
    global_reg.insert(0, 'ID', [f"GLOBAL_{i}" for i in range(num_entities)])
    global_reg.to_csv(f"{target_path}/Harmonized_Global_Herbal_Dataset.csv", index=False)

    print(f"Success: {domain_name} dataset saved to {target_path}")
    print(f"- Entities: {num_entities} ({entity_name})")
    print(f"- Groups: {num_groups} ({group_name})")

if __name__ == "__main__":
    # Example 1: Original Herbal Medicine
    create_domain_simulation(
        domain_name="Vietnamese Traditional Medicine",
        entity_name="Herb",
        group_name="Formula",
        target_path="data/raw"
    )

    # Example 2: Western Drug-Drug Synergy (Uncomment to use)
    # create_domain_simulation(
    #     domain_name="Pharma_Synergy",
    #     entity_name="Drug_Compound",
    #     group_name="Treatment_Protocol",
    #     num_entities=500,
    #     num_groups=100,
    #     target_path="data/raw_pharma"
    # )
