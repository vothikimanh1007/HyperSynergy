import pandas as pd
import numpy as np
import os

def generate_benchmark_simulation(output_dir="data/raw"):
    """
    Generates a simulated dataset mimicking the DoTatLoi-714 structure.
    Used for local testing and framework validation.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    num_herbs = 714
    num_formulas = 150
    feat_dim = 22

    print(f"Generating simulated dataset in {output_dir}...")

    # 1. ViThuoc_final.csv (Registry)
    registry = pd.DataFrame({
        'ID_ViThuoc': [f"V{i}" for i in range(num_herbs)],
        'TenVietNam': [f"Herb_Viet_{i}" for i in range(num_herbs)]
    })
    registry.to_csv(os.path.join(output_dir, "ViThuoc_final.csv"), index=False)

    # 2. CongThuc_updated.csv (Incidence)
    # Simulate high-order sparsity (95.5%)
    num_links = 2000
    links = pd.DataFrame({
        'ID_BaiThuoc': [f"B{np.random.randint(0, num_formulas)}" for _ in range(num_links)],
        'ID_ViThuoc': [f"V{np.random.randint(0, num_herbs)}" for _ in range(num_links)]
    })
    links.to_csv(os.path.join(output_dir, "CongThuc_updated.csv"), index=False)

    # 3. DoTatLoi_714_Enriched.csv (Semantic Features)
    enriched = pd.DataFrame({
        'ID_ViThuoc': [f"V{i}" for i in range(num_herbs)],
        'Semantic_Feature_Vector': [str(list(np.random.randn(feat_dim))) for _ in range(num_herbs)]
    })
    enriched.to_csv(os.path.join(output_dir, "DoTatLoi_714_Enriched.csv"), index=False)

    # 4. Harmonized_Global_Herbal_Dataset.csv (TCM Global)
    tcm_global = pd.DataFrame(np.random.randn(num_herbs, feat_dim))
    tcm_global.insert(0, 'ID', range(num_herbs))
    tcm_global.insert(1, 'Name', [f"TCM_{i}" for i in range(num_herbs)])
    tcm_global.to_csv(os.path.join(output_dir, "Harmonized_Global_Herbal_Dataset.csv"), index=False)

    print("Simulation Complete. Ready for PMEA alignment.")

if __name__ == "__main__":
    generate_benchmark_simulation()
