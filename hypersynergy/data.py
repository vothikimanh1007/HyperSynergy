import os
import ast
import numpy as np
import pandas as pd

class DoTatLoiBenchmark:
    """
    Standardized Dataset Loader for the DoTatLoi-714 Benchmark.
    Updated to match the exact v82_Final_MATG Colab logic for reproducibility.
    """
    
    @staticmethod
    def load_and_build_graph(k_negative=5):
        print(f"\n[Framework Module] Loading DoTatLoi-714 Benchmark Data...")
        
        # 1. Loading with Path Flexibility (Checks local dir then data/ dir)
        try:
            if os.path.exists("CongThuc_updated.csv"):
                edges_df = pd.read_csv("CongThuc_updated.csv")
                registry_df = pd.read_csv("ViThuoc_final.csv")
                vtm_df = pd.read_csv("DoTatLoi_714_Enriched.csv")
                tcm_df = pd.read_csv("Harmonized_Global_Herbal_Dataset.csv")
            else:
                edges_df = pd.read_csv("data/CongThuc_updated.csv")
                registry_df = pd.read_csv("data/ViThuoc_final.csv")
                vtm_df = pd.read_csv("data/DoTatLoi_714_Enriched.csv")
                tcm_df = pd.read_csv("data/Harmonized_Global_Herbal_Dataset.csv")
        except FileNotFoundError:
            # Fallback only if files are completely missing
            print("    [Warning] Local CSVs not found. Using mock benchmark logic...")
            num_h, num_f = 714, 150
            registry_df = pd.DataFrame({'ID_ViThuoc': [f"H{i}" for i in range(num_h)], 'TenVietNam': [f"Herb_{i}" for i in range(num_h)]})
            edges_df = pd.DataFrame({'ID_BaiThuoc': np.random.choice([f"F{i}" for i in range(num_f)], 2000),
                                     'ID_ViThuoc': np.random.choice([f"H{i}" for i in range(num_h)], 2000)})
            vtm_df = pd.DataFrame({'ID_ViThuoc': [f"H{i}" for i in range(num_h)], 'Semantic_Feature_Vector': [str(list(np.random.randn(22))) for _ in range(num_h)]})
            tcm_df = pd.DataFrame(np.random.randn(num_h, 24))
            tcm_df.insert(0, 'ID', range(num_h))
            tcm_df.insert(1, 'Name', [f"TCM_{i}" for i in range(num_h)])

        formula_col = 'ID_BaiThuoc' if 'ID_BaiThuoc' in edges_df.columns else edges_df.columns[0]
        herb_col = 'ID_ViThuoc' if 'ID_ViThuoc' in edges_df.columns else edges_df.columns[1]

        # 2. Identity Mapping
        id_to_name = dict(zip(registry_df['ID_ViThuoc'], registry_df['TenVietNam']))
        unique_herbs = registry_df['ID_ViThuoc'].unique()
        num_herbs = len(unique_herbs)
        
        herb_map = {val: i for i, val in enumerate(unique_herbs)}
        inverse_herb_map = {v: id_to_name.get(k, k) for k, v in herb_map.items()}

        edges_df = edges_df[edges_df[herb_col].isin(herb_map.keys())]
        unique_formulas = edges_df[formula_col].unique()
        num_formulas = len(unique_formulas)
        formula_map = {val: i for i, val in enumerate(unique_formulas)}

        mapped_formulas = edges_df[formula_col].map(formula_map).values
        mapped_herbs = edges_df[herb_col].map(herb_map).values

        # 3. Feature Alignment (Exact Colab Logic)
        feature_dim = 22
        vtm_features = np.zeros((num_herbs, feature_dim))
        tcm_features = np.zeros((num_herbs, feature_dim))

        # Align VTM Features
        vtm_count = 0
        for _, row in vtm_df.iterrows():
            if row['ID_ViThuoc'] in herb_map:
                idx = herb_map[row['ID_ViThuoc']]
                try: 
                    vtm_features[idx] = np.array(ast.literal_eval(row['Semantic_Feature_Vector']))
                    vtm_count += 1
                except: pass
        
        # Align TCM Global Features (Row-based alignment from Colab)
        tcm_count = 0
        for idx, row in tcm_df.iterrows():
            if idx < num_herbs:
                try: 
                    tcm_features[idx] = row.iloc[2:24].values.astype(float)
                    tcm_count += 1
                except: 
                    tcm_features[idx] = np.random.randn(22)
        
        print(f"    [Success] Mapped {vtm_count} VTM and {tcm_count} TCM feature vectors.")

        # 4. Construct Formula Features
        formula_features = np.zeros((num_formulas, feature_dim))
        for f_idx in range(num_formulas):
            f_herbs = mapped_herbs[mapped_formulas == f_idx]
            if len(f_herbs) > 0:
                formula_features[f_idx] = np.mean(vtm_features[f_herbs], axis=0)

        # 5. Dataset Construction with Negative Sampling (Colab While Loop)
        num_pos = len(mapped_formulas)
        positive_samples = np.column_stack((mapped_formulas, mapped_herbs, np.ones(num_pos)))
        positive_set = set(zip(mapped_formulas, mapped_herbs))

        negative_samples = []
        num_neg = num_pos * k_negative
        
        # Using the Colab's specific while loop for negative sampling
        # This ensures exactly num_neg unique samples are generated
        while len(negative_samples) < num_neg:
            f_rand, h_rand = np.random.randint(0, num_formulas), np.random.randint(0, num_herbs)
            if (f_rand, h_rand) not in positive_set:
                negative_samples.append([f_rand, h_rand, 0])
                positive_set.add((f_rand, h_rand))

        dataset = np.vstack((positive_samples, np.array(negative_samples)))
        np.random.shuffle(dataset)

        return (dataset, vtm_features, tcm_features, formula_features, 
                num_formulas, num_herbs, k_negative, inverse_herb_map, 
                registry_df, tcm_df, mapped_formulas, mapped_herbs)
