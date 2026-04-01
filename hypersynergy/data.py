import os
import ast
import numpy as np
import pandas as pd

class DoTatLoiBenchmark:
    """
    Strict Dataset Loader for the DoTatLoi-714 Heterogeneous Graph Benchmark.
    Ensures semantic features are correctly mapped to avoid training on zero-vectors.
    """
    
    @staticmethod
    def load_and_build_graph(k_negative=5):
        print(f"\n[Framework Module] Attempting to load DoTatLoi-714 Benchmark Data...")
        
        # Define expected paths
        paths = {
            'edges': "data/CongThuc_updated.csv",
            'registry': "data/ViThuoc_final.csv",
            'vtm': "data/DoTatLoi_714_Enriched.csv",
            'tcm': "data/Harmonized_Global_Herbal_Dataset.csv"
        }

        # Check if files exist, otherwise check root
        for key in paths:
            if not os.path.exists(paths[key]):
                root_path = os.path.basename(paths[key])
                if os.path.exists(root_path):
                    paths[key] = root_path
                else:
                    raise FileNotFoundError(f"CRITICAL ERROR: {paths[key]} not found. "
                                          f"Please ensure your CSV files are in the /data/ folder.")

        # Load Data
        edges_df = pd.read_csv(paths['edges'])
        registry_df = pd.read_csv(paths['registry'])
        vtm_df = pd.read_csv(paths['vtm'])
        tcm_df = pd.read_csv(paths['tcm'])

        print(f"    [Success] Loaded {len(registry_df)} herbs and {len(edges_df)} incidence links.")

        formula_col = 'ID_BaiThuoc' if 'ID_BaiThuoc' in edges_df.columns else edges_df.columns[0]
        herb_col = 'ID_ViThuoc' if 'ID_ViThuoc' in edges_df.columns else edges_df.columns[1]

        # 1. Map Identities
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

        # 2. Extract and Align Features
        feature_dim = 22
        vtm_features = np.zeros((num_herbs, feature_dim))
        tcm_features = np.zeros((num_herbs, feature_dim))

        # Align VTM Features (Crucial for MATG Wisdom)
        vtm_count = 0
        for _, row in vtm_df.iterrows():
            h_id = row['ID_ViThuoc']
            if h_id in herb_map:
                idx = herb_map[h_id]
                try: 
                    feat_vec = ast.literal_eval(row['Semantic_Feature_Vector'])
                    vtm_features[idx] = np.array(feat_vec)
                    vtm_count += 1
                except: pass
        
        print(f"    [Alignment] Successfully mapped {vtm_count}/{num_herbs} semantic feature vectors.")

        # Align TCM Global Features
        tcm_count = 0
        for _, row in tcm_df.iterrows():
            # Match by name or ID if possible
            h_id = row['ID'] if 'ID' in row else None
            if h_id and h_id in herb_map:
                idx = herb_map[h_id]
                tcm_features[idx] = row.iloc[2:24].values.astype(float)
                tcm_count += 1
        
        # Fill missing with small noise instead of zeros to maintain manifold volume
        if vtm_count < num_herbs:
            vtm_features[vtm_features.sum(axis=1) == 0] = np.random.normal(0, 0.01, (num_herbs - vtm_count, 22))

        # 3. Construct Hyperedge (Formula) Features
        formula_features = np.zeros((num_formulas, feature_dim))
        for f_idx in range(num_formulas):
            f_herbs = mapped_herbs[mapped_formulas == f_idx]
            if len(f_herbs) > 0:
                formula_features[f_idx] = np.mean(vtm_features[f_herbs], axis=0)

        # 4. Construct Dataset with Negative Sampling (1:5 Ratio)
        num_pos = len(mapped_formulas)
        positive_samples = np.column_stack((mapped_formulas, mapped_herbs, np.ones(num_pos)))
        positive_set = set(zip(mapped_formulas, mapped_herbs))

        negative_samples = []
        num_neg = num_pos * k_negative
        
        # Faster negative sampling
        all_f = np.random.randint(0, num_formulas, num_neg * 2)
        all_h = np.random.randint(0, num_herbs, num_neg * 2)
        
        for f, h in zip(all_f, all_h):
            if len(negative_samples) >= num_neg: break
            if (f, h) not in positive_set:
                negative_samples.append([f, h, 0])
                positive_set.add((f, h))

        dataset = np.vstack((positive_samples, np.array(negative_samples)))
        np.random.shuffle(dataset)

        return (dataset, vtm_features, tcm_features, formula_features, 
                num_formulas, num_herbs, k_negative, inverse_herb_map, 
                registry_df, tcm_df, mapped_formulas, mapped_herbs)
