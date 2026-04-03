import ast
import numpy as np
import pandas as pd
import os

class DoTatLoiBenchmark:
    """
    Reusable Dataset Loader for the DoTatLoi-714 Heterogeneous Graph Benchmark.
    Handles PMEA-aligned data loading and negative sampling for synergy prediction.
    """
    @staticmethod
    def load_and_build_graph(data_dir="data/raw", k_negative=5):
        print(f"\n[HyperG-TCM] Loading DoTatLoi-714 Benchmark Data from {data_dir}...")
        
        # Define expected file paths
        edges_path = os.path.join(data_dir, "CongThuc_updated.csv")
        registry_path = os.path.join(data_dir, "ViThuoc_final.csv")
        vtm_path = os.path.join(data_dir, "DoTatLoi_714_Enriched.csv")
        tcm_path = os.path.join(data_dir, "Harmonized_Global_Herbal_Dataset.csv")

        # Load CSVs
        try:
            edges_df = pd.read_csv(edges_path)
            registry_df = pd.read_csv(registry_path)
            vtm_df = pd.read_csv(vtm_path)
            tcm_df = pd.read_csv(tcm_path)
        except FileNotFoundError as e:
            print(f"    [Error] Critical dataset missing: {e}")
            print("    Falling back to mock data generation for demonstration...")
            return DoTatLoiBenchmark._generate_mock_data(k_negative)

        # 1. Map Herbs and Formulas
        formula_col = 'ID_BaiThuoc' if 'ID_BaiThuoc' in edges_df.columns else edges_df.columns[0]
        herb_col = 'ID_ViThuoc' if 'ID_ViThuoc' in edges_df.columns else edges_df.columns[1]
        
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

        # 2. Extract Features (VTM & TCM)
        feature_dim = 22
        vtm_features = np.zeros((num_herbs, feature_dim))
        tcm_features = np.zeros((num_herbs, feature_dim))

        for _, row in vtm_df.iterrows():
            if row['ID_ViThuoc'] in herb_map:
                idx = herb_map[row['ID_ViThuoc']]
                try:
                    vtm_features[idx] = np.array(ast.literal_eval(row['Semantic_Feature_Vector']))
                except:
                    pass

        for idx, row in tcm_df.iterrows():
            if idx < num_herbs:
                try:
                    # Assumes global features start at col index 2
                    tcm_features[idx] = row.iloc[2:24].values.astype(float)
                except:
                    tcm_features[idx] = np.random.randn(feature_dim)

        # 3. Construct Formula (Hyperedge) Features via Mean Pooling
        formula_features = np.zeros((num_formulas, feature_dim))
        for f_idx in range(num_formulas):
            f_herbs = mapped_herbs[mapped_formulas == f_idx]
            if len(f_herbs) > 0:
                formula_features[f_idx] = np.mean(vtm_features[f_herbs], axis=0)

        # 4. Dataset Generation (Positive & Negative Samples)
        num_pos = len(mapped_formulas)
        positive_samples = np.column_stack((mapped_formulas, mapped_herbs, np.ones(num_pos)))
        positive_set = set(zip(mapped_formulas, mapped_herbs))
        
        negative_samples = []
        num_neg = num_pos * k_negative
        while len(negative_samples) < num_neg:
            f_rand = np.random.randint(0, num_formulas)
            h_rand = np.random.randint(0, num_herbs)
            if (f_rand, h_rand) not in positive_set:
                negative_samples.append([f_rand, h_rand, 0])
                positive_set.add((f_rand, h_rand))
        
        dataset = np.vstack((positive_samples, np.array(negative_samples)))
        np.random.shuffle(dataset)

        return dataset, vtm_features, tcm_features, formula_features, num_formulas, num_herbs, k_negative, inverse_herb_map

    @staticmethod
    def _generate_mock_data(k_negative):
        """Internal helper for mock data generation if local CSVs are not provided."""
        num_h, num_f = 714, 150
        vtm_feats = np.random.randn(num_h, 22)
        tcm_feats = np.random.randn(num_h, 22)
        form_feats = np.random.randn(num_f, 22)
        
        # Create a randomized incidence matrix
        num_edges = 2000
        f_idx = np.random.randint(0, num_f, num_edges)
        h_idx = np.random.randint(0, num_h, num_edges)
        
        pos_samples = np.column_stack((f_idx, h_idx, np.ones(num_edges)))
        neg_samples = np.column_stack((
            np.random.randint(0, num_f, num_edges * k_negative),
            np.random.randint(0, num_h, num_edges * k_negative),
            np.zeros(num_edges * k_negative)
        ))
        
        dataset = np.vstack((pos_samples, neg_samples))
        inverse_map = {i: f"Herb_{i}" for i in range(num_h)}
        
        return dataset, vtm_feats, tcm_feats, form_feats, num_f, num_h, k_negative, inverse_map
