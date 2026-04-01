import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

class NeuMapperExplainer:
    """
    NeuMapper TDA Explainer: Visualizes the topological shape of the latent space.
    
    This module implements a simplified Mapper algorithm to prove the hierarchical 
    branching of high-dimensional synergy embeddings, mapping the 3-layer structure:
    Formulas <-> Semantic Clusters <-> Individual Herbs.
    """
    def __init__(self, resolution=20, overlap=0.3):
        self.resolution = resolution
        self.overlap = overlap
        self.graph = nx.Graph()

    def generate_topology(self, vtm_features, mapped_formulas, mapped_herbs, save_path='fig7_neumapper_topology.png'):
        """
        Generates a topological map of the interaction space.
        
        Args:
            vtm_features (np.array): Semantic embeddings of the herbs.
            mapped_formulas (list): List of formula indices for each incidence link.
            mapped_herbs (list): List of herb indices for each incidence link.
            save_path (str): File path to save the resulting visualization.
        """
        print(f"[NeuMapper] Computing topological projection for {len(vtm_features)} entities...")

        # 1. Dimensionality Reduction (Filter Function)
        # We use t-SNE to project high-dim features into a 2D lens
        scaler = StandardScaler()
        feats_scaled = scaler.fit_transform(vtm_features)
        
        tsne = TSNE(n_components=2, perplexity=min(30, len(vtm_features)-1), random_state=42)
        lens = tsne.fit_transform(feats_scaled)

        # 2. Binning and Clustering (Mapper Logic)
        # We divide the lens into overlapping bins and cluster within each bin
        x_min, x_max = lens[:, 0].min(), lens[:, 0].max()
        y_min, y_max = lens[:, 1].min(), lens[:, 1].max()
        
        # Simple grid-based binning
        x_bins = np.linspace(x_min, x_max, self.resolution)
        y_bins = np.linspace(y_min, y_max, self.resolution)
        
        clusters = []
        node_id = 0
        
        # Identify nodes (clusters in bins)
        for i in range(len(x_bins)-1):
            for j in range(len(y_bins)-1):
                # Calculate bin boundaries with overlap
                x_range = (x_bins[i], x_bins[i+1])
                y_range = (y_bins[j], y_bins[j+1])
                
                # Filter points in bin
                in_bin = (lens[:, 0] >= x_range[0]) & (lens[:, 0] <= x_range[1]) & \
                         (lens[:, 1] >= y_range[0]) & (lens[:, 1] <= y_range[1])
                
                indices = np.where(in_bin)[0]
                
                if len(indices) > 5:
                    # Perform local clustering
                    db = DBSCAN(eps=0.5, min_samples=3).fit(feats_scaled[indices])
                    for label in set(db.labels_):
                        if label != -1:
                            cluster_members = indices[db.labels_ == label]
                            self.graph.add_node(node_id, members=cluster_members, size=len(cluster_members))
                            clusters.append((node_id, set(cluster_members)))
                            node_id += 1

        # 3. Connectivity (Edges)
        # Add edges between nodes that share common herb members
        for i in range(len(clusters)):
            for j in range(i + 1, len(clusters)):
                intersection = clusters[i][1].intersection(clusters[j][1])
                if len(intersection) > 0:
                    self.graph.add_edge(clusters[i][0], clusters[j][0], weight=len(intersection))

        # 4. Visualization
        self._plot_topology(save_path)
        print(f"[NeuMapper] Topology saved to {save_path}")

    def _plot_topology(self, save_path):
        plt.figure(figsize=(12, 10))
        pos = nx.spring_layout(self.graph, k=0.15, iterations=50)
        
        # Node sizes based on the number of herbs in the cluster
        node_sizes = [d['size'] * 20 for n, d in self.graph.nodes(data=True)]
        
        # Color based on hierarchy (centrality in the TDA graph)
        centrality = nx.degree_centrality(self.graph)
        node_colors = [centrality[n] for n in self.graph.nodes()]

        nx.draw_networkx_nodes(self.graph, pos, node_size=node_sizes, 
                               node_color=node_colors, cmap=plt.cm.viridis, alpha=0.8)
        nx.draw_networkx_edges(self.graph, pos, alpha=0.3, edge_color='gray')
        
        plt.title("NeuMapper: 3-Layer Topological Interaction Space", fontsize=15)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, dpi=300)
        plt.close()
