# HyperSynergy: Topological Knowledge Bridging

Official implementation of the methodology detailed in: **"Topological Knowledge Bridging: The DoTatLoi-714 Benchmark and a Heterogeneous Hypergraph Framework for Cross-Cultural Herbal Synergy Prediction."**

## 📖 Overview

Standard Graph Neural Networks (GNNs) suffer from "Dimensional Congestion" when attempting to model complex, multi-entity biological synergies in flat Euclidean space. **HyperSynergy** is a domain-agnostic algorithmic library that maps heterogeneous bipartite hypergraphs into a Poincaré hyperbolic manifold.

Powered by **Manifold-Aware Transformer Gating (MATG)** and **Riemannian Residual Gating (RRG)**, the library dynamically penalizes Euclidean node-collisions using exact Poincaré log-map distances, achieving state-of-the-art performance (0.9051 Accuracy, 0.6224 F1-Score) on highly sparse, imbalanced interaction datasets.

## ✨ Key Features

* 🌐 **PMEA Entity Alignment:** A probabilistic knowledge-base pipeline to resolve nomenclature heterogeneity across isolated registries.

* 📐 **Manifold-Aware Transformer Gating (MATG):** Natively resolves hierarchical node collisions using hyperbolic geometric priors.

* ⚖️ **Calibrated Graph Focal Loss:** Mathematically optimized to handle extreme (1:5) negative class imbalances common in clinical incidence data.

* 🧠 **NeuMapper TDA Explainer:** An integrated Topological Data Analysis module to visually extract and prove the non-Euclidean, hierarchical branching of your latent spaces.

* 📊 **The DoTatLoi-714 Benchmark:** Included ready-to-use dataloaders for the digitized VTM clinical synergy hypergraph.

## ⚙️ Installation & Setup

We recommend using a virtual environment (e.g., `conda` or `venv`) to avoid dependency conflicts.

```bash
# 1. Clone the repository
git clone [https://github.com/vothikimanh/DoTatLoi-714-MATG.git](https://github.com/vothikimanh/DoTatLoi-714-MATG.git)
cd DoTatLoi-714-MATG

# 2. Create a virtual environment
conda create -n hypersynergy python=3.9
conda activate hypersynergy

# 3. Install the library in editable mode
pip install -e .

