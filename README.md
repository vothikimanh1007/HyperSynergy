# HyperSynergy: Topological Knowledge Bridging

Official implementation of the methodology detailed in: **"Topological Knowledge Bridging: The DoTatLoi-714 Benchmark and a Heterogeneous Hypergraph Framework for Cross-Cultural Herbal Synergy Prediction."**

## 📖 Overview

Standard Graph Neural Networks (GNNs) suffer from "Dimensional Congestion" when attempting to model complex, multi-entity biological synergies in flat Euclidean space. **HyperSynergy** is a domain-agnostic algorithmic library that maps heterogeneous bipartite hypergraphs into a Poincaré hyperbolic manifold.

Powered by **Manifold-Aware Transformer Gating (MATG)** and **Riemannian Residual Gating (RRG)**, the library dynamically resolves hierarchical node collisions using exact Poincaré log-map distances, achieving state-of-the-art performance (0.9051 Accuracy, 0.6224 F1-Score) on highly sparse, imbalanced interaction datasets.

## 🧠 Problem Formulation

We formulate herbal synergy prediction as a **Node-Hyperedge Incidence Prediction** task on a heterogeneous hypergraph .

- **Nodes ():** 714 medicinal herbs (entities).
- **Hyperedges ():** 150 multi-herb formulations (groups).
- **Goal:** Predict the probability  of a synergy link between entity  and group .

## ✨ Key Features

- 🌐 **PMEA Entity Alignment:** A probabilistic knowledge-base pipeline to resolve nomenclature heterogeneity across isolated registries.
- 📐 **Manifold-Aware Transformer Gating (MATG):** Natively resolves hierarchical node collisions using hyperbolic geometric priors.
- ⚖️ **Calibrated Graph Focal Loss:** Mathematically optimized () to handle extreme (1:5) negative class imbalances.
- 🧠 **NeuMapper TDA Explainer:** An integrated Topological Data Analysis module to visually extract and prove the non-Euclidean, hierarchical branching of latent spaces.
- 📊 **DoTatLoi-714 Benchmark:** Ready-to-use dataloaders for the digitized VTM clinical synergy hypergraph.

## 📂 Repository Structure

HyperSynergy/  
├── hypersynergy/ # Core Library (models, data loaders)  
├── weights/ # Pre-trained Weights (.pth files)  
├── data/  
│ └── raw/ # Benchmark CSV files  
├── examples/  
│ ├── generate_simulation.py # Domain-agnostic data simulation  
│ └── predict_custom.py # Inference on new datasets  
├── tests/ # Unit tests for CI/CD pipelines  
├── notebooks/ # Tutorial notebooks (Colab-ready)  
├── CONTRIBUTING.md # Contribution guidelines  
├── LICENSE # MIT License  
├── setup.py # Package configuration  
└── requirements.txt # Dependency list  

## ⚙️ Installation

\# 1. Clone the repository  
git clone [[...]] cd hypersynergy


cd HyperSynergy  
<br/>\# 2. Install dependencies  
pip install -r requirements.txt  
<br/>\# 3. Install the library in editable mode  
pip install -e .  

## 🚀 Usage & Inference

### Using Pre-trained Weights

The repository includes the v82_Final weights. You can perform synergy inference on custom domain data or simulated datasets using the provided predictor:

\# Run custom inference using pre-trained weights  
python examples/predict_custom.py --weights weights/Proposed_MATG_Ours_v82_Final_MATG_Best.pth  

### Data Simulation for Different Domains

To test the model on different domain knowledge (e.g., drug-drug synergy or food pairings), generate a simulated dataset:

\# Generate a simulated pharma-domain dataset  
python examples/generate_simulation.py --domain Pharma  

## 📊 Performance Benchmark (v82)

| **Model** | **Accuracy** | **F1-Score** | **ROC-AUC** |
| --- | --- | --- | --- |
| GCN Baseline |  0.8538 ± 0.024   | 0.4909 | 0.8504 |
| GAT Attentive | 0.8821 ± 0.022    | 0.5468 | 0.8533 |
| **MATG (Proposed)** |  **0.9051 ± 0.024**   | **0.6224** | **0.8329** |

## 📖 Citation

@article{anh2026topological,  
title={Topological Knowledge Bridging: The DoTatLoi-714 Benchmark and a Heterogeneous Hypergraph Framework for Cross-Cultural Herbal Synergy Prediction},  
author={},  
journal={(Under Review)},  
year={2026}  
}
