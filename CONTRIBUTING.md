# Developer Guide & Contributing: Building and Deploying HyperSynergy

This document outlines the step-by-step process used to transition the HyperG-TCM framework from a single Google Colab script into a modular, production-ready Python library suitable for a Q1 journal submission. It also serves as a guide for future researchers who wish to contribute to this repository.

## Step 1: The Repository Structure

A professional Python library requires a strict folder hierarchy. The HyperSynergy repository is structured as follows:

HyperSynergy/  
│  
├── hypersynergy/ # The actual Python library modules  
│ ├── \__init_\_.py # Exposes the main classes to the user  
│ ├── data.py # DoTatLoiBenchmark & data loaders  
│ ├── models.py # MATG_Model & Riemannian Decoders  
│ ├── losses.py # Calibrated Graph Focal Loss  
│ ├── explainers.py # NeuMapperExplainer (TDA)  
│ └── evaluation.py # ModelEvaluator & metrics  
│  
├── examples/ # Scripts for other researchers to use  
│ └── run_benchmark.py # The main training and evaluation script  
│  
├── data/ # Your raw CSV files go here  
│ ├── CongThuc_updated.csv  
│ └── ViThuoc_final.csv  
│  
├── tests/ # For testing the library architecture  
│ └── test_architecture.py  
│  
├── setup.py # Allows users to 'pip install' your library  
├── README.md # The front page of your GitHub  
├── LICENSE # MIT Open Source License  
└── CONTRIBUTING.md # This file

## Step 2: Modularizing the Code

To make the methodology "reusable" (as requested by peer reviewers), the monolithic analysis script is broken down into specific operational files inside the hypersynergy/ folder.

### 1\. hypersynergy/models.py (The Architecture)

This file contains the core Neural Network classes, specifically separating the hyperbolic routing from the standard semantic attention.
```python
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
<br/>class SafeMATG_Decoder(nn.Module):  
"""Riemannian Residual Gating (RRG) using Poincaré distance."""  
\# Implements the exact Poincaré log-map distance to penalize Euclidean collisions  
<br/>class MATG_Model(nn.Module):  
"""Manifold-Aware Transformer Gating Network."""  
\# Fuses the local Euclidean features with the Riemannian topological priors
```
### 2\. hypersynergy/losses.py (The Optimization)

This isolates the custom loss function, highlighting the specific mathematical solution for the 1:5 data sparsity problem.
```python
import torch  
import torch.nn as nn  
import torch.nn.functional as F  
<br/>class GraphFocalLoss(nn.Module):  
"""Calibrated loss to handle 1:5 class imbalance in synergy prediction."""  
\# Prevents majority negative samples from washing out the gradient
```
### 3\. hypersynergy/\__init_\_.py (The API Exposure)

This file is crucial. It defines the public-facing API when someone types from hypersynergy import ....
```python
from .models import MATG_Model  
from .losses import GraphFocalLoss  
from .data import DoTatLoiBenchmark  
from .explainers import NeuMapperExplainer  
from .evaluation import ModelEvaluator  
<br/>\_\_version\_\_ = "1.0.0"
```
## Step 3: Packaging the Library (setup.py)

To allow researchers to install the code easily across different environments (e.g., pip install -e .), the setup.py configuration dynamically manages dependencies:
```python
from setuptools import setup, find_packages  
<br/>setup(  
name="hypersynergy",  
version="1.0.0",  
author="Vo Thi Kim Anh",  
author_email="<vothikimanh@tdtu.edu.vn>",  
description="A Heterogeneous Hypergraph Framework for Synergy Prediction",  
packages=find_packages(),  
install_requires=\[  
"numpy",  
"pandas",  
"torch>=1.10.0",  
"scikit-learn",  
"matplotlib",  
"seaborn",  
"networkx",  
"matplotlib-venn"  
\],  
python_requires=">=3.8",  
)
```
## Step 4: How to Contribute

We welcome pull requests from the community. If you are extending the framework (for instance, applying it to new molecular synergy datasets or adding new TDA filters), please follow this workflow:
```bash
- **Fork** the repository on GitHub.
- **Clone** your fork locally.
- Create a new **branch** for your feature:  
   git checkout -b feature/your-new-feature

- **Commit** your changes with clear, descriptive messages:  
   git commit -m "Added advanced TDA overlap metric"

- **Push** to your branch:  
   git push origin feature/your-new-feature

- Open a **Pull Request** to the main branch of this repository.
```
### Bug Reports

If you encounter bugs, please open an **Issue** on GitHub. Tag it with bug and include a minimum reproducible example along with your stack trace.
