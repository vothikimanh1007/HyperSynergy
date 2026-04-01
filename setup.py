from setuptools import setup, find_packages

setup(
    name="hypersynergy",
    version="1.0.0",
    author="Vo Thi Kim Anh",
    author_email="vothikimanh@tdtu.edu.vn",
    description="A Heterogeneous Hypergraph Framework for Synergy Prediction",
    packages=find_packages(),
    install_requires=[
        "numpy",
        "pandas",
        "torch>=1.10.0",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "networkx",
        "matplotlib-venn"
    ],
    python_requires=">=3.8",
)
