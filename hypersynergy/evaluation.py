import unittest
import torch
import numpy as np
import os
import sys

# Ensure the library is in the path for testing
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from hypersynergy.models import MATG_Model
from hypersynergy.losses import GraphFocalLoss
from hypersynergy.explainers import NeuMapperExplainer
from hypersynergy.data import DoTatLoiBenchmark

class TestHyperSynergyArchitecture(unittest.TestCase):
    """
    Unit tests to verify the structural integrity of the HyperSynergy library.
    Updated with realistic dimensions to catch index-swap errors.
    """

    def setUp(self):
        """Set up parameters matching the DoTatLoi-714 Benchmark."""
        self.num_nodes = 714       # Total Herbs
        self.num_hyperedges = 29    # Total Formulas
        self.feat_dim = 22
        self.embed_dim = 12
        
        # Create dummy feature tensors reflecting real-world sizes
        self.vtm_feats = np.random.randn(self.num_nodes, self.feat_dim)
        self.tcm_feats = np.random.randn(self.num_nodes, self.feat_dim)
        self.form_feats = np.random.randn(self.num_hyperedges, self.feat_dim)

    def test_model_initialization(self):
        """Test if the MATG model initializes correctly in different modes."""
        modes = ['proposed', 'gcn', 'gat']
        for mode in modes:
            model = MATG_Model(
                num_nodes=self.num_nodes,
                num_hyperedges=self.num_hyperedges,
                vtm_feats=self.vtm_feats,
                tcm_feats=self.tcm_feats,
                formula_feats=self.form_feats,
                mode=mode,
                embed_dim=self.embed_dim
            )
            self.assertIsInstance(model, MATG_Model)
            
    def test_model_forward_pass_correct_order(self):
        """
        CRITICAL TEST: Verifies forward pass with correct (Node, Hyperedge) order.
        If this fails with IndexError, check the tensor sizes in models.py.
        """
        model = MATG_Model(
            num_nodes=self.num_nodes,
            num_hyperedges=self.num_hyperedges,
            vtm_feats=self.vtm_feats,
            tcm_feats=self.tcm_feats,
            formula_feats=self.form_feats,
            mode='proposed',
            embed_dim=self.embed_dim
        )
        
        batch_size = 8
        # Correct order: model(herb_indices, formula_indices)
        herb_indices = torch.randint(0, self.num_nodes, (batch_size,))
        formula_indices = torch.randint(0, self.num_hyperedges, (batch_size,))
        
        try:
            logits = model(herb_indices, formula_indices)
            self.assertEqual(logits.shape, (batch_size,))
        except IndexError as e:
            self.fail(f"Forward pass failed with IndexError: {e}. Ensure indices are passed as (Herbs, Formulas).")

    def test_loss_function(self):
        """Test if GraphFocalLoss computes a valid scalar loss."""
        criterion = GraphFocalLoss(alpha=1.5, gamma=4.0)
        logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        
        loss = criterion(logits, targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0) 

    def test_explainer_logic(self):
        """Test if NeuMapperExplainer initializes properly."""
        explainer = NeuMapperExplainer(resolution=10, overlap=0.2)
        self.assertEqual(explainer.resolution, 10)
        self.assertEqual(explainer.overlap, 0.2)

    def test_data_loader_interface(self):
        """Test if the benchmark loader returns the correct structure."""
        results = DoTatLoiBenchmark.load_and_build_graph()
        self.assertEqual(len(results), 12)
        
        dataset = results[0]
        self.assertTrue(isinstance(dataset, np.ndarray))
        self.assertEqual(dataset.shape[1], 3)
        
        # Verify the mapping logic matches the model requirements
        num_formulas = results[4]
        num_herbs = results[5]
        self.assertTrue(np.all(dataset[:, 0] < num_formulas))
        self.assertTrue(np.all(dataset[:, 1] < num_herbs))

if __name__ == '__main__':
    unittest.main()
