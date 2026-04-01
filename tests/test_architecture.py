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
    """

    def setUp(self):
        """Set up dummy parameters for testing."""
        self.num_nodes = 714
        self.num_hyperedges = 29
        self.feat_dim = 22
        self.embed_dim = 12
        
        # Create dummy feature tensors
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
        """Test if a forward pass produces expected output shapes."""
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
        dummy_nodes = torch.randint(0, self.num_nodes, (batch_size,))
        dummy_edges = torch.randint(0, self.num_hyperedges, (batch_size,))
        
        logits = model(dummy_nodes, dummy_edges)
        self.assertEqual(logits.shape, (batch_size,))

    def test_loss_function(self):
        """Test if GraphFocalLoss computes a valid scalar loss."""
        criterion = GraphFocalLoss(alpha=1.5, gamma=4.0)
        logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        
        loss = criterion(logits, targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.dim(), 0) # Should be a scalar

    def test_explainer_logic(self):
        """Test if NeuMapperExplainer initializes and has the correct resolution."""
        explainer = NeuMapperExplainer(resolution=10, overlap=0.2)
        self.assertEqual(explainer.resolution, 10)
        self.assertEqual(explainer.overlap, 0.2)

    def test_data_loader_interface(self):
        """Test if the benchmark loader returns the correct number of objects."""
        # This tests the logic; it will use mock data if CSVs aren't present
        results = DoTatLoiBenchmark.load_and_build_graph()
        # Expecting a tuple of 12 items as defined in data.py
        self.assertEqual(len(results), 12)
        
        dataset = results[0]
        self.assertTrue(isinstance(dataset, np.ndarray))
        # Dataset columns: [formula_idx, herb_idx, label]
        self.assertEqual(dataset.shape[1], 3)

if __name__ == '__main__':
    unittest.main()
