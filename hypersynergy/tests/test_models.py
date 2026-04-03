import torch
import unittest
import numpy as np
from hypersynergy.models import MATG_Model, GraphFocalLoss

class TestMATGArchitecture(unittest.TestCase):
    def setUp(self):
        self.num_nodes = 100
        self.num_hyperedges = 20
        self.feat_dim = 22
        self.embed_dim = 12
        
        # Mock features
        self.vtm_feats = np.random.randn(self.num_nodes, self.feat_dim)
        self.tcm_feats = np.random.randn(self.num_nodes, self.feat_dim)
        self.form_feats = np.random.randn(self.num_hyperedges, self.feat_dim)
        
        self.model = MATG_Model(
            self.num_nodes, 
            self.num_hyperedges, 
            self.vtm_feats, 
            self.tcm_feats, 
            self.form_feats,
            mode='proposed',
            embed_dim=self.embed_dim
        )

    def test_model_initialization(self):
        """Verify that the model layers are correctly initialized with v82 settings."""
        self.assertEqual(self.model.embed_dim, 12)
        self.assertTrue(hasattr(self.model, 'decoder'))
        self.assertTrue(isinstance(self.model.node_top_emb, torch.nn.Embedding))

    def test_forward_pass(self):
        """Test the forward pass and output probability shapes."""
        node_idx = torch.LongTensor([0, 1, 2])
        edge_idx = torch.LongTensor([0, 0, 1])
        
        logits = self.model(node_idx, edge_idx)
        self.assertEqual(logits.shape, (3,))
        
        probs = torch.sigmoid(logits)
        self.assertTrue(torch.all(probs >= 0) and torch.all(probs <= 1))

    def test_focal_loss(self):
        """Ensure the focal loss handles class imbalance properly."""
        criterion = GraphFocalLoss(gamma=4.0, pos_weight=1.5)
        logits = torch.randn(10)
        targets = torch.randint(0, 2, (10,)).float()
        
        loss = criterion(logits, targets)
        self.assertGreater(loss.item(), 0)

if __name__ == '__main__':
    unittest.main()
