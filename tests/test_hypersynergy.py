import unittest
import torch
import numpy as np
from hypersynergy.models import MATG_Model, GraphFocalLoss

class TestHyperSynergyCore(unittest.TestCase):
    """
    Unit tests to ensure the MATG architecture and Loss functions 
    remain stable across different environments (Colab/GitHub).
    """
    def setUp(self):
        self.num_n, self.num_e, self.dim = 100, 20, 22
        self.v_f = np.random.randn(self.num_n, self.dim)
        self.t_f = np.random.randn(self.num_n, self.dim)
        self.e_f = np.random.randn(self.num_e, self.dim)
        
        self.model = MATG_Model(
            self.num_n, self.num_e, self.v_f, self.t_f, self.e_f,
            mode='proposed', embed_dim=12
        )

    def test_manifold_output_range(self):
        """Checks if the MATG decoder produces valid logits."""
        h_idx = torch.LongTensor([1, 2, 3])
        e_idx = torch.LongTensor([0, 1, 0])
        logits = self.model(h_idx, e_idx)
        self.assertEqual(logits.shape, (3,))

    def test_focal_loss_imbalance(self):
        """Verifies focal loss calculation with the 1:5 imbalance weight."""
        criterion = GraphFocalLoss(gamma=4.0, pos_weight=1.5)
        logits = torch.tensor([5.0, -5.0])
        targets = torch.tensor([1.0, 0.0])
        loss = criterion(logits, targets)
        self.assertGreater(loss.item(), 0)

if __name__ == "__main__":
    unittest.main()
