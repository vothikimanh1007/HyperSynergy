import torch
import torch.nn as nn
import torch.nn.functional as F

class RiemannianResidualGating(nn.Module):
    """
    Implementation of Equation (2) from the paper: 
    Manifold-Aware Transformer Gating (MATG) with Exponential Decay.
    
    Revised for Stability: Ensures the gate does not collapse under high weight 
    decay (1e-1) by using a more permissive initial gating factor.
    """
    def __init__(self, embed_dim):
        super(RiemannianResidualGating, self).__init__()
        # Manifold parameters (Optimized to prevent Fold-specific crashes)
        self.r = nn.Parameter(torch.tensor(5.0))         
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, 1)
        self.curv = nn.Parameter(torch.tensor([1.0]))    # Initial curvature (c=1.0)
        self.alpha = nn.Parameter(torch.tensor([1.5]))   # Increased initial alpha to keep gate open
        self.beta = nn.Parameter(torch.tensor([0.15]))   # Residual weight
        
        # Explicit initialization to resist heavy weight decay
        nn.init.xavier_normal_(self.bilinear.weight)

    def forward(self, u, e):
        """
        Calculates the gated synergy score based on hyperbolic separation.
        """
        # 1. Poincaré Distance Calculation d_P(u, e)
        # Numerical clamping is critical for acosh stability on CPU
        u_norm = F.normalize(u, p=2, dim=-1) * 0.92
        e_norm = F.normalize(e, p=2, dim=-1) * 0.92
        
        sqdist = torch.sum((u_norm - e_norm) ** 2, dim=-1)
        denom = torch.clamp((1 - torch.sum(u_norm**2, dim=-1)) * (1 - torch.sum(e_norm**2, dim=-1)), min=1e-6)
        
        # dist is the manifold separation metric
        dist = torch.acosh(torch.clamp(1 + 2 * torch.abs(self.curv) * sqdist / denom, min=1.0001, max=15.0))
        
        # 2. Semantic Interaction (The core synergism signal)
        interaction = self.bilinear(u, e).squeeze(-1)
        
        # 3. Riemannian Residual Gating Logic
        # manifold_gate scales how much the model 'trusts' the semantic match 
        # based on the hierarchical distance in the Poincare ball.
        manifold_gate = torch.exp(-dist / (torch.abs(self.alpha) + 1e-5))
        
        # Residual term provides gradient even if the gate is closed
        residual = (self.r - dist) * self.beta
        
        return (interaction * manifold_gate) + residual

class MATG_Model(nn.Module):
    """
    The Heterogeneous Hypergraph Framework: HyperG-TCM.
    
    Fuses local VTM features with global TCMID features and learns a manifold-aware
    projection to resolve the 'Identity Bottleneck' and 'Dimensional Congestion'.
    """
    def __init__(self, num_nodes, num_hyperedges, vtm_feats, tcm_feats, formula_feats, mode='proposed', embed_dim=12):
        super(MATG_Model, self).__init__()
        self.embed_dim = embed_dim
        self.mode = mode
        
        # 1. Shared Projection (Orthogonal init ensures manifold stability)
        self.proj = nn.Sequential(
            nn.Linear(vtm_feats.shape[1], embed_dim), 
            nn.LayerNorm(embed_dim), 
            nn.GELU()
        )
        nn.init.orthogonal_(self.proj[0].weight)
        
        # 2. PMEA Data Buffers (Aligned features)
        self.register_buffer('vtm_raw', torch.FloatTensor(vtm_feats))
        self.register_buffer('tcm_raw', torch.FloatTensor(tcm_feats))
        self.register_buffer('form_raw', torch.FloatTensor(formula_feats))
        
        self.pmea_fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim))
        
        # 3. Topological Embeddings
        self.node_top_emb = nn.Embedding(num_nodes, embed_dim)
        self.hyperedge_emb = nn.Embedding(num_hyperedges, embed_dim)
        nn.init.orthogonal_(self.node_top_emb.weight)
        nn.init.orthogonal_(self.hyperedge_emb.weight)
        
        self.dropout = nn.Dropout(0.4)
        
        # 4. Feature-Level Attention Gate (MATG)
        if mode in ['proposed', 'gat']:
            self.attn_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, 64), 
                nn.ReLU(), 
                nn.Linear(64, 1)
            )
        
        # 5. Manifold-Aware Decoder (RRG) vs Baselines
        if mode == 'proposed':
            self.decoder = RiemannianResidualGating(embed_dim)
        else:
            # Standard Euclidean Bilinear for Baselines
            self.decoder = nn.Bilinear(embed_dim, embed_dim, 1)

    def forward(self, node_indices, hyperedge_indices, return_attn=False):
        # --- A. Semantic Feature Processing (PMEA Alignment) ---
        v_vtm = self.proj(self.vtm_raw[node_indices])
        v_tcm = self.proj(self.tcm_raw[node_indices])
        h_sem = self.pmea_fusion(torch.cat([v_vtm, v_tcm], dim=-1))
        
        # --- B. Topological Embedding Processing ---
        h_top = self.node_top_emb(node_indices)
        
        # --- C. Feature-Level Gating (MATG) ---
        if self.mode in ['proposed', 'gat']:
            # Learns to balance topology (identity) vs. semantics (clinical properties)
            alpha_gate = torch.sigmoid(self.attn_gate(torch.cat([h_top, h_sem], dim=-1)))
            h_fused = self.dropout(alpha_gate * h_top + (1 - alpha_gate) * h_sem)
        else:
            alpha_gate = None
            h_fused = self.dropout(h_top + h_sem)
            
        # Formula Projection
        f_sem = self.proj(self.form_raw[hyperedge_indices])
        f_top = self.hyperedge_emb(hyperedge_indices)
        f_final = self.dropout(f_sem + f_top)
        
        # --- D. Riemannian Decoding ---
        if self.mode == 'proposed':
            logits = self.decoder(h_fused, f_final)
        else:
            logits = self.decoder(h_fused, f_final).squeeze(-1)
            
        if return_attn and self.mode in ['proposed', 'gat']:
            return logits, alpha_gate
        return logits
