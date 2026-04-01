import torch
import torch.nn as nn
import torch.nn.functional as F

class RiemannianResidualGating(nn.Module):
    """
    CONTRIBUTION 3.2: Riemannian Residual Gating (RRG)
    Implemented as Equation (2) in the manuscript.
    
    This engine resolves 'Dimensional Congestion' by gating semantic 
    hits with exact Poincaré manifold distances.
    """
    def __init__(self, embed_dim, curvature=1.5, alpha=0.7, beta=0.15):
        super(RiemannianResidualGating, self).__init__()
        # Manifold Hyperparameters (v82_Final verified)
        self.r = nn.Parameter(torch.tensor(5.0))         
        self.bilinear = nn.Bilinear(embed_dim, embed_dim, 1)
        self.curv = nn.Parameter(torch.tensor([curvature])) 
        self.alpha = nn.Parameter(torch.tensor([alpha]))   
        self.beta = beta # Residual scaling from v82 Colab
        
        # v82 Critical: Orthogonal init is required for hyperbolic stability
        nn.init.orthogonal_(self.bilinear.weight)

    def forward(self, u, e):
        """
        Calculates the gated synergy score based on hyperbolic separation.
        """
        # 1. Poincaré Distance Calculation d_P(u, e)
        # Numerical scaling to 0.90 for boundary stability in the Poincare Ball
        u_norm = F.normalize(u, p=2, dim=-1) * 0.90
        e_norm = F.normalize(e, p=2, dim=-1) * 0.90
        
        sqdist = torch.sum((u_norm - e_norm) ** 2, dim=-1)
        denom = torch.clamp((1 - torch.sum(u_norm**2, dim=-1)) * (1 - torch.sum(e_norm**2, dim=-1)), min=1e-5)
        
        # Hyperbolic log-map distance
        dist = torch.acosh(torch.clamp(1 + 2 * torch.abs(self.curv) * sqdist / denom, min=1.0001))
        
        # 2. Semantic Interaction (Standard Bilinear Matching)
        interaction = self.bilinear(u, e).squeeze(-1)
        
        # 3. Decision Gating (The v82 "Shattering" logic)
        # manifold_gate scales how much the model trusts the semantic match 
        # based on the hierarchical position in the manifold.
        manifold_gate = torch.exp(-dist / (torch.abs(self.alpha) + 1e-8))
        
        # Final Score: modulated interaction + topological residual
        return (interaction * manifold_gate) + (self.r - dist) * self.beta

class MATG_Model(nn.Module):
    """
    CONTRIBUTION 3: Manifold-Aware Transformer Gating (MATG) Framework.
    
    A heterogeneous framework that fuses aligned PMEA features with 
    Riemannian topological priors to predict clinical synergies.
    """
    def __init__(self, num_nodes, num_hyperedges, vtm_feats, tcm_feats, formula_feats, mode='proposed', embed_dim=12):
        super(MATG_Model, self).__init__()
        self.embed_dim = embed_dim
        self.mode = mode
        
        # --- PHASE A: PMEA Semantic Projection ---
        # Shared projection forces cross-ontology manifold alignment
        self.proj = nn.Sequential(
            nn.Linear(vtm_feats.shape[1], embed_dim), 
            nn.LayerNorm(embed_dim), 
            nn.GELU()
        )
        nn.init.orthogonal_(self.proj[0].weight)
        
        # Buffers for raw semantic features
        self.register_buffer('vtm_raw', torch.FloatTensor(vtm_feats))
        self.register_buffer('tcm_raw', torch.FloatTensor(tcm_feats))
        self.register_buffer('form_raw', torch.FloatTensor(formula_feats))
        
        self.pmea_fusion = nn.Sequential(nn.Linear(embed_dim * 2, embed_dim), nn.LayerNorm(embed_dim))
        
        # --- PHASE B: Topological Embedding ---
        self.node_top_emb = nn.Embedding(num_nodes, embed_dim)
        self.hyperedge_emb = nn.Embedding(num_hyperedges, embed_dim)
        nn.init.orthogonal_(self.node_top_emb.weight)
        nn.init.orthogonal_(self.hyperedge_emb.weight)
        
        self.dropout = nn.Dropout(0.4)
        
        # --- PHASE C: Manifold-Aware Gating (MATG) ---
        if mode in ['proposed', 'gat']:
            self.attn_gate = nn.Sequential(
                nn.Linear(embed_dim * 2, 64), 
                nn.ReLU(), 
                nn.Linear(64, 1)
            )
        
        # --- PHASE D: Riemannian Decoding ---
        if mode == 'proposed':
            self.decoder = RiemannianResidualGating(embed_dim)
        else:
            # Euclidean baseline fallback (GCN/GAT)
            self.decoder = nn.Bilinear(embed_dim, embed_dim, 1)

    def forward(self, node_indices, hyperedge_indices, return_attn=False):
        """
        Forward pass implementing the v82_Final fusion logic.
        """
        # 1. Semantic Flow (PMEA Alignment)
        v_vtm = self.proj(self.vtm_raw[node_indices])
        v_tcm = self.proj(self.tcm_raw[node_indices])
        h_sem = self.pmea_fusion(torch.cat([v_vtm, v_tcm], dim=-1))
        
        # 2. Topological Flow
        h_top = self.node_top_emb(node_indices)
        
        # 3. Feature-Level Attention Gating
        if self.mode in ['proposed', 'gat']:
            # Learns to balance clinical semantic vectors vs. graph identity
            alpha_gate = torch.sigmoid(self.attn_gate(torch.cat([h_top, h_sem], dim=-1)))
            h_fused = self.dropout(alpha_gate * h_top + (1 - alpha_gate) * h_sem)
        else:
            alpha_gate = None
            h_fused = self.dropout(h_top + h_sem)
            
        # Formula-side hyperedge embedding
        f_sem = self.proj(self.form_raw[hyperedge_indices])
        f_top = self.hyperedge_emb(hyperedge_indices)
        f_final = self.dropout(f_sem + f_top)
        
        # 4. Final Synergy Scoring
        if self.mode == 'proposed':
            logits = self.decoder(h_fused, f_final)
        else:
            logits = self.decoder(h_fused, f_final).squeeze(-1)
            
        if return_attn and self.mode in ['proposed', 'gat']:
            return logits, alpha_gate
        return logits
