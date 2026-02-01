
# ============================================================================
# 2. CROSS-ATTENTION FUSION MODULE
# ============================================================================

import torch
import torch.nn as nn


class CrossAttentionFusion(nn.Module):
    """Cross-Attention Module for fusing spatial and frequency features"""
    
    def __init__(self, spatial_dim: int, freq_dim: int, hidden_dim: int = 256):
        super().__init__()
        
        # Projections for spatial features
        self.spatial_q = nn.Conv2d(spatial_dim, hidden_dim, 1)
        self.spatial_k = nn.Conv2d(spatial_dim, hidden_dim, 1)
        self.spatial_v = nn.Conv2d(spatial_dim, hidden_dim, 1)
        
        # Projections for frequency features
        self.freq_q = nn.Conv2d(freq_dim, hidden_dim, 1)
        self.freq_k = nn.Conv2d(freq_dim, hidden_dim, 1)
        self.freq_v = nn.Conv2d(freq_dim, hidden_dim, 1)
        
        # Multi-head attention
        self.attention = nn.MultiheadAttention(hidden_dim, num_heads=8, batch_first=False)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # FFN
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 4),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim * 4, hidden_dim)
        )
        
        # Output projection
        self.output_proj = nn.Conv2d(hidden_dim * 2, spatial_dim, 1)
        
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, spatial_feat: torch.Tensor, freq_feat: torch.Tensor) -> torch.Tensor:
        """
        Fuse spatial and frequency features using cross-attention
        
        Args:
            spatial_feat: [B, C_spatial, H, W]
            freq_feat: [B, C_freq, H, W]
            
        Returns:
            Fused features: [B, C_spatial, H, W]
        """
        B, C_s, H, W = spatial_feat.shape
        B, C_f, H_f, W_f = freq_feat.shape
        
        # Reshape to [B, C, H*W] -> [H*W, B, C] for attention
        spatial_flat = spatial_feat.view(B, C_s, -1).permute(2, 0, 1)
        freq_flat = freq_feat.view(B, C_f, -1).permute(2, 0, 1)
        
        # Project spatial features
        spatial_q = self.spatial_q(spatial_feat).view(B, -1, H*W).permute(2, 0, 1)
        spatial_k = self.spatial_k(spatial_feat).view(B, -1, H*W).permute(2, 0, 1)
        spatial_v = self.spatial_v(spatial_feat).view(B, -1, H*W).permute(2, 0, 1)
        
        # Project frequency features
        freq_q = self.freq_q(freq_feat).view(B, -1, H_f*W_f).permute(2, 0, 1)
        freq_k = self.freq_k(freq_feat).view(B, -1, H_f*W_f).permute(2, 0, 1)
        freq_v = self.freq_v(freq_feat).view(B, -1, H_f*W_f).permute(2, 0, 1)
        
        # Cross-attention: spatial queries attend to frequency keys/values
        attn_output, _ = self.attention(
            spatial_q, freq_k, freq_v,
            need_weights=False
        )

        # Use the projected spatial values as the residual (same hidden dim)
        spatial_proj = spatial_v  # [seq_len, B, hidden_dim]

        # Add & Norm using projected spatial features
        attn_output = self.norm1(spatial_proj + self.dropout(attn_output))

        # FFN
        ffn_output = self.ffn(attn_output)
        ffn_output = self.norm2(attn_output + self.dropout(ffn_output))

        # Reshape back: [seq_len, B, hidden] -> [B, hidden, H, W]
        ffn_output = ffn_output.permute(1, 2, 0).view(B, -1, H, W)
        spatial_proj = spatial_proj.permute(1, 2, 0).view(B, -1, H, W)

        # Concatenate projected spatial and fused features (both hidden_dim)
        combined = torch.cat([spatial_proj, ffn_output], dim=1)  # [B, 2*hidden, H, W]
        output = self.output_proj(combined)
        
        return output
