
# ============================================================================
# 1. WAVELET TRANSFORM LAYER (CUDA/MPS Compatible)
# ============================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pywt


class WaveletTransform2D(nn.Module):
    """2D Discrete Wavelet Transform Layer for PyTorch"""
    
    def __init__(self, wavelet='haar', level=1):
        super().__init__()
        self.wavelet = wavelet
        self.level = level
        
        # Precompute wavelet filters
        self._init_wavelet_filters()
    
    def _init_wavelet_filters(self):
        """Initialize wavelet filters as convolutional weights"""
        # Get 1D wavelet filters
        w = pywt.Wavelet(self.wavelet)
        
        # 1D decomposition filters
        dec_lo = np.array(w.dec_lo, dtype=np.float32)
        dec_hi = np.array(w.dec_hi, dtype=np.float32)
        
        # Create 2D filters through outer product
        LL = np.outer(dec_lo, dec_lo)
        LH = np.outer(dec_lo, dec_hi)
        HL = np.outer(dec_hi, dec_lo)
        HH = np.outer(dec_hi, dec_hi)
        
        # Stack filters for RGB channels into grouped filters for conv2d with groups=3
        base_filters = np.stack([LL, LH, HL, HH], axis=0)  # [4, h, w]
        # Tile for each input channel: [3, 4, h, w] -> reshape to [12, 1, h, w]
        tiled = np.tile(base_filters[np.newaxis, :, :, :], (3, 1, 1, 1))  # [3,4,h,w]
        filters = tiled.reshape(-1, base_filters.shape[1], base_filters.shape[2])  # [12, h, w]
        filters = filters[:, np.newaxis, :, :]  # [12, 1, h, w]
        # Register as non-trainable buffer
        self.register_buffer('filters', torch.from_numpy(filters))
        
        # For inverse transform
        rec_lo = np.array(w.rec_lo, dtype=np.float32)
        rec_hi = np.array(w.rec_hi, dtype=np.float32)
        
        LL_inv = np.outer(rec_lo, rec_lo)
        LH_inv = np.outer(rec_lo, rec_hi)
        HL_inv = np.outer(rec_hi, rec_lo)
        HH_inv = np.outer(rec_hi, rec_hi)
        
        inv_base = np.stack([LL_inv, LH_inv, HL_inv, HH_inv], axis=0)  # [4, h, w]
        inv_tiled = np.tile(inv_base[np.newaxis, :, :, :], (3, 1, 1, 1))  # [3,4,h,w]
        inv_filters = inv_tiled.reshape(-1, inv_base.shape[1], inv_base.shape[2])  # [12, h, w]
        inv_filters = inv_filters[:, np.newaxis, :, :]  # [12, 1, h, w]
        self.register_buffer('inv_filters', torch.from_numpy(inv_filters))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply 2D DWT to input tensor
        
        Args:
            x: Input tensor of shape [B, 3, H, W]
            
        Returns:
            Wavelet coefficients of shape [B, 12, H//2, W//2]
        """
        batch_size = x.shape[0]
        
        # Apply convolution with stride 2 (downsampling)
        coeffs = F.conv2d(x, self.filters, stride=2, groups=3)
        
        # Rearrange to combine all sub-bands
        B, C, H, W = coeffs.shape
        # Reshape to [B, 3, 4, H, W] -> [B, 12, H, W]
        coeffs = coeffs.view(B, 3, 4, H, W).view(B, 12, H, W)
        
        return coeffs
    
    def inverse(self, coeffs: torch.Tensor) -> torch.Tensor:
        """
        Apply inverse 2D DWT
        
        Args:
            coeffs: Wavelet coefficients of shape [B, 12, H, W]
            
        Returns:
            Reconstructed image of shape [B, 3, H*2, W*2]
        """
        B, C, H, W = coeffs.shape
        
        # coeffs expected shape [B, 12, H, W]
        # Apply transposed convolution with groups=3 to map 12->3 channels
        output = F.conv_transpose2d(
            coeffs,
            self.inv_filters,
            stride=2,
            groups=3
        )
        
        return output