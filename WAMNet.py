
# ============================================================================
# 3. WAM-NET ARCHITECTURE
# ============================================================================

class WAMNet(nn.Module):
    """Wavelet-Attention Mobile Network for Crack Detection"""
    
    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        
        # ==================== Spatial Branch ====================
        # Use MobileNetV3-Small as backbone
        backbone = models.mobilenet_v3_small(pretrained=pretrained)
        
        # Extract feature layers
        self.spatial_features = nn.Sequential(
            backbone.features[:6],  # First 6 blocks
            nn.AdaptiveAvgPool2d((14, 14))  # Fixed output size
        )
        
        # ==================== Frequency Branch ====================
        self.wavelet_transform = WaveletTransform2D(wavelet='haar', level=1)
        
        self.freq_conv = nn.Sequential(
            # Input: 12 channels (3 RGB * 4 wavelet sub-bands)
            nn.Conv2d(12, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((14, 14))  # Match spatial branch size
        )
        
        # ==================== Fusion Block ====================
        spatial_dim = backbone.features[5].out_channels  # Output channels from spatial branch
        freq_dim = 128  # Output channels from frequency branch
        
        self.fusion = CrossAttentionFusion(
            spatial_dim=spatial_dim,
            freq_dim=freq_dim,
            hidden_dim=256
        )
        
        # ==================== Classifier ====================
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(spatial_dim, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes)
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights for custom layers"""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # ==================== Spatial Branch ====================
        spatial_feat = self.spatial_features(x)  # [B, C, 14, 14]
        
        # ==================== Frequency Branch ====================
        wavelet_coeffs = self.wavelet_transform(x)  # [B, 12, H/2, W/2]
        freq_feat = self.freq_conv(wavelet_coeffs)  # [B, 128, 14, 14]
        
        # ==================== Fusion ====================
        fused_feat = self.fusion(spatial_feat, freq_feat)
        
        # ==================== Classification ====================
        output = self.classifier(fused_feat)
        
        return output
    
    def get_attention_maps(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get attention maps for visualization"""
        with torch.no_grad():
            spatial_feat = self.spatial_features(x)
            wavelet_coeffs = self.wavelet_transform(x)
            freq_feat = self.freq_conv(wavelet_coeffs)
            
            # Get high-frequency sub-bands for visualization
            B, C, H, W = wavelet_coeffs.shape
            hf_coeffs = wavelet_coeffs[:, 3:12, :, :]  # High frequency bands
            
            return spatial_feat, hf_coeffs.mean(dim=1, keepdim=True)
