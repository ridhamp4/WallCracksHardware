
# ============================================================================
# 8. VISUALIZATION UTILITIES
# ============================================================================

def visualize_wavelet_transform(image_array):
    """Visualize wavelet decomposition of an image"""
    
    # Convert to tensor
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    image_tensor = transform(image_array).unsqueeze(0)
    
    # Initialize wavelet transform
    wavelet = WaveletTransform2D(wavelet='haar')
    
    # Apply transform
    coeffs = wavelet(image_tensor)
    
    # Reshape coefficients for visualization
    B, C, H, W = coeffs.shape
    coeffs_reshaped = coeffs.view(3, 4, H, W)  # [3, 4, H, W]
    
    # Create visualization
    fig, axes = plt.subplots(3, 4, figsize=(12, 9))
    
    # Sub-band names
    subband_names = ['LL', 'LH', 'HL', 'HH']
    
    for i in range(3):  # RGB channels
        for j in range(4):  # Wavelet sub-bands
            coeff = coeffs_reshaped[i, j].detach().numpy()
            
            axes[i, j].imshow(coeff, cmap='gray' if i == 0 else None)
            axes[i, j].set_title(f'Channel {i}, {subband_names[j]}')
            axes[i, j].axis('off')
    
    plt.suptitle('Wavelet Transform Sub-bands (RGB Channels)', fontsize=16)
    plt.tight_layout()
    plt.savefig('wavelet_transform.png', dpi=150)
    plt.show()
    
    # Show high-frequency components
    hf_coeffs = coeffs[:, 3:12, :, :].mean(dim=1, keepdim=True)
    hf_image = hf_coeffs.squeeze().detach().numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    axes[0].imshow(image_array)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    axes[1].imshow(hf_image, cmap='hot')
    axes[1].set_title('High-Frequency Components (Crack Features)')
    axes[1].axis('off')
    
    plt.tight_layout()
    plt.savefig('crack_features.png', dpi=150)
    plt.show()
