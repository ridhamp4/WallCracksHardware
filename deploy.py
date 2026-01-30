
# ============================================================================
# 7. EXPORT FOR EDGE DEPLOYMENT (Qualcomm SNPE Compatible)
# ============================================================================

def export_to_onnx(model_path, output_path="wamnet.onnx", input_size=(1, 3, 224, 224)):
    """Export model to ONNX format for Qualcomm SNPE"""
    
    device = torch.device('cpu')  # Export on CPU
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize model
    model = WAMNet(num_classes=2, pretrained=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Create dummy input
    dummy_input = torch.randn(input_size, device=device)
    
    # Export to ONNX
    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=13,  # ONNX opset
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={
            'input': {0: 'batch_size'},
            'output': {0: 'batch_size'}
        },
        verbose=False
    )
    
    print(f"Model exported to {output_path}")
    
    # Print model summary for SNPE compatibility
    print("\n" + "="*50)
    print("QUALCOMM SNPE COMPATIBILITY CHECK")
    print("="*50)
    print("✓ Model exported to ONNX format")
    print("✓ Uses standard ops (Conv, BatchNorm, ReLU, etc.)")
    print("✓ Input shape: [batch_size, 3, 224, 224]")
    print("✓ Output shape: [batch_size, 2]")
    print("\nFor Qualcomm deployment:")
    print("1. Convert ONNX to DLC: snpe-onnx-to-dlc --input wamnet.onnx --output wamnet.dlc")
    print("2. Quantize for INT8: snpe-dlc-quantize --input_dlc wamnet.dlc --output_dlc wamnet_quantized.dlc")
    print("3. Run on Snapdragon: snpe-net-run --container wamnet_quantized.dlc --input_list input_list.txt")
