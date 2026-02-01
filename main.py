import os
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
import torchvision.transforms.functional as TF

import pywt
import pywt.data
from typing import Tuple, Optional, List

# Local modules
from WAMNet import WAMNet
from train import Trainer, InferenceEngine
from dataset import get_data_loaders
from evaluate import evaluate_model
from deploy import export_to_onnx

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 
                     'mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Using device: {device}")

# For reproducibility
torch.manual_seed(42)
np.random.seed(42)




# ============================================================================
# 9. MAIN TRAINING SCRIPT
# ============================================================================

def main():
    """Main training and evaluation pipeline"""
    
    print("="*50)
    print("WAM-NET: Wavelet-Attention Mobile Network")
    print(f"Using device: {device}")
    print("="*50)
    
    # Configuration
    config = {
        'data_dir': 'concrete_crack_data',  # Update this path
        'batch_size': 32,
        'num_workers': 4 if device.type == 'cuda' else 2,
        'learning_rate': 0.001,
        'weight_decay': 1e-4,
        'epochs': 50,
        'patience': 10
    }
    
    # Step 1: Load data
    print("\n[1/5] Loading data...")
    train_loader, val_loader, test_loader = get_data_loaders(
        config['data_dir'],
        batch_size=config['batch_size'],
        num_workers=config['num_workers']
    )
    
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples: {len(val_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    
    # Step 2: Initialize model
    print("\n[2/5] Initializing WAM-Net model...")
    model = WAMNet(num_classes=2, pretrained=True)
    print(f"  Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"  Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Step 3: Setup training
    print("\n[3/5] Setting up training...")
    
    # Loss function with class weighting (if imbalanced)
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer with gradient clipping
    optimizer = optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=config['epochs'],
        eta_min=1e-6
    )
    
    # Initialize trainer
    trainer = Trainer(model, device, checkpoint_dir='wamnet_checkpoints')
    
    # Step 4: Train model
    print("\n[4/5] Training model...")
    print("-" * 50)
    
    best_val_acc = trainer.train(
        train_loader,
        val_loader,
        criterion,
        optimizer,
        scheduler,
        epochs=config['epochs'],
        early_stopping_patience=config['patience']
    )
    
    print(f"\nBest validation accuracy: {best_val_acc:.2f}%")
    
    # Plot training history
    trainer.plot_training_history()
    
    # Step 5: Evaluate on test set
    print("\n[5/5] Evaluating model...")
    model_path = trainer.checkpoint_dir / 'best_model.pth'
    
    if model_path.exists():
        metrics = evaluate_model(str(model_path), test_loader, device)
        
        # Export to ONNX for Qualcomm deployment
        print("\nExporting model for Qualcomm SNPE...")
        export_to_onnx(str(model_path), "wamnet.onnx")
    else:
        print("Model checkpoint not found!")
    
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)

# ============================================================================
# 10. DEMO INFERENCE SCRIPT
# ============================================================================

def demo_inference(model_path="wamnet_checkpoints/best_model.pth", image_path=None):
    """Demo inference on a single image"""
    
    print("\n" + "="*50)
    print("DEMO INFERENCE")
    print("="*50)
    
    # Load inference engine
    inference_engine = InferenceEngine(model_path, device)
    
    # Load or generate test image
    if image_path and os.path.exists(image_path):
        image = plt.imread(image_path)
    else:
        # Generate a synthetic concrete image for demo
        image = np.random.randint(100, 200, (227, 227, 3), dtype=np.uint8)
        # Add a synthetic crack
        image[100:120, :, :] = 50  # Dark line
    
    # Predict
    result = inference_engine.predict_single(image)
    
    # Display results
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"Input Image (227x227)")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    classes = ['No Crack', 'Crack']
    colors = ['green', 'red']
    
    bars = plt.bar(classes, result['probabilities'], color=colors)
    plt.ylim(0, 1)
    plt.title('Prediction Probabilities')
    plt.ylabel('Probability')
    
    # Add probability values on bars
    for bar, prob in zip(bars, result['probabilities']):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{prob:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.show()
    
    # Print results
    print(f"\nPrediction: {classes[result['prediction']]}")
    print(f"Confidence: {result['confidence']:.3f}")
    print(f"\nProbabilities:")
    for i, (cls, prob) in enumerate(zip(classes, result['probabilities'])):
        print(f"  {cls}: {prob:.4f}")
    
    # Show wavelet features if available
    if hasattr(inference_engine.model, 'get_attention_maps'):
        print("\nExtracting wavelet features...")
        image_tensor = inference_engine.transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            spatial_feat, hf_map = inference_engine.model.get_attention_maps(image_tensor)
        
        hf_map_np = hf_map.squeeze().cpu().numpy()
        
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        axes[1].imshow(hf_map_np, cmap='hot')
        axes[1].set_title('High-Frequency Features')
        axes[1].axis('off')
        
        # Overlay on original
        axes[2].imshow(image)
        axes[2].imshow(hf_map_np, cmap='hot', alpha=0.5)
        axes[2].set_title('Crack Detection Overlay')
        axes[2].axis('off')
        
        plt.tight_layout()
        plt.savefig('inference_visualization.png', dpi=150)
        plt.show()

# ============================================================================
# 11. PERFORMANCE BENCHMARKING
# ============================================================================

def benchmark_model(model_path, test_loader, num_iterations=100):
    """Benchmark model performance"""
    
    print("\n" + "="*50)
    print("PERFORMANCE BENCHMARKING")
    print("="*50)
    
    inference_engine = InferenceEngine(model_path, device)
    model = inference_engine.model
    
    # Warm-up
    print("Warming up...")
    dummy_input = torch.randn(1, 3, 224, 224).to(device)
    for _ in range(10):
        _ = model(dummy_input)
    
    if device.type == 'cuda':
        torch.cuda.synchronize()
    
    # Benchmark inference speed
    print("Benchmarking inference speed...")
    
    import time
    times = []
    
    with torch.no_grad():
        for _ in range(num_iterations):
            start_time = time.perf_counter()
            
            if device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    _ = model(dummy_input)
                torch.cuda.synchronize()
            else:
                _ = model(dummy_input)
            
            end_time = time.perf_counter()
            times.append((end_time - start_time) * 1000)  # Convert to ms
    
    # Calculate statistics
    avg_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000 / avg_time
    
    print(f"\nInference Performance:")
    print(f"  Average time: {avg_time:.2f} ± {std_time:.2f} ms")
    print(f"  FPS: {fps:.1f}")
    print(f"  Device: {device}")
    
    # Memory usage (CUDA only)
    if device.type == 'cuda':
        memory_allocated = torch.cuda.max_memory_allocated() / 1024**2
        memory_reserved = torch.cuda.max_memory_reserved() / 1024**2
        print(f"  Max memory allocated: {memory_allocated:.1f} MB")
        print(f"  Max memory reserved: {memory_reserved:.1f} MB")
    
    # Model size
    import os
    model_size = os.path.getsize(model_path) / 1024**2
    print(f"  Model size: {model_size:.2f} MB")
    
    return {
        'avg_inference_time_ms': avg_time,
        'fps': fps,
        'model_size_mb': model_size
    }

# ============================================================================
# 12. RUN SCRIPT
# ============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='WAM-Net for Concrete Crack Detection')
    parser.add_argument('--mode', type=str, default='train', 
                       choices=['train', 'infer', 'benchmark', 'export'],
                       help='Mode to run: train, infer, benchmark, or export')
    parser.add_argument('--model-path', type=str, 
                       default='wamnet_checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--image-path', type=str, 
                       help='Path to image for inference')
    parser.add_argument('--data-dir', type=str, 
                       default='concrete_crack_data',
                       help='Path to dataset directory')
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        # Update config
        import sys
        sys.path.insert(0, '.')
        
        # Create data directory if it doesn't exist
        if not os.path.exists(args.data_dir):
            print(f"Data directory {args.data_dir} not found!")
            print("Please create a directory structure:")
            print(f"  {args.data_dir}/Positive/  # Crack images")
            print(f"  {args.data_dir}/Negative/  # No-crack images")
            exit(1)
        
        main()
    
    elif args.mode == 'infer':
        demo_inference(args.model_path, args.image_path)
    
    elif args.mode == 'benchmark':
        # Load test data
        _, _, test_loader = get_data_loaders(
            args.data_dir,
            batch_size=32,
            num_workers=0
        )
        benchmark_model(args.model_path, test_loader)
    
    elif args.mode == 'export':
        export_to_onnx(args.model_path, "wamnet_quantcomm.onnx")
    
    print("\n" + "="*50)
    print("WAM-Net Implementation Complete")
    print("="*50)