# WallCracksHardware - WAM-Net: Wavelet-Attention Mobile Network for Concrete Crack Detection

## 📋 Project Overview

**WAM-Net** (Wavelet-Attention Mobile Network) is a deep learning-based system for automated detection of cracks in concrete structures. This project implements a lightweight, efficient neural network architecture optimized for edge deployment on Qualcomm Snapdragon hardware, making it suitable for real-time crack detection in field conditions.

### Key Features

- **Dual-Branch Architecture**: Combines spatial and frequency domain features for robust crack detection
- **Wavelet Transform Integration**: Extracts high-frequency features that highlight crack patterns
- **Cross-Attention Fusion**: Intelligently fuses spatial and frequency features
- **Mobile-Optimized**: Based on MobileNetV3-Small for efficient inference
- **Edge Deployment Ready**: ONNX export for Qualcomm SNPE compatibility
- **GPU Accelerated**: Mixed-precision training with CUDA/MPS support
- **Comprehensive Evaluation**: Includes metrics, visualization, and benchmarking tools

## 🏗️ Architecture

WAM-Net employs a novel dual-branch architecture:

### 1. Spatial Branch
- Uses **MobileNetV3-Small** as backbone for efficient feature extraction
- Extracts spatial features from concrete images
- Optimized for mobile and edge devices

### 2. Frequency Branch
- **Haar Wavelet Transform**: Decomposes images into frequency sub-bands
- Captures high-frequency crack features that are often missed by spatial methods
- Custom CNN processes wavelet coefficients (12 channels: 3 RGB × 4 sub-bands)

### 3. Cross-Attention Fusion Module
- Multi-head attention mechanism (8 heads)
- Fuses spatial and frequency features adaptively
- Layer normalization and Feed-Forward Network (FFN)
- Projects features back to spatial dimension

### 4. Classifier
- Adaptive pooling followed by fully connected layers
- Binary classification: Crack vs. No Crack
- Dropout regularization (0.3) to prevent overfitting

## 📂 Repository Structure

```
WallCracksHardware/
├── README.md                    # This file
├── main.py                      # Main entry point for training/inference
├── WAMNet.py                    # WAM-Net model architecture
├── train.py                     # Training and inference engine
├── dataset.py                   # Data loading and augmentation
├── evaluate.py                  # Model evaluation and metrics
├── deploy.py                    # ONNX export for deployment
├── visualize.py                 # Visualization utilities
├── waveletTransform.py          # 2D Discrete Wavelet Transform layer
└── crossattention.py            # Cross-attention fusion module
```

### File Descriptions

#### `main.py`
Main script that orchestrates the entire pipeline:
- **Training mode**: End-to-end training with validation
- **Inference mode**: Single image prediction with visualization
- **Benchmark mode**: Performance evaluation (FPS, latency, memory)
- **Export mode**: ONNX model export for deployment

Key functions:
- `main()`: Complete training pipeline with data loading, model initialization, training, and evaluation
- `demo_inference()`: Demo inference on single images with attention map visualization
- `benchmark_model()`: Performance benchmarking with detailed metrics

#### `WAMNet.py`
Core model architecture implementation:
- `WAMNet` class: Main network architecture
  - Spatial branch using MobileNetV3-Small
  - Frequency branch with wavelet transform and custom CNN
  - Cross-attention fusion module
  - Classifier head
- `forward()`: Forward pass through the network
- `get_attention_maps()`: Extract attention maps for visualization
- Weight initialization with Kaiming normalization

#### `train.py`
Training and inference utilities:
- `Trainer` class: Manages training loop with GPU optimization
  - Mixed precision training (AMP) for CUDA
  - Early stopping with patience mechanism
  - Learning rate scheduling
  - Checkpoint management
  - Training history visualization
- `InferenceEngine` class: Optimized inference
  - Model loading and compilation
  - Single and batch prediction
  - PyTorch 2.0 compile support for faster inference

#### `dataset.py`
Data loading and preprocessing:
- `ConcreteCrackDataset`: Custom dataset class
  - Expected structure: `data/Positive/` and `data/Negative/`
  - Handles RGB and grayscale images
  - Automatic class mapping
- `get_data_loaders()`: Creates train/val/test splits (70/15/15)
  - Training augmentation: RandomResizedCrop, RandomHorizontalFlip, RandomRotation, ColorJitter
  - Validation/test: Resize + CenterCrop
  - ImageNet normalization
  - Optimized data loading with `pin_memory` and `persistent_workers`

#### `evaluate.py`
Comprehensive model evaluation:
- `evaluate_model()`: Complete evaluation suite
  - Accuracy, Precision, Recall, F1-Score
  - ROC-AUC calculation
  - Confusion matrix visualization
  - Classification report
  - Saves results as PNG images

#### `deploy.py`
Model export for edge deployment:
- `export_to_onnx()`: Exports trained model to ONNX format
  - ONNX opset 13
  - Dynamic batch size support
  - Optimized for Qualcomm SNPE
  - Includes deployment instructions for Snapdragon devices

#### `visualize.py`
Visualization utilities:
- `visualize_wavelet_transform()`: Visualizes wavelet decomposition
  - Shows all 4 sub-bands (LL, LH, HL, HH) for each RGB channel
  - Highlights high-frequency components
  - Overlays crack features on original image

#### `waveletTransform.py`
Discrete Wavelet Transform implementation:
- `WaveletTransform2D`: PyTorch module for 2D DWT
  - Haar wavelet decomposition
  - GPU-compatible implementation
  - Forward and inverse transforms
  - Uses grouped convolutions for efficiency
  - Non-trainable filters (registered as buffers)

#### `crossattention.py`
Cross-attention fusion module:
- `CrossAttentionFusion`: Attention-based feature fusion
  - Projects spatial and frequency features to hidden dimension
  - Multi-head attention (8 heads) for cross-modal fusion
  - Layer normalization for stable training
  - Feed-forward network for feature refinement
  - Residual connections

## 🚀 Installation

### Prerequisites
- Python 3.8+
- PyTorch 1.12+
- CUDA 11.0+ (optional, for GPU acceleration)

### Dependencies

```bash
pip install torch torchvision
pip install numpy matplotlib seaborn
pip install scikit-learn
pip install PyWavelets
pip install onnx
```

Or install all at once:

```bash
pip install torch torchvision numpy matplotlib seaborn scikit-learn PyWavelets onnx
```

## 📊 Dataset Structure

Organize your dataset in the following structure:

```
data/
├── Positive/       # Images with cracks
│   ├── crack_001.jpg
│   ├── crack_002.jpg
│   └── ...
└── Negative/       # Images without cracks
    ├── no_crack_001.jpg
    ├── no_crack_002.jpg
    └── ...
```

- **Image format**: JPG, PNG
- **Recommended size**: 224×224 or higher
- **Classes**: 
  - `Positive`: Concrete surfaces with visible cracks
  - `Negative`: Concrete surfaces without cracks

## 💻 Usage

### Training

Train the model on your dataset:

```bash
python main.py --mode train --data-dir data --learning-rate 1e-4
```

**Parameters:**
- `--data-dir`: Path to dataset directory (default: `data`)
- `--learning-rate`: Learning rate for optimizer (default: `1e-4`)

**Training configuration:**
- Batch size: 32
- Epochs: 50 (with early stopping patience of 10)
- Optimizer: AdamW with weight decay 1e-4
- Scheduler: CosineAnnealingLR
- Mixed precision training (automatic on CUDA)

**Output:**
- Model checkpoints saved in `wamnet_checkpoints/`
- Best model: `wamnet_checkpoints/best_model.pth`
- Training history plot: `wamnet_checkpoints/training_history.png`

### Inference

Run inference on a single image:

```bash
python main.py --mode infer --model-path wamnet_checkpoints/best_model.pth --image-path path/to/image.jpg
```

This will:
- Load the trained model
- Predict crack presence
- Display prediction probabilities
- Show attention maps highlighting crack regions
- Save visualization as `inference_visualization.png`

### Evaluation

Evaluate model on test set:

```bash
# Evaluation is automatically performed after training
# Or manually evaluate by running:
python main.py --mode train --data-dir data
```

**Metrics calculated:**
- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### Benchmarking

Measure inference performance:

```bash
python main.py --mode benchmark --model-path wamnet_checkpoints/best_model.pth --data-dir data
```

**Benchmark results:**
- Average inference time (ms)
- Frames per second (FPS)
- GPU memory usage (if CUDA)
- Model size (MB)

### Export for Deployment

Export model to ONNX format for Qualcomm SNPE:

```bash
python main.py --mode export --model-path wamnet_checkpoints/best_model.pth
```

**Qualcomm Deployment Steps:**

1. Convert ONNX to DLC:
```bash
snpe-onnx-to-dlc --input wamnet_quantcomm.onnx --output wamnet.dlc
```

2. Quantize to INT8:
```bash
snpe-dlc-quantize --input_dlc wamnet.dlc --output_dlc wamnet_quantized.dlc
```

3. Run on Snapdragon:
```bash
snpe-net-run --container wamnet_quantized.dlc --input_list input_list.txt
```

## 🎯 Model Performance

### Expected Metrics
- **Accuracy**: >95%
- **Inference Speed**: ~50-100 FPS (Snapdragon 855+)
- **Model Size**: ~5-10 MB
- **Input Size**: 224×224×3

### Hardware Compatibility
- **Training**: NVIDIA GPU (CUDA), Apple Silicon (MPS), or CPU
- **Deployment**: Qualcomm Snapdragon (via SNPE)
- **Minimum RAM**: 2GB for inference

## 🔧 Advanced Configuration

### Custom Training Parameters

Edit the configuration in `main.py`:

```python
config = {
    'data_dir': 'data',
    'batch_size': 32,           # Adjust based on GPU memory
    'num_workers': 4,           # Data loading workers
    'learning_rate': 1e-4,      # Learning rate
    'weight_decay': 1e-4,       # L2 regularization
    'epochs': 50,               # Maximum epochs
    'patience': 10              # Early stopping patience
}
```

### Modifying Architecture

To change the backbone or fusion parameters, edit `WAMNet.py`:

- **Backbone**: Change `models.mobilenet_v3_small` to other models
- **Wavelet type**: Change `wavelet='haar'` to 'db1', 'db2', etc.
- **Attention heads**: Modify `num_heads=8` in CrossAttentionFusion
- **Hidden dimension**: Adjust `hidden_dim=512` for fusion module

## 📈 Visualization

### Training History
Automatically generated after training:
- Loss curves (training and validation)
- Accuracy curves (training and validation)
- Saved as `wamnet_checkpoints/training_history.png`

### Attention Maps
Generated during inference:
- Spatial features from MobileNetV3
- High-frequency wavelet features
- Crack detection overlay
- Saved as `inference_visualization.png`

### Wavelet Decomposition
Visualize wavelet transform:

```python
from visualize import visualize_wavelet_transform
import matplotlib.pyplot as plt

image = plt.imread('path/to/image.jpg')
visualize_wavelet_transform(image)
```

## 🛠️ Troubleshooting

### Common Issues

1. **Out of Memory Error**
   - Reduce batch size in config
   - Use smaller input images
   - Disable mixed precision training

2. **Data Directory Not Found**
   - Ensure `data/Positive/` and `data/Negative/` exist
   - Check image file extensions (.jpg, .png)

3. **CUDA Not Available**
   - Install CUDA-enabled PyTorch
   - Or use CPU/MPS (Apple Silicon) mode

4. **Slow Training**
   - Enable CUDA if available
   - Increase `num_workers` for faster data loading
   - Use mixed precision training (automatic on CUDA)

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{wamnet2024,
  title={WAM-Net: Wavelet-Attention Mobile Network for Concrete Crack Detection},
  author={WallCracksHardware Team},
  year={2024},
  url={https://github.com/ridhamp4/WallCracksHardware}
}
```

## 📄 License

This project is open-source and available for academic and commercial use.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

## 📧 Contact

For questions or support, please open an issue on GitHub.

---

**Note**: This implementation is optimized for Qualcomm Snapdragon hardware but can run on any platform supporting PyTorch (CUDA, MPS, CPU).
