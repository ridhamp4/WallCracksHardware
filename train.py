
# ============================================================================
# 5. TRAINING FUNCTIONS WITH GPU OPTIMIZATION
# ============================================================================

import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from WAMNet import WAMNet


class Trainer:
    """Training and validation manager with GPU optimization"""
    
    def __init__(self, model, device, checkpoint_dir='checkpoints'):
        self.model = model.to(device)
        self.device = device
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(exist_ok=True)
        
        # Mixed precision training for faster computation
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
        
        # Track metrics
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
        
    def train_epoch(self, train_loader, criterion, optimizer, scheduler=None):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            optimizer.zero_grad(set_to_none=True)  # Faster zero_grad
            
            # Mixed precision training for CUDA
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(optimizer)
                self.scaler.update()
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            
            # Update learning rate scheduler
            if scheduler:
                scheduler.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            # Progress printing
            if batch_idx % 10 == 0:
                print(f'  Batch {batch_idx}/{len(train_loader)}: Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    @torch.no_grad()
    def validate(self, val_loader, criterion):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        all_preds = []
        all_targets = []
        
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = criterion(outputs, targets)
            else:
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        epoch_loss = running_loss / len(val_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc, all_preds, all_targets
    
    def train(self, train_loader, val_loader, criterion, optimizer, 
              scheduler=None, epochs=50, early_stopping_patience=10):
        """Full training loop"""
        
        best_val_acc = 0
        patience_counter = 0
        
        for epoch in range(epochs):
            print(f'\nEpoch {epoch+1}/{epochs}')
            print('-' * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(
                train_loader, criterion, optimizer, scheduler
            )
            
            # Validate
            val_loss, val_acc, val_preds, val_targets = self.validate(
                val_loader, criterion
            )
            
            # Save metrics
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                self.save_checkpoint(f'best_model.pth', epoch, val_acc)
                print(f'  -> New best model saved!')
            else:
                patience_counter += 1
                print(f'  -> No improvement for {patience_counter} epochs')
            
            # Early stopping
            if patience_counter >= early_stopping_patience:
                print(f'Early stopping triggered at epoch {epoch+1}')
                break
        
        # Load best model
        self.load_checkpoint('best_model.pth')
        
        return best_val_acc
    
    def save_checkpoint(self, filename, epoch, val_acc):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': None,  # Don't save optimizer for inference
            'val_acc': val_acc,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accs': self.train_accs,
            'val_accs': self.val_accs
        }
        
        torch.save(checkpoint, self.checkpoint_dir / filename)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint"""
        checkpoint = torch.load(self.checkpoint_dir / filename, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        if 'train_losses' in checkpoint:
            self.train_losses = checkpoint['train_losses']
            self.val_losses = checkpoint['val_losses']
            self.train_accs = checkpoint['train_accs']
            self.val_accs = checkpoint['val_accs']
        
        print(f"Loaded checkpoint from epoch {checkpoint['epoch']} with val_acc {checkpoint['val_acc']:.2f}%")
    
    def plot_training_history(self):
        """Plot training history"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        axes[0].plot(self.train_losses, label='Train Loss')
        axes[0].plot(self.val_losses, label='Val Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Loss')
        axes[0].set_title('Training and Validation Loss')
        axes[0].legend()
        axes[0].grid(True)
        
        # Accuracy plot
        axes[1].plot(self.train_accs, label='Train Acc')
        axes[1].plot(self.val_accs, label='Val Acc')
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel('Accuracy (%)')
        axes[1].set_title('Training and Validation Accuracy')
        axes[1].legend()
        axes[1].grid(True)
        
        plt.tight_layout()
        plt.savefig(self.checkpoint_dir / 'training_history.png', dpi=150)
        plt.show()

# ============================================================================
# 6. INFERENCE AND TESTING
# ============================================================================

class InferenceEngine:
    """Optimized inference engine for WAM-Net"""
    
    def __init__(self, model_path, device):
        self.device = device
        self.model = self.load_model(model_path)
        
        # Inference transform
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def load_model(self, model_path):
        """Load trained model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Initialize model
        model = WAMNet(num_classes=2, pretrained=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.to(self.device)
        model.eval()
        
        # Compile model for faster inference (PyTorch 2.0+)
        if hasattr(torch, 'compile'):
            model = torch.compile(model, mode='reduce-overhead')
        
        return model
    
    @torch.no_grad()
    def predict_single(self, image_array):
        """Predict single image"""
        # Convert to tensor
        if isinstance(image_array, np.ndarray):
            image_tensor = self.transform(image_array).unsqueeze(0)
        else:
            image_tensor = image_array
        
        # Move to device
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        if self.device.type == 'cuda':
            with torch.cuda.amp.autocast():
                outputs = self.model(image_tensor)
        else:
            outputs = self.model(image_tensor)
        
        # Get prediction
        probabilities = F.softmax(outputs, dim=1)
        confidence, prediction = probabilities.max(1)
        
        return {
            'prediction': prediction.item(),
            'confidence': confidence.item(),
            'probabilities': probabilities.cpu().numpy()[0]
        }
    
    @torch.no_grad()
    def predict_batch(self, dataloader):
        """Predict batch of images"""
        self.model.eval()
        all_predictions = []
        all_confidences = []
        all_targets = []
        
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            if self.device.type == 'cuda':
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
            else:
                outputs = self.model(inputs)
            
            probabilities = F.softmax(outputs, dim=1)
            confidence, predictions = probabilities.max(1)
            
            all_predictions.extend(predictions.cpu().numpy())
            all_confidences.extend(confidence.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
        
        return np.array(all_predictions), np.array(all_confidences), np.array(all_targets)