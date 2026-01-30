
# ============================================================================
# 4. DATA LOADING AND AUGMENTATION
# ============================================================================

class ConcreteCrackDataset(Dataset):
    """Custom dataset for concrete crack images"""
    
    def __init__(self, root_dir: str, transform=None, split: str = 'train'):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.split = split
        
        # Expected structure: root_dir/{class}/*.jpg
        self.classes = ['Positive', 'Negative']  # Crack, No Crack
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        
        # Collect all image paths
        self.images = []
        self.labels = []
        
        for class_name in self.classes:
            class_dir = self.root_dir / class_name
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.images.append(img_path)
                    self.labels.append(self.class_to_idx[class_name])
        
        print(f"Found {len(self.images)} images in {root_dir}")
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        img_path = self.images[idx]
        label = self.labels[idx]
        
        # Load image
        image = plt.imread(img_path)
        
        # Convert to RGB if grayscale
        if len(image.shape) == 2:
            image = np.stack([image] * 3, axis=-1)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_loaders(data_dir: str, batch_size: int = 32, num_workers: int = 4):
    """Create train, validation, and test dataloaders"""
    
    # Data augmentation for training
    train_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(224),  # MobileNetV3 expects 224x224
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Minimal transformation for validation/test
    val_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = ConcreteCrackDataset(data_dir, transform=val_transform)
    
    # Split into train/val/test (70/15/15)
    total_size = len(full_dataset)
    train_size = int(0.7 * total_size)
    val_size = int(0.15 * total_size)
    test_size = total_size - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(42)
    )
    
    # Apply training transform to train dataset
    train_dataset.dataset.transform = train_transform
    
    # Create dataloaders with GPU optimization
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True if num_workers > 0 else False
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader