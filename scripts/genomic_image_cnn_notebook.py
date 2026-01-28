"""
Genomic Image CNN for Phenotypic Prediction - Notebook Version
==============================================================

This notebook-friendly version implements a convolutional neural network that treats genomic embeddings as images, where:
- Height = embedding dimensions (960)
- Width = genomic bins (21)
- Channels = haplotypes (2)

The approach processes each locus separately through CNNs to extract relevant features,
then uses attention to learn relationships between loci for final prediction.

Usage: Copy this entire cell into your notebook and modify the parameters below as needed.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from tqdm.auto import tqdm
import os
import json
from datetime import datetime

print("📦 Genomic Image CNN notebook version loaded!")

# =============================================================================
# CORE FUNCTIONS AND CLASSES
# =============================================================================

def reshape_to_genomic_image(combined_embeddings):
    """
    Reshape embeddings from [samples, loci, bins, embedding_dim, haplotypes] 
    to the format expected by the CNN.
    
    Args:
        combined_embeddings: torch.Tensor of shape [N, 4, 21, 960, 2]
    
    Returns:
        torch.Tensor of shape [N, 4, 21, 960, 2] - ready for processing by GenomicImageCNN
    """
    # For the new architecture, we keep the original shape
    # The CNN will handle reshaping internally
    return combined_embeddings  # [N, 4, 21, 960, 2]

class PermutationInvariantConv2D(nn.Module):
    """
    Convolutional layer that treats haplotypes as image rows with permutation invariance.
    Uses max pooling across haplotype permutations to ensure order invariance.
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        
    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape [batch, loci, embedding_dim, bins*haplotypes]
        """
        batch_size, loci, embedding_dim, bins_haplotypes = x.shape
        
        # Split back into bins and haplotypes
        bins = bins_haplotypes // 2  # Assuming 2 haplotypes
        haplotypes = 2
        
        # Reshape to separate bins and haplotypes
        x_reshaped = x.view(batch_size, loci, embedding_dim, bins, haplotypes)
        
        # Create both possible haplotype orderings
        x_perm1 = x_reshaped  # Original ordering
        x_perm2 = torch.flip(x_reshaped, dims=[-1])  # Flipped ordering
        
        # Reshape both to image format for convolution
        x1_img = x_perm1.view(batch_size, loci, embedding_dim, bins * haplotypes)
        x2_img = x_perm2.view(batch_size, loci, embedding_dim, bins * haplotypes)
        
        # Apply convolution to both permutations
        conv1 = self.conv(x1_img)
        conv2 = self.conv(x2_img)
        
        # Take maximum to ensure permutation invariance
        conv_out = torch.maximum(conv1, conv2)
        
        return conv_out

class GenomicImageCNN(nn.Module):
    """
    CNN that treats genomic data as images where:
    - Height = embedding dimensions (960)
    - Width = genomic bins (21)
    - Channels = haplotypes (2)
    
    Each locus is processed separately through CNN layers to extract features,
    then attention learns relationships between loci.
    """
    def __init__(self, 
                 embedding_dim=960, 
                 num_loci=4, 
                 num_bins=21, 
                 num_haplotypes=2,
                 hidden_dim=512,
                 dropout=0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_loci = num_loci
        self.num_bins = num_bins
        self.num_haplotypes = num_haplotypes
        
        # CNN to process each locus independently
        # Input: [batch, haplotypes, embedding_dim, bins] = [batch, 2, 960, 21]
        self.conv1 = nn.Conv2d(
            in_channels=num_haplotypes,  # 2 channels (haplotypes)
            out_channels=64,
            kernel_size=(3, 3),  # (embedding_dim, bins)
            padding=(1, 1)
        )
        
        # Second conv layer
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        
        # Third conv layer
        self.conv3 = nn.Conv2d(128, 256, kernel_size=(3, 3), padding=(1, 1))
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection to feature space for attention
        self.locus_projection = nn.Sequential(
            nn.Linear(256, hidden_dim),  # Changed from 512 to 256
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Attention to learn relationships between loci
        self.locus_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=8,
            batch_first=True
        )
        
        # Final classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape [batch, loci, bins, embedding_dim, haplotypes]
        """
        batch_size = x.shape[0]
        
        # Process each locus separately
        locus_features = []
        
        for locus_idx in range(self.num_loci):
            # Extract one locus: [batch, bins, embedding_dim, haplotypes]
            locus_data = x[:, locus_idx, :, :, :]  # [batch, 21, 960, 2]
            
            # Reshape to image format: [batch, haplotypes, embedding_dim, bins]
            # This treats: channels=haplotypes (2), height=embedding_dim (960), width=bins (21)
            locus_img = locus_data.permute(0, 3, 2, 1)  # [batch, 2, 960, 21]
            
            # Apply CNN layers
            # Input: [batch, 2, 960, 21]
            x_conv = F.relu(self.conv1(locus_img))  # [batch, 64, 960, 21]
            x_conv = F.max_pool2d(x_conv, kernel_size=2, stride=2)  # [batch, 64, 480, 10]
            
            x_conv = F.relu(self.conv2(x_conv))  # [batch, 128, 480, 10]
            x_conv = F.max_pool2d(x_conv, kernel_size=2, stride=2)  # [batch, 128, 240, 5]
            
            x_conv = F.relu(self.conv3(x_conv))  # [batch, 256, 240, 5]
            x_conv = F.max_pool2d(x_conv, kernel_size=2, stride=2)  # [batch, 256, 120, 2]
            
            # Global pooling
            x_conv = self.global_pool(x_conv)  # [batch, 256, 1, 1]
            x_conv = x_conv.view(batch_size, -1)  # [batch, 256]
            
            locus_features.append(x_conv)
        
        # Stack locus features: [batch, loci, 256]
        locus_features = torch.stack(locus_features, dim=1)
        
        # Project to attention space
        processed_features = self.locus_projection(locus_features)  # [batch, loci, hidden_dim//2]
        
        # Attention across loci to learn relationships
        attended_features, attention_weights = self.locus_attention(
            processed_features, processed_features, processed_features
        )
        
        # Aggregate across loci (mean pooling)
        sample_features = torch.mean(attended_features, dim=1)  # [batch, hidden_dim//2]
        
        # Final prediction
        prediction = self.classifier(sample_features)
        
        return prediction, attention_weights

class MultiScaleGenomicImageCNN(nn.Module):
    """
    Multi-scale CNN that processes genomic images at different resolutions.
    Each locus is processed separately with different kernel sizes to capture
    different scales of genomic features.
    """
    def __init__(self, embedding_dim=960, num_loci=4, num_bins=21, num_haplotypes=2, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_loci = num_loci
        self.num_bins = num_bins
        self.num_haplotypes = num_haplotypes
        
        # Multi-scale convolutions with different kernel sizes
        # Small kernel - local features
        self.conv1_small = nn.Conv2d(num_haplotypes, 64, kernel_size=(3, 3), padding=(1, 1))
        self.conv2_small = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        
        # Medium kernel - mid-range features
        self.conv1_medium = nn.Conv2d(num_haplotypes, 64, kernel_size=(5, 5), padding=(2, 2))
        self.conv2_medium = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        
        # Large kernel - global features
        self.conv1_large = nn.Conv2d(num_haplotypes, 64, kernel_size=(7, 7), padding=(3, 3))
        self.conv2_large = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=(1, 1))
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Projection to feature space
        self.locus_projection = nn.Sequential(
            nn.Linear(192, hidden_dim),  # 64*3 for three scales
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Locus aggregation (without attention for multi-scale)
        self.locus_aggregation = nn.Sequential(
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 4, 1)
        )
    
    def forward(self, x):
        """
        Args:
            x: torch.Tensor of shape [batch, loci, bins, embedding_dim, haplotypes]
        """
        batch_size = x.shape[0]
        
        # Process each locus separately
        all_locus_features = []
        
        for locus_idx in range(self.num_loci):
            # Extract one locus: [batch, bins, embedding_dim, haplotypes]
            locus_data = x[:, locus_idx, :, :, :]  # [batch, 21, 960, 2]
            
            # Reshape to image format: [batch, haplotypes, embedding_dim, bins]
            locus_img = locus_data.permute(0, 3, 2, 1)  # [batch, 2, 960, 21]
            
            # Multi-scale processing
            # Small scale
            small = F.relu(self.conv1_small(locus_img))  # [batch, 64, 960, 21]
            small = F.relu(self.conv2_small(small))  # [batch, 64, 960, 21]
            small = F.max_pool2d(small, kernel_size=2, stride=2)  # [batch, 64, 480, 10]
            small = self.global_pool(small)  # [batch, 64, 1, 1]
            
            # Medium scale
            medium = F.relu(self.conv1_medium(locus_img))  # [batch, 64, 960, 21]
            medium = F.relu(self.conv2_medium(medium))  # [batch, 64, 960, 21]
            medium = F.max_pool2d(medium, kernel_size=2, stride=2)  # [batch, 64, 480, 10]
            medium = self.global_pool(medium)  # [batch, 64, 1, 1]
            
            # Large scale
            large = F.relu(self.conv1_large(locus_img))  # [batch, 64, 960, 21]
            large = F.relu(self.conv2_large(large))  # [batch, 64, 960, 21]
            large = F.max_pool2d(large, kernel_size=2, stride=2)  # [batch, 64, 480, 10]
            large = self.global_pool(large)  # [batch, 64, 1, 1]
            
            # Concatenate multi-scale features
            multi_scale = torch.cat([small, medium, large], dim=1)  # [batch, 192, 1, 1]
            multi_scale = multi_scale.view(batch_size, -1)  # [batch, 192]
            
            all_locus_features.append(multi_scale)
        
        # Stack locus features: [batch, loci, 192]
        locus_features = torch.stack(all_locus_features, dim=1)
        
        # Process each locus through projection
        processed_features = self.locus_projection(locus_features)  # [batch, loci, hidden_dim//2]
        
        # Aggregate across loci (mean pooling)
        sample_features = torch.mean(processed_features, dim=1)  # [batch, hidden_dim//2]
        
        # Final prediction
        prediction = self.locus_aggregation(sample_features)
        
        return prediction

# =============================================================================
# TRAINING AND EVALUATION FUNCTIONS
# =============================================================================

def train_genomic_image_cnn(model, embeddings, labels, device='cuda', epochs=100, batch_size=32, memory_efficient=True):
    """
    Train the genomic image CNN
    """
    model = model.to(device)
    embeddings = embeddings.to(device)
    labels = labels.to(device)
    
    # Reshape embeddings for image processing
    embeddings_reshaped = reshape_to_genomic_image(embeddings)
    
    # Split data
    train_size = int(0.8 * len(embeddings_reshaped))
    train_embeddings = embeddings_reshaped[:train_size]
    train_labels = labels[:train_size]
    val_embeddings = embeddings_reshaped[train_size:]
    val_labels = labels[train_size:]
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(train_embeddings, train_labels)
    val_dataset = torch.utils.data.TensorDataset(val_embeddings, val_labels)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Training setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10)
    
    # Training loop
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    for epoch in tqdm(range(epochs), desc="Training"):
        # Training
        model.train()
        train_loss = 0
        for batch_embeddings, batch_labels in train_loader:
            optimizer.zero_grad()
            
            # Handle different model types
            if hasattr(model, 'locus_attention'):  # GenomicImageCNN
                predictions, attention_weights = model(batch_embeddings)
            else:  # MultiScaleGenomicImageCNN
                predictions = model(batch_embeddings)
            
            loss = criterion(predictions.squeeze(), batch_labels.float())
            
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
            # Memory management
            if memory_efficient and device.type == 'cuda':
                torch.cuda.empty_cache()
        
        train_loss /= len(train_loader)
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch_embeddings, batch_labels in val_loader:
                # Handle different model types
                if hasattr(model, 'locus_attention'):  # GenomicImageCNN
                    predictions, attention_weights = model(batch_embeddings)
                else:  # MultiScaleGenomicImageCNN
                    predictions = model(batch_embeddings)
                
                loss = criterion(predictions.squeeze(), batch_labels.float())
                val_loss += loss.item()
                
                # Calculate accuracy
                pred_binary = (torch.sigmoid(predictions) > 0.5).float()
                val_correct += (pred_binary.squeeze() == batch_labels).sum().item()
                val_total += batch_labels.size(0)
                
                # Memory management
                if memory_efficient and device.type == 'cuda':
                    torch.cuda.empty_cache()
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(val_loss)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
            if device.type == 'cuda':
                gpu_memory = torch.cuda.memory_allocated() / 1024**3
                print(f"  GPU Memory Used: {gpu_memory:.2f}GB")
    
    return model, train_losses, val_losses, val_accuracies

def plot_training_curves(train_losses, val_losses, val_accuracies, save_path=None):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss', color='blue')
    ax1.plot(val_losses, label='Val Loss', color='red')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curve
    ax2.plot(val_accuracies, label='Val Accuracy', color='green')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Training curves saved to {save_path}")
    
    plt.show()

def plot_roc_curve(model, embeddings, labels, device='cuda', save_path=None, test_split=0.2, verbose=False):
    """
    Plot ROC curve for model predictions on VALIDATION/TEST data only.
    
    The embeddings are split into train/test using the same ratio as during training,
    and only the test set predictions are used for the ROC curve.
    """
    model.eval()
    
    # Reshape embeddings for image processing
    embeddings_reshaped = reshape_to_genomic_image(embeddings)
    
    # Split data the same way as during training (80/20 split)
    train_size = int((1 - test_split) * len(embeddings_reshaped))
    
    # Extract test data
    test_embeddings = embeddings_reshaped[train_size:]
    test_labels = labels[train_size:]
    
    if verbose:
        print(f"📊 Evaluating on test set: {len(test_embeddings)} samples")
    
    with torch.no_grad():
        # Process in small batches to avoid memory issues
        all_predictions = []
        test_embeddings_cpu = test_embeddings.cpu()
        
        for i in range(0, len(test_embeddings), 4):  # Small batch size
            batch_embeddings = test_embeddings_cpu[i:i+4].to(device)
            
            # Handle different model types
            if hasattr(model, 'locus_attention'):  # GenomicImageCNN
                predictions, attention_weights = model(batch_embeddings)
            else:  # MultiScaleGenomicImageCNN
                predictions = model(batch_embeddings)
            
            all_predictions.append(predictions.cpu())
            
            # Clear memory
            del batch_embeddings
            if device.type == 'cuda':
                torch.cuda.empty_cache()
        
        # Concatenate all predictions
        predictions = torch.cat(all_predictions, dim=0)
        probs = torch.sigmoid(predictions).cpu().numpy().reshape(-1)
        labels_np = test_labels.cpu().numpy().reshape(-1)
        
        if verbose:
            print(f"📊 Predictions shape: {predictions.shape}")
            print(f"📊 Probability range: [{probs.min():.3f}, {probs.max():.3f}]")
            print(f"📊 Label distribution: {np.bincount(labels_np.astype(int))}")
    
    # Check for invalid predictions
    if np.isnan(probs).any() or (probs.min() == probs.max()):
        print(f"⚠️  Warning: All predictions are the same or contain NaN!")
        print(f"    Min prob: {probs.min():.6f}, Max prob: {probs.max():.6f}")
        return 0.5  # Return random performance
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels_np, probs)
    roc_auc = roc_auc_score(labels_np, probs)
    
    # Plot ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkred', lw=2, label=f'Test ROC (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Random Classifier')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Test Set')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Test ROC curve saved to {save_path}")
    
    plt.show()
    
    return roc_auc

def analyze_attention_weights(model, embeddings, device='cuda', save_path=None):
    """Analyze attention weights to see which loci are most important"""
    if not hasattr(model, 'locus_attention'):
        print("❌ Model does not have attention weights to analyze")
        return None
        
    model.eval()
    embeddings = embeddings.to(device)
    embeddings_reshaped = reshape_to_genomic_image(embeddings)
    
    with torch.no_grad():
        predictions, attention_weights = model(embeddings_reshaped)
    
    # Average attention weights across samples
    avg_attention = attention_weights.mean(dim=0).cpu().numpy()
    
    # Plot attention heatmap
    plt.figure(figsize=(8, 6))
    plt.imshow(avg_attention, cmap='Blues', aspect='auto')
    plt.colorbar(label='Attention Weight')
    plt.xlabel('Key Locus')
    plt.ylabel('Query Locus')
    plt.title('Attention Weights Between Loci')
    
    # Add locus labels
    locus_names = [f'Locus {i}' for i in range(avg_attention.shape[0])]
    plt.xticks(range(len(locus_names)), locus_names, rotation=45)
    plt.yticks(range(len(locus_names)), locus_names)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Attention heatmap saved to {save_path}")
    
    plt.show()
    
    # Calculate locus importance
    locus_importance = avg_attention.sum(axis=1)
    
    # Plot locus importance
    plt.figure(figsize=(10, 6))
    bars = plt.bar(range(len(locus_importance)), locus_importance)
    plt.xlabel('Locus')
    plt.ylabel('Attention Importance')
    plt.title('Locus Importance (Sum of Attention Weights)')
    plt.xticks(range(len(locus_names)), locus_names, rotation=45)
    
    # Color bars by importance
    colors = plt.cm.viridis(locus_importance / locus_importance.max())
    for bar, color in zip(bars, colors):
        bar.set_color(color)
    
    plt.tight_layout()
    
    if save_path:
        importance_path = save_path.replace('.png', '_importance.png')
        plt.savefig(importance_path, dpi=300, bbox_inches='tight')
        print(f"📊 Locus importance plot saved to {importance_path}")
    
    plt.show()
    
    return locus_importance

# =============================================================================
# DATA LOADING AND SYNTHETIC DATA FUNCTIONS
# =============================================================================

def load_data(embeddings_path, labels_path):
    """Load embeddings and labels from files"""
    print(f"📂 Loading embeddings from {embeddings_path}")
    embeddings = torch.load(embeddings_path)
    
    print(f"📂 Loading labels from {labels_path}")
    labels = torch.load(labels_path)
    
    print(f"📊 Embeddings shape: {embeddings.shape}")
    print(f"📊 Labels shape: {labels.shape}")
    print(f"📊 Label distribution: {torch.bincount(labels)}")
    
    return embeddings, labels

def create_synthetic_data(n_samples=1575, n_loci=4, n_bins=21, embedding_dim=960, n_haplotypes=2):
    """Create synthetic data for testing"""
    print("🔬 Creating synthetic data for demonstration...")
    combined_embeddings = torch.randn(n_samples, n_loci, n_bins, embedding_dim, n_haplotypes)
    labels = torch.randint(0, 2, (n_samples,))
    
    print(f"📊 Data shape: {combined_embeddings.shape}")
    print(f"📊 Labels shape: {labels.shape}")
    print(f"📊 Label distribution: {torch.bincount(labels)}")
    
    return combined_embeddings, labels

# =============================================================================
# MAIN TRAINING FUNCTION - MODIFY PARAMETERS HERE
# =============================================================================

def run_genomic_image_cnn_training(
    embeddings_path=None,
    labels_path=None,
    model_type='cnn',  # 'cnn' or 'multiscale'
    epochs=50,
    batch_size=32,
    device='auto',  # 'cuda', 'cpu', or 'auto'
    output_dir='./results',
    use_synthetic=True,  # Set to False to use real data
    save_model=True,
    analyze_attention=True,
    synthetic_samples=1575,
    memory_efficient=True  # Enable memory-efficient training
):
    """
    Main function to run genomic image CNN training.
    
    Parameters:
    -----------
    embeddings_path : str, optional
        Path to embeddings tensor file (.pt) - required if use_synthetic=False
    labels_path : str, optional  
        Path to labels tensor file (.pt) - required if use_synthetic=False
    model_type : str, default 'cnn'
        Type of model to train ('cnn' or 'multiscale')
    epochs : int, default 50
        Number of training epochs
    batch_size : int, default 32
        Batch size for training
    device : str, default 'auto'
        Device to use ('cuda', 'cpu', or 'auto')
    output_dir : str, default './results'
        Output directory for results
    use_synthetic : bool, default True
        Use synthetic data for testing
    save_model : bool, default True
        Save trained model
    analyze_attention : bool, default True
        Analyze attention weights (only for CNN model)
    synthetic_samples : int, default 1575
        Number of synthetic samples to generate
    """
    
    # Set device
    if device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"🖥️  Using device: {device}")
    
    # Memory management for CUDA
    if device.type == 'cuda':
        if memory_efficient:
            # Clear cache and set memory-efficient settings
            torch.cuda.empty_cache()
            torch.backends.cudnn.benchmark = False  # Disable for memory efficiency
            torch.backends.cudnn.deterministic = True
            print("🧠 Memory-efficient CUDA settings enabled")
        
        # Print GPU memory info
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        gpu_free = (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1024**3
        print(f"📊 GPU Memory: {gpu_free:.2f}GB free / {gpu_memory:.2f}GB total")
        
        # Warn if batch size might be too large
        estimated_memory_per_sample = 0.1  # Rough estimate in GB
        estimated_batch_memory = batch_size * estimated_memory_per_sample
        if estimated_batch_memory > gpu_free * 0.8:
            print(f"⚠️  Warning: Batch size {batch_size} might be too large for available memory")
            print(f"💡 Consider reducing batch_size to {int(gpu_free * 0.8 / estimated_memory_per_sample)} or less")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    if use_synthetic:
        embeddings, labels = create_synthetic_data(n_samples=synthetic_samples)
    else:
        if not embeddings_path or not labels_path:
            raise ValueError("Must provide embeddings_path and labels_path when use_synthetic=False")
        embeddings, labels = load_data(embeddings_path, labels_path)
    
    # Create model
    if model_type == 'cnn':
        model = GenomicImageCNN(
            embedding_dim=960,
            num_loci=4,
            num_bins=21,
            num_haplotypes=2
        )
        model_name = "GenomicImageCNN"
    else:
        model = MultiScaleGenomicImageCNN(
            embedding_dim=960,
            num_loci=4,
            num_bins=21,
            num_haplotypes=2
        )
        model_name = "MultiScaleGenomicImageCNN"
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"🏗️  Created {model_name} with {total_params:,} parameters")
    
    # Train the model
    print(f"🚀 Training {model_name}...")
    trained_model, train_losses, val_losses, val_accuracies = train_genomic_image_cnn(
        model, embeddings, labels, device=device, epochs=epochs, batch_size=batch_size, memory_efficient=memory_efficient
    )
    
    # Plot training curves
    curves_path = os.path.join(output_dir, f"{model_name}_{timestamp}_training_curves.png")
    plot_training_curves(train_losses, val_losses, val_accuracies, save_path=curves_path)
    
    # Plot ROC curve on TEST set
    roc_path = os.path.join(output_dir, f"{model_name}_{timestamp}_roc_curve.png")
    roc_auc = plot_roc_curve(trained_model, embeddings, labels, device=device, 
                            save_path=roc_path, verbose=True)  # Evaluate on test set only
    print(f"📈 Test ROC AUC: {roc_auc:.3f}")
    
    # Analyze attention weights if requested and model supports it
    locus_importance = None
    if analyze_attention and hasattr(trained_model, 'locus_attention'):
        attention_path = os.path.join(output_dir, f"{model_name}_{timestamp}_attention.png")
        locus_importance = analyze_attention_weights(trained_model, embeddings, device=device, save_path=attention_path)
        print(f"🎯 Locus importance scores: {locus_importance}")
    
    # Save model if requested
    if save_model:
        model_path = os.path.join(output_dir, f"{model_name}_{timestamp}.pt")
        torch.save(trained_model.state_dict(), model_path)
        print(f"💾 Model saved to {model_path}")
    
    # Save training results
    results = {
        'model_type': model_name,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_val_accuracy': val_accuracies[-1],
        'test_roc_auc': roc_auc,  # Renamed to clarify it's on test set
        'epochs': epochs,
        'batch_size': batch_size,
        'device': str(device),
        'timestamp': timestamp,
        'total_parameters': total_params
    }
    
    if locus_importance is not None:
        results['locus_importance'] = locus_importance.tolist()
    
    results_path = os.path.join(output_dir, f"{model_name}_{timestamp}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"📄 Results saved to {results_path}")
    print("✅ Training completed successfully!")
    
    return trained_model, results

# =============================================================================
# EXAMPLE USAGE - MODIFY THESE PARAMETERS AS NEEDED
# =============================================================================

# Example 1: Quick test with synthetic data (memory-efficient)
# model, results = run_genomic_image_cnn_training(
#     model_type='cnn',
#     epochs=20,
#     batch_size=4,  # Reduced batch size for memory efficiency
#     use_synthetic=True,
#     synthetic_samples=200  # Reduced sample size
# )

# Example 2: Train with real data (memory-efficient)
# model, results = run_genomic_image_cnn_training(
#     embeddings_path="path/to/your/embeddings.pt",
#     labels_path="path/to/your/labels.pt",
#     model_type='cnn',
#     epochs=100,
#     batch_size=8,  # Smaller batch size for memory efficiency
#     use_synthetic=False,
#     analyze_attention=True,
#     memory_efficient=True
# )

# Example 3: Multi-scale model (memory-efficient)
# model, results = run_genomic_image_cnn_training(
#     model_type='multiscale',
#     epochs=50,
#     batch_size=4,  # Even smaller batch size for multi-scale
#     use_synthetic=True,
#     analyze_attention=False,  # Multi-scale model doesn't have attention
#     memory_efficient=True
# )

# Example 4: CPU training (if GPU memory is insufficient)
# model, results = run_genomic_image_cnn_training(
#     model_type='cnn',
#     epochs=20,
#     batch_size=16,
#     device='cpu',  # Force CPU training
#     use_synthetic=True,
#     synthetic_samples=200
# )

print("🎯 Ready to train! Modify the example usage section above and run the cell to start training.")
print("💡 Tip: Start with synthetic data to test the setup, then switch to your real data.")
