#!/usr/bin/env python3
"""
Genomic Image CNN for Phenotypic Prediction

This script implements a convolutional neural network that treats genomic embeddings as images, where:
- Height = embedding dimensions (960)
- Width = genomic bins × haplotypes (21 × 2 = 42)  
- Channels = loci (4)

The approach treats haplotypes as additional "rows" in a genomic image and uses convolutions 
for joint bin-haplotype processing with natural permutation invariance.

Usage:
    python genomic_image_cnn.py --embeddings path/to/embeddings.pt --labels path/to/labels.pt
"""

import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, average_precision_score
from tqdm import tqdm
import os
import json
from datetime import datetime


def reshape_to_genomic_image(combined_embeddings):
    """
    Reshape embeddings from [samples, loci, bins, embedding_dim, haplotypes] 
    to [samples, loci, embedding_dim, bins, haplotypes] for image-like processing
    
    Args:
        combined_embeddings: torch.Tensor of shape [N, 4, 21, 960, 2]
    
    Returns:
        torch.Tensor of shape [N, 4, 960, 21, 2] - ready for 2D convolution
    """
    # Permute dimensions: [samples, loci, bins, embedding_dim, haplotypes] 
    # -> [samples, loci, embedding_dim, bins, haplotypes]
    return combined_embeddings.permute(0, 1, 3, 2, 4)  # [N, 4, 960, 21, 2]


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
        x_perm1 = x_reshaped  # Original ordering: [batch, loci, embedding_dim, bins, haplotypes]
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
    - Width = genomic bins × haplotypes (21 × 2 = 42)
    - Channels = loci (4)
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
        
        # First conv layer: process genomic bins and haplotypes jointly
        self.conv1 = PermutationInvariantConv2D(
            in_channels=num_loci,
            out_channels=64,
            kernel_size=(3, 3),  # 3x3 kernel across embedding_dim and bins*haplotypes
            padding=1
        )
        
        # Second conv layer: further feature extraction
        self.conv2 = nn.Conv2d(
            in_channels=64,
            out_channels=128,
            kernel_size=(3, 3),
            padding=1
        )
        
        # Third conv layer: high-level features
        self.conv3 = nn.Conv2d(
            in_channels=128,
            out_channels=256,
            kernel_size=(3, 3),
            padding=1
        )
        
        # Global pooling
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        # Locus-specific processing
        self.locus_processor = nn.Sequential(
            nn.Linear(256, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Final locus aggregation and prediction
        self.locus_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim // 2,
            num_heads=8,
            batch_first=True
        )
        
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
        
        # Reshape to image format: [batch, loci, embedding_dim, bins*haplotypes]
        x_img = x.permute(0, 1, 3, 2, 4).contiguous()  # [batch, loci, embedding_dim, bins, haplotypes]
        x_img = x_img.view(batch_size, self.num_loci, self.embedding_dim, 
                          self.num_bins * self.num_haplotypes)
        
        # Apply convolutional layers
        x = F.relu(self.conv1(x_img))  # [batch, 64, embedding_dim, bins*haplotypes]
        x = F.max_pool2d(x, kernel_size=2, stride=2)  # Reduce spatial dimensions
        
        x = F.relu(self.conv2(x))  # [batch, 128, ...]
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        x = F.relu(self.conv3(x))  # [batch, 256, ...]
        x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # Global pooling
        x = self.global_pool(x)  # [batch, 256, 1, 1]
        x = x.view(batch_size, self.num_loci, -1)  # [batch, loci, 256]
        
        # Process each locus
        locus_features = self.locus_processor(x)  # [batch, loci, hidden_dim//2]
        
        # Attention across loci
        attended_features, attention_weights = self.locus_attention(
            locus_features, locus_features, locus_features
        )
        
        # Aggregate across loci (mean pooling)
        sample_features = torch.mean(attended_features, dim=1)  # [batch, hidden_dim//2]
        
        # Final prediction
        prediction = self.classifier(sample_features)
        
        return prediction, attention_weights


class MultiScaleGenomicImageCNN(nn.Module):
    """
    Multi-scale CNN that processes genomic images at different resolutions
    """
    def __init__(self, embedding_dim=960, num_loci=4, num_bins=21, num_haplotypes=2):
        super().__init__()
        
        self.embedding_dim = embedding_dim
        self.num_loci = num_loci
        self.num_bins = num_bins
        self.num_haplotypes = num_haplotypes
        
        # Different kernel sizes to capture different genomic scales
        self.conv_small = PermutationInvariantConv2D(num_loci, 64, kernel_size=(3, 3))
        self.conv_medium = PermutationInvariantConv2D(num_loci, 64, kernel_size=(5, 5))
        self.conv_large = PermutationInvariantConv2D(num_loci, 64, kernel_size=(7, 7))
        
        # Additional processing layers
        self.conv_fusion = nn.Conv2d(192, 256, kernel_size=(3, 3), padding=1)
        
        # Global pooling and classification
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1)
        )
    
    def forward(self, x):
        batch_size = x.shape[0]
        
        # Reshape to image format
        x_img = x.permute(0, 1, 3, 2, 4).contiguous()
        x_img = x_img.view(batch_size, self.num_loci, self.embedding_dim, 
                          self.num_bins * self.num_haplotypes)
        
        # Multi-scale convolutions
        small_features = F.relu(self.conv_small(x_img))
        medium_features = F.relu(self.conv_medium(x_img))
        large_features = F.relu(self.conv_large(x_img))
        
        # Concatenate multi-scale features
        multi_scale = torch.cat([small_features, medium_features, large_features], dim=1)
        
        # Further processing
        fused_features = F.relu(self.conv_fusion(multi_scale))
        fused_features = F.max_pool2d(fused_features, kernel_size=2, stride=2)
        
        # Global pooling and classification
        global_features = self.global_pool(fused_features)
        global_features = global_features.view(batch_size, -1)
        
        prediction = self.classifier(global_features)
        
        return prediction


def train_genomic_image_cnn(model, embeddings, labels, device='cuda', epochs=100, batch_size=32):
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
        
        val_loss /= len(val_loader)
        val_accuracy = val_correct / val_total
        
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)
        
        scheduler.step(val_loss)
        
        # Print progress
        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}")
    
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
        print(f"Training curves saved to {save_path}")
    
    plt.show()


def plot_roc_curve(model, embeddings, labels, device='cuda', save_path=None):
    """Plot ROC curve for model predictions"""
    model.eval()
    embeddings = embeddings.to(device)
    labels = labels.to(device)
    
    # Reshape embeddings for image processing
    embeddings_reshaped = reshape_to_genomic_image(embeddings)
    
    with torch.no_grad():
        # Handle different model types
        if hasattr(model, 'locus_attention'):  # GenomicImageCNN
            predictions, attention_weights = model(embeddings_reshaped)
        else:  # MultiScaleGenomicImageCNN
            predictions = model(embeddings_reshaped)
        
        probs = torch.sigmoid(predictions).cpu().numpy().reshape(-1)
        labels_np = labels.cpu().numpy().reshape(-1)
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels_np, probs)
    roc_auc = roc_auc_score(labels_np, probs)
    
    # Plot ROC curve
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='darkred', lw=2, label=f'ROC Curve (AUC={roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"ROC curve saved to {save_path}")
    
    plt.show()
    
    return roc_auc


def analyze_attention_weights(model, embeddings, device='cuda', save_path=None):
    """Analyze attention weights to see which loci are most important"""
    if not hasattr(model, 'locus_attention'):
        print("Model does not have attention weights to analyze")
        return None
        
    model.eval()
    embeddings = embeddings.to(device)
    embeddings_reshaped = reshape_to_genomic_image(embeddings)
    
    with torch.no_grad():
        predictions, attention_weights = model(embeddings_reshaped)
    
    # Average attention weights across samples
    avg_attention = attention_weights.mean(dim=0).cpu().numpy()  # [num_loci, num_loci]
    
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
        print(f"Attention heatmap saved to {save_path}")
    
    plt.show()
    
    # Calculate locus importance (sum of attention weights for each locus)
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
        print(f"Locus importance plot saved to {importance_path}")
    
    plt.show()
    
    return locus_importance


def load_data(embeddings_path, labels_path):
    """Load embeddings and labels from files"""
    print(f"Loading embeddings from {embeddings_path}")
    embeddings = torch.load(embeddings_path)
    
    print(f"Loading labels from {labels_path}")
    labels = torch.load(labels_path)
    
    print(f"Embeddings shape: {embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {torch.bincount(labels)}")
    
    return embeddings, labels


def create_synthetic_data(n_samples=1575, n_loci=4, n_bins=21, embedding_dim=960, n_haplotypes=2):
    """Create synthetic data for testing"""
    print("Creating synthetic data for demonstration...")
    combined_embeddings = torch.randn(n_samples, n_loci, n_bins, embedding_dim, n_haplotypes)
    labels = torch.randint(0, 2, (n_samples,))
    
    print(f"Data shape: {combined_embeddings.shape}")
    print(f"Labels shape: {labels.shape}")
    print(f"Label distribution: {torch.bincount(labels)}")
    
    return combined_embeddings, labels


def main():
    parser = argparse.ArgumentParser(description='Train Genomic Image CNN for phenotypic prediction')
    parser.add_argument('--embeddings', type=str, help='Path to embeddings tensor file (.pt)')
    parser.add_argument('--labels', type=str, help='Path to labels tensor file (.pt)')
    parser.add_argument('--model_type', type=str, choices=['cnn', 'multiscale'], default='cnn',
                       help='Type of model to train')
    parser.add_argument('--epochs', type=int, default=50, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
    parser.add_argument('--device', type=str, default='auto', help='Device to use (cuda/cpu/auto)')
    parser.add_argument('--output_dir', type=str, default='./results', help='Output directory for results')
    parser.add_argument('--synthetic', action='store_true', help='Use synthetic data for testing')
    parser.add_argument('--save_model', action='store_true', help='Save trained model')
    parser.add_argument('--analyze_attention', action='store_true', help='Analyze attention weights')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Load data
    if args.synthetic:
        embeddings, labels = create_synthetic_data()
    else:
        if not args.embeddings or not args.labels:
            raise ValueError("Must provide --embeddings and --labels paths when not using --synthetic")
        embeddings, labels = load_data(args.embeddings, args.labels)
    
    # Create model
    if args.model_type == 'cnn':
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
    
    print(f"Created {model_name} with {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train the model
    print(f"Training {model_name}...")
    trained_model, train_losses, val_losses, val_accuracies = train_genomic_image_cnn(
        model, embeddings, labels, device=device, epochs=args.epochs, batch_size=args.batch_size
    )
    
    # Plot training curves
    curves_path = os.path.join(args.output_dir, f"{model_name}_{timestamp}_training_curves.png")
    plot_training_curves(train_losses, val_losses, val_accuracies, save_path=curves_path)
    
    # Plot ROC curve
    roc_path = os.path.join(args.output_dir, f"{model_name}_{timestamp}_roc_curve.png")
    roc_auc = plot_roc_curve(trained_model, embeddings, labels, device=device, save_path=roc_path)
    print(f"Final ROC AUC: {roc_auc:.3f}")
    
    # Analyze attention weights if requested and model supports it
    if args.analyze_attention and hasattr(trained_model, 'locus_attention'):
        attention_path = os.path.join(args.output_dir, f"{model_name}_{timestamp}_attention.png")
        locus_importance = analyze_attention_weights(trained_model, embeddings, device=device, save_path=attention_path)
        print(f"Locus importance scores: {locus_importance}")
    
    # Save model if requested
    if args.save_model:
        model_path = os.path.join(args.output_dir, f"{model_name}_{timestamp}.pt")
        torch.save(trained_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")
    
    # Save training results
    results = {
        'model_type': model_name,
        'final_train_loss': train_losses[-1],
        'final_val_loss': val_losses[-1],
        'final_val_accuracy': val_accuracies[-1],
        'final_roc_auc': roc_auc,
        'epochs': args.epochs,
        'batch_size': args.batch_size,
        'device': str(device),
        'timestamp': timestamp
    }
    
    results_path = os.path.join(args.output_dir, f"{model_name}_{timestamp}_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {results_path}")
    print("Training completed successfully!")


if __name__ == "__main__":
    main()
