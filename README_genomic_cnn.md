# Genomic Image CNN for Phenotypic Prediction

This repository contains a complete implementation of a convolutional neural network that treats genomic embeddings as images for phenotypic prediction.

## Overview

The approach treats genomic embeddings as 2D images where:
- **Height**: 960 embedding dimensions
- **Width**: 21 genomic bins × 2 haplotypes = 42 pixels  
- **Channels**: 4 loci

Key features:
- **Haplotype Permutation Invariance**: Automatically handles both haplotype orderings
- **Joint Learning**: Learns patterns across both genomic bins and haplotypes simultaneously
- **Multi-Scale Features**: Captures both local and global genomic patterns
- **Attention Mechanisms**: Provides interpretability by showing which loci are most important

## Files

- `genomic_image_cnn.py`: Main script with complete implementation
- `example_usage.py`: Example script showing how to use the main script
- `README.md`: This documentation file

## Requirements

```bash
pip install torch torchvision numpy matplotlib scikit-learn tqdm
```

## Quick Start

### 1. Test with Synthetic Data

```bash
python genomic_image_cnn.py --synthetic --epochs 10 --analyze_attention
```

### 2. Train with Your Own Data

```bash
python genomic_image_cnn.py \
    --embeddings path/to/embeddings.pt \
    --labels path/to/labels.pt \
    --epochs 50 \
    --batch_size 32 \
    --output_dir ./results \
    --analyze_attention \
    --save_model
```

### 3. Run Examples

```bash
python example_usage.py
```

## Data Format

Your data should be in PyTorch tensor format:

- **Embeddings**: `torch.Tensor` of shape `[N, 4, 21, 960, 2]`
  - N: number of samples
  - 4: number of loci
  - 21: number of genomic bins
  - 960: embedding dimension
  - 2: number of haplotypes

- **Labels**: `torch.Tensor` of shape `[N]` with binary labels (0 or 1)

## Command Line Options

```
usage: genomic_image_cnn.py [-h] [--embeddings EMBEDDINGS] [--labels LABELS]
                           [--model_type {cnn,multiscale}] [--epochs EPOCHS]
                           [--batch_size BATCH_SIZE] [--device DEVICE]
                           [--output_dir OUTPUT_DIR] [--synthetic]
                           [--save_model] [--analyze_attention]

Train Genomic Image CNN for phenotypic prediction

optional arguments:
  -h, --help            show this help message and exit
  --embeddings EMBEDDINGS
                        Path to embeddings tensor file (.pt)
  --labels LABELS       Path to labels tensor file (.pt)
  --model_type {cnn,multiscale}
                        Type of model to train (default: cnn)
  --epochs EPOCHS       Number of training epochs (default: 50)
  --batch_size BATCH_SIZE
                        Batch size for training (default: 32)
  --device DEVICE       Device to use (cuda/cpu/auto) (default: auto)
  --output_dir OUTPUT_DIR
                        Output directory for results (default: ./results)
  --synthetic           Use synthetic data for testing
  --save_model          Save trained model
  --analyze_attention   Analyze attention weights
```

## Model Architectures

### 1. GenomicImageCNN
- Uses attention mechanisms across loci
- Returns attention weights for interpretability
- Good for understanding which genomic regions are most important

### 2. MultiScaleGenomicImageCNN
- Uses different kernel sizes (3x3, 5x5, 7x7) to capture multi-scale patterns
- Simpler architecture without attention
- Good for capturing patterns at different genomic scales

## Output Files

The script generates several output files in the specified output directory:

- `{model_name}_{timestamp}_training_curves.png`: Training and validation loss/accuracy curves
- `{model_name}_{timestamp}_roc_curve.png`: ROC curve with AUC score
- `{model_name}_{timestamp}_attention.png`: Attention weight heatmap (CNN only)
- `{model_name}_{timestamp}_attention_importance.png`: Locus importance plot (CNN only)
- `{model_name}_{timestamp}.pt`: Trained model weights (if --save_model)
- `{model_name}_{timestamp}_results.json`: Training results summary

## Example Results

The script will output:
- Training progress every 10 epochs
- Final ROC AUC score
- Locus importance scores (for CNN model)
- Paths to saved files

## Customization

You can modify the model architectures by editing the classes in `genomic_image_cnn.py`:

- `GenomicImageCNN`: Main CNN with attention
- `MultiScaleGenomicImageCNN`: Multi-scale CNN
- `PermutationInvariantConv2D`: Custom conv layer with haplotype invariance

## Performance Tips

1. **GPU Usage**: Use `--device cuda` if you have a GPU available
2. **Batch Size**: Increase `--batch_size` for larger datasets (e.g., 64, 128)
3. **Epochs**: Start with fewer epochs (10-20) for testing, then increase for final training
4. **Memory**: Reduce batch size if you run out of GPU memory

## Troubleshooting

### Common Issues:

1. **CUDA out of memory**: Reduce batch size or use CPU
2. **File not found**: Check paths to embeddings and labels files
3. **Shape mismatch**: Ensure your data matches the expected format `[N, 4, 21, 960, 2]`

### Getting Help:

Run the script with `--help` to see all available options:
```bash
python genomic_image_cnn.py --help
```

## Citation

If you use this code in your research, please cite:

```bibtex
@software{genomic_image_cnn,
  title={Genomic Image CNN for Phenotypic Prediction},
  author={Your Name},
  year={2024},
  url={https://github.com/your-repo/genomic-image-cnn}
}
```
