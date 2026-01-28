#!/usr/bin/env python3
"""
Example usage of the Genomic Image CNN script

This script demonstrates how to use the genomic_image_cnn.py script with different options.
"""

import subprocess
import sys
import os

def run_command(cmd):
    """Run a command and print the output"""
    print(f"Running: {' '.join(cmd)}")
    print("-" * 50)
    result = subprocess.run(cmd, capture_output=True, text=True)
    print(result.stdout)
    if result.stderr:
        print("STDERR:", result.stderr)
    print("-" * 50)
    return result.returncode == 0

def main():
    print("Genomic Image CNN - Example Usage")
    print("=" * 50)
    
    # Example 1: Train with synthetic data (for testing)
    print("\n1. Training with synthetic data (CNN model):")
    success = run_command([
        sys.executable, "genomic_image_cnn.py",
        "--synthetic",
        "--model_type", "cnn",
        "--epochs", "10",
        "--batch_size", "16",
        "--output_dir", "./example_results",
        "--analyze_attention",
        "--save_model"
    ])
    
    if not success:
        print("Example 1 failed!")
        return
    
    # Example 2: Train multi-scale model with synthetic data
    print("\n2. Training with synthetic data (Multi-scale model):")
    success = run_command([
        sys.executable, "genomic_image_cnn.py",
        "--synthetic",
        "--model_type", "multiscale",
        "--epochs", "10",
        "--batch_size", "16",
        "--output_dir", "./example_results"
    ])
    
    if not success:
        print("Example 2 failed!")
        return
    
    # Example 3: Show help
    print("\n3. Help information:")
    run_command([sys.executable, "genomic_image_cnn.py", "--help"])
    
    print("\n" + "=" * 50)
    print("Examples completed!")
    print("\nTo use with your own data:")
    print("python genomic_image_cnn.py --embeddings path/to/embeddings.pt --labels path/to/labels.pt")
    print("\nFor more options, run:")
    print("python genomic_image_cnn.py --help")

if __name__ == "__main__":
    main()
