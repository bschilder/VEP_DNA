# TreeSequenceCollection: Managing Multiple tskit Tree Sequences

This module provides efficient ways to load and work with multiple tskit tree sequences using built-in Python classes and tskit's native capabilities.

## Overview

The `TreeSequenceCollection` classes provide several approaches for handling multiple tree sequences:

1. **TreeSequenceCollection**: Loads all trees into memory for fast access
2. **LazyTreeSequenceCollection**: Loads trees on-demand to manage memory usage
3. **Utility functions**: For merging, filtering, and batch processing

## Installation

First, install tskit and tszip:

```bash
conda install -c conda-forge tskit tszip
```

## Quick Start

### Basic Usage

```python
import src.tskit as tskit_utils
from src.tskit import TreeSequenceCollection, create_tree_collection_from_directory

# Load all tree files from a directory
collection = create_tree_collection_from_directory("/path/to/tree/files")

# Iterate over all tree sequences
for name, ts in collection.items():
    print(f"{name}: {ts.num_samples} samples, {ts.num_sites} sites")

# Get summary statistics
stats = collection.get_summary_stats()
print(stats)
```

### Lazy Loading for Large Collections

```python
from src.tskit import LazyTreeSequenceCollection

# Create lazy collection (files loaded only when accessed)
tree_files = ["tree1.trees.tsz", "tree2.trees.tsz", "tree3.trees.tsz"]
lazy_collection = LazyTreeSequenceCollection(tree_files)

# Access individual trees (loaded on demand)
for i in range(len(lazy_collection)):
    ts = lazy_collection[i]
    print(f"Tree {i}: {ts.num_samples} samples")

# Clear cache when done
lazy_collection.clear_cache()
```

## Key Features

### 1. Multiple Iteration Methods

```python
# Iterate over tree sequences
for ts in collection:
    print(ts.num_samples)

# Iterate over (name, tree_sequence) pairs
for name, ts in collection.items():
    print(f"{name}: {ts.num_samples}")

# Iterate over all individual trees across all sequences
for name, tree_idx, tree in collection.iterate_all_trees():
    print(f"{name}[{tree_idx}]: {tree.interval}")

# Iterate over all sites across all sequences
for name, site_idx, site in collection.iterate_all_sites():
    print(f"{name}[{site_idx}]: {site.position}")

# Iterate over all mutations across all sequences
for name, mut_idx, mutation in collection.iterate_all_mutations():
    print(f"{name}[{mut_idx}]: {mutation.site}")
```

### 2. Functional Programming Methods

```python
# Map functions over tree sequences
sample_counts = collection.map_trees(lambda ts: ts.num_samples)
site_counts = collection.map_trees_with_names(lambda ts: ts.num_sites)

# Filter trees based on criteria
large_trees = collection.filter_trees(lambda ts: ts.num_sites > 1000)

# Custom analysis
def analyze_tree(ts):
    return {
        'num_samples': ts.num_samples,
        'num_sites': ts.num_sites,
        'diversity': ts.diversity() if ts.num_sites > 0 else 0
    }

analyses = collection.map_trees_with_names(analyze_tree)
```

### 3. Tree Merging

```python
from src.tskit import create_merged_collection

# Merge all trees into a single tree sequence
merged_ts = create_merged_collection(collection, method="concatenate")

# Merge in chunks to manage memory
chunked_merged = create_merged_collection(collection, method="concatenate", chunk_size=5)
```

### 4. Batch Processing

```python
from src.tskit import batch_process_trees

def process_batch(batch):
    return sum(ts.num_sites for ts in batch)

# Process in batches to manage memory
batch_results = batch_process_trees(collection, process_batch, batch_size=10)
```

## Memory Management Strategies

### For Large Collections

1. **Use LazyTreeSequenceCollection**:
   ```python
   lazy_collection = LazyTreeSequenceCollection(tree_files)
   for name, ts in lazy_collection.items():
       # Process one tree at a time
       process_tree(ts)
   lazy_collection.clear_cache()
   ```

2. **Use Batch Processing**:
   ```python
   def memory_intensive_analysis(batch):
       results = []
       for ts in batch:
           result = perform_analysis(ts)
           results.append(result)
       return results
   
   batch_results = batch_process_trees(collection, memory_intensive_analysis, batch_size=5)
   ```

3. **Merge in Chunks**:
   ```python
   merged = create_merged_collection(collection, chunk_size=10)
   ```

## Real-World Examples

### Population Genetic Analysis

```python
import json
import pandas as pd

def population_analysis(collection):
    results = {}
    
    for name, ts in collection.items():
        # Get population metadata
        pop_metadata = [json.loads(pop.metadata) for pop in ts.populations()]
        
        # Calculate diversity for each population
        pop_diversities = {}
        for pop_idx, pop in enumerate(ts.populations()):
            if pop.metadata:
                pop_name = json.loads(pop.metadata).get('name', f'pop_{pop_idx}')
                pop_samples = [i for i in range(ts.num_samples) 
                             if ts.node(ts.sample(i)).population == pop_idx]
                if pop_samples:
                    diversity = ts.diversity(sample_sets=[pop_samples])
                    pop_diversities[pop_name] = diversity[0]
        
        results[name] = {
            'num_samples': ts.num_samples,
            'num_sites': ts.num_sites,
            'total_diversity': ts.diversity()[0] if ts.num_sites > 0 else 0,
            'population_diversities': pop_diversities
        }
    
    return results

# Run analysis
results = population_analysis(collection)
```

### Site-Based Analysis

```python
import pandas as pd

def site_analysis(collection):
    site_data = []
    
    for name, site_idx, site in collection.iterate_all_sites():
        site_data.append({
            'tree_name': name,
            'site_index': site_idx,
            'position': site.position,
            'ancestral_state': site.ancestral_state,
            'num_mutations': len(site.mutations)
        })
    
    return pd.DataFrame(site_data)

# Run analysis
site_df = site_analysis(collection)
```

## Performance Tips

1. **For small collections** (< 100 trees): Use `TreeSequenceCollection`
2. **For large collections** (> 100 trees): Use `LazyTreeSequenceCollection`
3. **For memory-intensive operations**: Use batch processing
4. **For analysis across all trees**: Use the iterator methods
5. **For combining trees**: Use merging functions with appropriate chunk sizes

## API Reference

### TreeSequenceCollection

- `__init__(tree_sequences, names=None, metadata=None)`
- `__len__()`: Number of tree sequences
- `__getitem__(idx)`: Get tree sequence by index or name
- `__iter__()`: Iterate over tree sequences
- `items()`: Iterate over (name, tree_sequence) pairs
- `map_trees(func, *args, **kwargs)`: Apply function to all trees
- `map_trees_with_names(func, *args, **kwargs)`: Apply function with names
- `filter_trees(predicate)`: Filter trees based on predicate
- `get_summary_stats()`: Get summary statistics
- `iterate_all_trees()`: Iterate over all individual trees
- `iterate_all_sites()`: Iterate over all sites
- `iterate_all_mutations()`: Iterate over all mutations

### LazyTreeSequenceCollection

- `__init__(tree_files, names=None, metadata=None)`
- `__len__()`: Number of tree files
- `__getitem__(idx)`: Get tree sequence (loads if needed)
- `__iter__()`: Iterate over tree sequences
- `items()`: Iterate over (name, tree_sequence) pairs
- `preload_all()`: Load all trees and return TreeSequenceCollection
- `clear_cache()`: Clear loaded sequence cache

### Utility Functions

- `create_tree_collection_from_files(tree_files, names=None, metadata=None, show_progress=True)`
- `create_tree_collection_from_directory(directory, suffix=".trees.tsz", names=None, metadata=None, show_progress=True)`
- `merge_tree_sequences(tree_sequences, method="concatenate")`
- `create_merged_collection(tree_collection, method="concatenate", chunk_size=None)`
- `batch_process_trees(tree_collection, func, batch_size=10, *args, **kwargs)`

## Examples

See `examples/tree_collection_usage.py` for comprehensive examples of all features. 