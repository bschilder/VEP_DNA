from sys import path
import numpy as np
import pandas as pd
import logging
import tensorflow as tf
from tensorflow.keras import layers, Input, Model
from typing import Tuple, List, Dict, Optional
import gc
import time
from random import choice
import os
from math import ceil
from tqdm import tqdm
from typing import Dict

def setup_gpu_optimizations():
    """
    Set up GPU optimizations for H100. 
    MUST be called before any TensorFlow operations!
    
    Returns:
        bool: True if all optimizations were successfully applied
    """
    success = True
    print("Setting up H100 GPU optimizations...")
    
    # 1. GPU Memory Growth (must be first)
    try:
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print(f"✓ Memory growth enabled for {len(gpus)} GPU(s)")
        else:
            print("⚠ No GPUs found")
            success = False
    except RuntimeError as e:
        print(f"⚠ GPU memory configuration failed: {e}")
        print("  (This is normal if TensorFlow has already initialized)")
        success = False

    # 2. Mixed Precision
    try:
        policy = tf.keras.mixed_precision.Policy('mixed_float16')
        tf.keras.mixed_precision.set_global_policy(policy)
        print("✓ Mixed precision (FP16) enabled")
    except Exception as e:
        print(f"⚠ Mixed precision setup failed: {e}")
        success = False
    
    # 3. XLA Compilation
    try:
        tf.config.optimizer.set_jit(True)
        print("✓ XLA JIT compilation enabled")
    except Exception as e:
        print(f"⚠ XLA setup failed: {e}")
        success = False
    
    return success

def one_hot_encode_batch(seq: str, 
                transpose: bool = False,
                **kwargs) -> np.ndarray:
    # mapping = {'A':0,'C':1,'G':2,'T':3}
    # arr = np.zeros((4, len(seq)), dtype=np.float32)
    # for i, b in enumerate(seq):
    #     idx = mapping.get(b)
    #     if idx is not None:
    #         arr[idx, i] = 1.0
    import seqpro as sp
    return sp.DNA.ohe(seq, **kwargs).T if transpose else sp.DNA.ohe(seq, **kwargs)

class OptimizedSpliceAIScorer:
    """
    Vectorized SpliceAI scorer optimized for large-scale processing on H100 GPUs
    
    This class implements the SpliceAI scoring algorithm in a vectorized manner,
    processing batches of sequences to compute splice site gain/loss scores for
    both positive and negative strands.

    IMPORTANT: Call setup_gpu_optimizations() before loading models and creating this scorer.
    """
    # For SpliceAI:ls
    # - Context window: 10,000 bp (5K upstream + 5K downstream)  
    # - Prediction region: Variable (e.g., 5, 101, 201 positions)
    # - Total input length: 10,000 + prediction_region_length
    # - Model output length: prediction_region_length (only the center region)
    
    # Example:
    # To predict 101 positions: input = (B, 10101, 4), output = (B, 101, 3)
    # To predict 5 positions: input = (B, 10005, 4), output = (B, 5, 3)
    def __init__(self, 
                 models: List, 
                 use_ensemble: bool = True,
                 mask=False):
        """
        Initialize SpliceAI scorer with ensemble of models
        
        Args:
            models: List of trained TensorFlow/Keras models for ensemble prediction
                    Each model should accept input of shape (batch_size, context_length + prediction_length, 4)
                    and output shape (batch_size, prediction_length, 3)
            use_ensemble: Whether to create ensemble model for parallel execution (recommended for H100)
        """
        self.models = models
        self.num_models = len(models)
        self.use_ensemble = use_ensemble
        self.mask = mask
        # Check if optimizations are enabled
        self._check_optimizations()
        
        # Setup ensemble model if requested
        if use_ensemble:
            self._setup_optimized_ensemble()
        else:
            self.ensemble_model = None
    
    def _check_optimizations(self):
        """Check and report which optimizations are active"""
        print("Checking active optimizations:")
        
        # Check mixed precision
        policy = tf.keras.mixed_precision.global_policy()
        if policy.name == 'mixed_float16':
            print("✓ Mixed precision (FP16) is enabled")
        else:
            print(f"⚠ Mixed precision is NOT enabled (current: {policy.name})")
        
        # Check XLA
        if tf.config.optimizer.get_jit():
            print("✓ XLA JIT compilation is enabled")
        else:
            print("⚠ XLA JIT compilation is NOT enabled")
        
        # Check GPU availability
        gpus = tf.config.experimental.list_physical_devices('GPU')
        if gpus:
            print(f"✓ Found {len(gpus)} GPU(s)")
            for i, gpu in enumerate(gpus):
                try:
                    growth = tf.config.experimental.get_memory_growth(gpu)
                    print(f"  GPU {i}: Memory growth = {growth}")
                except:
                    pass
        else:
            print("⚠ No GPUs found")
    
    def _setup_optimized_ensemble(self):
        """Create optimized ensemble model with mixed precision support"""
        try:
            print(f"Creating ensemble from {len(self.models)} models...")
            
            # Get input shape from first model
            input_shape = self.models[0].input_shape
            print(f"Input shape: {input_shape}")
            
            # Option 1: Try using models directly in a functional way
            # Create input layer
            inputs = tf.keras.Input(shape=input_shape[1:], name='ensemble_input')
            
            # Get predictions from each model
            predictions = []
            for i, model in enumerate(self.models):
                # Use Lambda layer to wrap the model call with a unique name
                pred = tf.keras.layers.Lambda(
                    lambda x, m=model: m(x, training=False),
                    name=f'model_{i}_prediction'
                )(inputs)
                predictions.append(pred)
            
            # Average the predictions
            if len(predictions) > 1:
                averaged = tf.keras.layers.Average(name='ensemble_average')(predictions)
            else:
                averaged = predictions[0]
            
            # Create the ensemble model
            self.ensemble_model = tf.keras.Model(
                inputs=inputs, 
                outputs=averaged, 
                name='spliceai_ensemble'
            )
            
            print(f"✓ Ensemble model created successfully")
            
        except Exception as e:
            print(f"⚠ Ensemble creation failed: {e}")
            print("Trying alternative ensemble method...")
            
            # Option 2: Create a simple averaging function
            try:
                self._create_manual_ensemble()
            except Exception as e2:
                print(f"⚠ Alternative ensemble also failed: {e2}")
                print("Will use sequential prediction (still optimized with mixed precision)")
                self.ensemble_model = None
    
    def _create_manual_ensemble(self):
        """Create a manual ensemble that averages predictions"""
        # This is a simpler approach that doesn't create a combined model
        # but still provides the averaging functionality
        
        def ensemble_predict(x, training=False):
            """Manual ensemble prediction"""
            predictions = []
            for model in self.models:
                pred = model(x, training=training)
                predictions.append(pred)
            
            # Average predictions
            return tf.reduce_mean(tf.stack(predictions, axis=0), axis=0)
        
        # Create a simple wrapper model
        input_shape = self.models[0].input_shape
        inputs = tf.keras.Input(shape=input_shape[1:])
        outputs = tf.keras.layers.Lambda(ensemble_predict)(inputs)
        
        self.ensemble_model = tf.keras.Model(inputs=inputs, outputs=outputs)
        print("✓ Manual ensemble created")
 
    def predict_batch(self, x_ref: np.ndarray, x_alt: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model predictions for a batch of sequences
        
        Processes both reference and alternate sequences through the ensemble of models
        for both forward (positive) and reverse complement (negative) orientations.
        
        Args:
            x_ref: Reference sequences of shape (B, C+P, 4) where:
                   - B = batch size
                   - C = context length (typically 10,000 bp for SpliceAI)
                   - P = prediction region length (center region to be predicted)
                   - 4 = one-hot encoded nucleotides [A, C, G, T]
            x_alt: Alternate sequences of shape (B, C+P, 4) with same format as x_ref
            
        Returns:
            Tuple of 2 prediction arrays:
            - y_pos: Positive strand predictions of shape (B, 2, P, 3) where:
                     [:,0,:,:] = reference predictions (shape: B, P, 3)
                     [:,1,:,:] = alternate predictions (shape: B, P, 3)
                     Last dim: [no_splice, acceptor, donor] probabilities
            - y_neg: Negative strand predictions of shape (B, 2, P, 3) with same format
                     (reverse complement sequences with reversed position dimension)
        """
        B = x_ref.shape[0]
        pred_length = x_ref.shape[1] - 10000
        pred_shape = (B, pred_length, 3)

        # Reverse complement in NumPy (reverse both sequence and nucleotide dimensions)
        x_ref_rc = x_ref[:, ::-1, ::-1]
        x_alt_rc = x_alt[:, ::-1, ::-1]

        # Stack all input variants: (4B, C+P, 4)
        x_all = np.concatenate([x_ref, x_alt, x_ref_rc, x_alt_rc], axis=0)

        # Predict using model.predict with automatic batching
        if self.ensemble_model is not None:
            y_all = self.ensemble_model.predict(x_all, batch_size=8192, verbose=1)
        else:
            # Fallback to sequential prediction (ensemble average across models)
            # Model input: (B, C+P, 4) → Model output: (B, P, 3)
            y_all = np.mean([
                model.predict(x_all, batch_size=1000, verbose=1)
                for model in self.models
            ], axis=0)

        # Split predictions back into 4 groups
        y_ref    = y_all[      :B      ]  # (B, P, 3)
        y_alt    = y_all[  B : 2*B     ]  # (B, P, 3)
        y_ref_rc = y_all[2*B : 3*B][:, ::-1, :]  # (B, P, 3) - reverse positions
        y_alt_rc = y_all[3*B :     ][:, ::-1, :]  # (B, P, 3) - reverse positions

        # Concatenate ref and alt predictions along new dimension
        y_pos = np.stack([y_ref, y_alt], axis=1)  # (B, 2, P, 3)
        y_neg = np.stack([y_ref_rc, y_alt_rc], axis=1)  # (B, 2, P, 3)

        return y_pos, y_neg

    def calculate_splicing_scores(self, 
                                  y: np.ndarray, 
                                  anno_3ss: np.ndarray, 
                                  anno_5ss: np.ndarray) -> Tuple[np.ndarray, ...]:
        """
        Calculate the four delta scores with masking
        
        Computes acceptor/donor gain/loss scores by comparing alternate vs reference
        predictions and applying annotation-based masking to reduce false positives.
        
        Args:
            y: Model predictions array of shape (B, 2, P, 3) where:
               - B = batch size (number of sequences)
               - 2 = [reference, alternate] predictions
               - P = prediction region length (center region of input sequence)
               - 3 = [no_splice_site, acceptor, donor] probabilities
            anno_3ss: 3' splice site (acceptor) annotations of shape (B, P) where:
                      1 = annotated acceptor site present, 0 = absent
            anno_5ss: 5' splice site (donor) annotations of shape (B, P) where:
                      1 = annotated donor site present, 0 = absent
            
        Returns:
            Tuple of 4 arrays, each of shape (B, P):
            - acceptor_gain: (alt - ref) acceptor scores, masked at annotated acceptors
            - acceptor_loss: (ref - alt) acceptor scores, masked at non-annotated positions  
            - donor_gain: (alt - ref) donor scores, masked at annotated donors
            - donor_loss: (ref - alt) donor scores, masked at non-annotated positions
        """
        if self.mask:
            # Pre-compute masks and ensure float32 for consistency
            mask_3ss_inv = (1 - anno_3ss).astype(np.float32)
            mask_5ss_inv = (1 - anno_5ss).astype(np.float32)
            anno_3ss_f32 = anno_3ss.astype(np.float32)
            anno_5ss_f32 = anno_5ss.astype(np.float32)
            
            # Compute differences once
            acceptor_diff = y[:, 1, :, 1] - y[:, 0, :, 1]  # alt - ref
            donor_diff = y[:, 1, :, 2] - y[:, 0, :, 2]     # alt - ref
            
            # Apply masks
            acceptor_gain = acceptor_diff * mask_3ss_inv
            acceptor_loss = -acceptor_diff * anno_3ss_f32
            donor_gain = donor_diff * mask_5ss_inv
            donor_loss = -donor_diff * anno_5ss_f32

        else:
            # Compute differences once
            acceptor_diff = y[:, 1, :, 1] - y[:, 0, :, 1]  # alt - ref
            donor_diff = y[:, 1, :, 2] - y[:, 0, :, 2]     # alt - ref
            
            # Apply masks
            acceptor_gain = acceptor_diff
            acceptor_loss = -acceptor_diff 
            donor_gain = donor_diff 
            donor_loss = -donor_diff
        
        return acceptor_gain, acceptor_loss, donor_gain, donor_loss
    
    def get_max_scores_and_coords(self, 
                                  scores: Tuple[np.ndarray, ...], 
                                  ref_coord: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get maximum scores, indices, and coordinates for all score types
        
        For each sequence and each score type, finds the position with maximum
        score and maps it to genomic coordinates.
        
        Args:
            scores: Tuple of 4 score arrays, each of shape (B, P):
                    (acceptor_gain, acceptor_loss, donor_gain, donor_loss)
            ref_coord: Reference coordinate mapping of shape (B, P) where:
                       ref_coord[i, j] = genomic coordinate for sequence i at position j
            
        Returns:
            Dictionary with 12 arrays, each of shape (B,):
            - '{score_type}_max': Maximum score value for each sequence
            - '{score_type}_max_idx': Position index of maximum score (0 to P-1)
            - '{score_type}_max_coord': Genomic coordinate of maximum score position
            
            Where score_type ∈ ['acceptor_gain', 'acceptor_loss', 'donor_gain', 'donor_loss']
        """
        score_names = ['acceptor_gain', 'acceptor_loss', 'donor_gain', 'donor_loss']
        
        # Stack all scores for vectorized operations
        all_scores = np.stack(scores, axis=2)  # Shape: (B, P, 4)
        
        # Find max along position axis in one go
        max_scores = np.max(all_scores, axis=1)      # Shape: (B, 4)
        max_indices = np.argmax(all_scores, axis=1)  # Shape: (B, 4)
        
        # Get coordinates using advanced indexing
        batch_size = scores[0].shape[0]
        batch_idx = np.arange(batch_size)[:, np.newaxis]  # Shape: (B, 1)
        max_coords = ref_coord[batch_idx, max_indices]    # Shape: (B, 4)
        
        # Build results dictionary
        results = {}
        for i, name in enumerate(score_names):
            results[f'{name}_max'] = max_scores[:, i]
            results[f'{name}_max_idx'] = max_indices[:, i]
            results[f'{name}_max_coord'] = max_coords[:, i]
        
        return results
    
    def get_overall_max_delta(self, pos_results: Dict[str, np.ndarray], 
                             neg_results: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Find the maximum delta score across all 8 score types for each sequence
        
        Args:
            pos_results: Dictionary with positive strand results, each array of shape (B,)
            neg_results: Dictionary with negative strand results, each array of shape (B,)
            
        Returns:
            Dictionary with overall maximum information, each array of shape (B,):
            - 'max_delta_score': Highest score across all 8 score types
            - 'max_delta_coord': Genomic coordinate of the maximum score
            - 'max_delta_score_type': Which score type achieved the maximum
            - 'max_delta_strand': Which strand ('pos' or 'neg') achieved the maximum
        """
        # Combine all 8 max scores into a matrix: (B, 8)
        all_scores = np.column_stack([
            pos_results['acceptor_gain_max'],
            pos_results['acceptor_loss_max'], 
            pos_results['donor_gain_max'],
            pos_results['donor_loss_max'],
            neg_results['acceptor_gain_max'],
            neg_results['acceptor_loss_max'],
            neg_results['donor_gain_max'],
            neg_results['donor_loss_max']
        ])
        
        # Combine all 8 coordinates: (B, 8)
        all_coords = np.column_stack([
            pos_results['acceptor_gain_max_coord'],
            pos_results['acceptor_loss_max_coord'],
            pos_results['donor_gain_max_coord'], 
            pos_results['donor_loss_max_coord'],
            neg_results['acceptor_gain_max_coord'],
            neg_results['acceptor_loss_max_coord'],
            neg_results['donor_gain_max_coord'],
            neg_results['donor_loss_max_coord']
        ])

        # Find maximum across all score types for each sequence
        max_indices = np.argmax(all_scores, axis=1)  # Shape: (B,)
        batch_indices = np.arange(len(all_scores))
        
        max_scores = all_scores[batch_indices, max_indices]
        max_coords = all_coords[batch_indices, max_indices]
        
        # Map indices to score type names and strands
        score_type_names = [
            'acceptor_gain', 'acceptor_loss', 'donor_gain', 'donor_loss',  # pos strand
            'acceptor_gain', 'acceptor_loss', 'donor_gain', 'donor_loss'   # neg strand
        ]
        strand_names = ['pos', 'pos', 'pos', 'pos', 'neg', 'neg', 'neg', 'neg']
        
        max_score_types = np.array([score_type_names[i] for i in max_indices])
        max_strands = np.array([strand_names[i] for i in max_indices])
        
        return {
            'max_delta_score': max_scores,
            'max_delta_coord': max_coords,
            'max_delta_score_type': max_score_types,
            'max_delta_strand': max_strands
        }
    
    def score_batch(self, batch_data: Dict) -> Dict[str, np.ndarray]:
        """
        Score a batch of sequences for splice site effects
        
        Complete pipeline: sequences → model predictions → delta scores → max values → overall max.
        Processes both positive and negative strands independently, then finds overall maximum.
        
        Args:
            batch_data: Dictionary containing all required data with keys:
                - 'x_ref': Reference sequences, shape (B, C+P, 4)
                - 'x_alt': Alternate sequences, shape (B, C+P, 4)  
                - 'anno_3ss_pos': 3'SS annotations for positive strand, shape (B, P)
                - 'anno_5ss_pos': 5'SS annotations for positive strand, shape (B, P)
                - 'anno_3ss_neg': 3'SS annotations for negative strand, shape (B, P)
                - 'anno_5ss_neg': 5'SS annotations for negative strand, shape (B, P)
                - 'ref_coord': Reference coordinate mapping, shape (B, P)
                - 'metadata': List of B metadata dictionaries (optional), shape (B,)
                
                Where:
                - B = batch size
                - C = context length (typically 10,000 bp)
                - P = prediction region length

            Metadata Format:
                Each metadata entry should be a dictionary with the following structure:
                {
                    # Required fields:
                    'chromosome': str,           # Chromosome name, e.g., 'chr1', '1', 'X'
                    'position': int,             # Genomic position (1-based), e.g., 1000000
                    'ref_allele': str,           # Reference allele sequence, e.g., 'A', 'ATCG'
                    'alt_allele': str,           # Alternate allele sequence, e.g., 'T', 'A'
                    'hap_id': srt,               # 1K genome haplotype identifier
                    'sample_id': str,            # 1K genome sample identifier, each sample is an individual with 2 haplotypes
                    'variant_id': str,           # Unique variant identifier, e.g., 'chr1:1000000_A_T'

                    # Optional fields:
                    'gene_id': str,              # Gene identifier, e.g., 'ENSG00000123456'
                    'gene_name': str,            # Gene symbol, e.g., 'BRCA1'
                    'transcript_id': str,        # Transcript identifier, e.g., 'ENST00000123456'
                    'variant_type': str,         # Type of variant, e.g., 'SNV', 'indel', 'deletion'
                    'sample_id': str,            # Sample identifier, e.g., 'sample_001'
                    'quality_score': float,      # Variant quality score, e.g., 99.5
                    'allele_frequency': float,   # Population allele frequency, e.g., 0.001
                    'clinical_significance': str, # Clinical annotation, e.g., 'pathogenic'
                    'custom_annotations': dict   # Any additional custom fields
                }
                
        Returns:
            Dictionary with strand-prefixed results, all arrays of shape (B,):
            
            Positive strand results (prefix 'pos_'), all have shape (B,):
            - 'pos_acceptor_gain_max', 'pos_acceptor_gain_max_idx', 'pos_acceptor_gain_max_coord'
            - 'pos_acceptor_loss_max', 'pos_acceptor_loss_max_idx', 'pos_acceptor_loss_max_coord'  
            - 'pos_donor_gain_max', 'pos_donor_gain_max_idx', 'pos_donor_gain_max_coord'
            - 'pos_donor_loss_max', 'pos_donor_loss_max_idx', 'pos_donor_loss_max_coord'
            
            Negative strand results (prefix 'neg_'), all have shape (B,):
            - 'neg_acceptor_gain_max', 'neg_acceptor_gain_max_idx', 'neg_acceptor_gain_max_coord'
            - 'neg_acceptor_loss_max', 'neg_acceptor_loss_max_idx', 'neg_acceptor_loss_max_coord'
            - 'neg_donor_gain_max', 'neg_donor_gain_max_idx', 'neg_donor_gain_max_coord'  
            - 'neg_donor_loss_max', 'neg_donor_loss_max_idx', 'neg_donor_loss_max_coord'
            
            Overall maximum results, all have shape (B,):
            - 'max_delta_score': Highest score across all 8 score types, shape (B,)
            - 'max_delta_coord': Genomic coordinate of maximum score, shape (B,)
            - 'max_delta_score_type': Score type of maximum, shape (B,) dtype=object
            - 'max_delta_strand': Strand of maximum ('pos'/'neg'), shape (B,) dtype=object

            Optional meta data, a list of len B, each element is a dictionary.
            
        Example Usage:
            # Process batch of 1000 variants
            batch_data = {
                'x_ref': np.array(...),          # shape: (1000, 10101, 4)
                'x_alt': np.array(...),          # shape: (1000, 10101, 4)
                'anno_3ss_pos': np.array(...),   # shape: (1000, 101)
                'anno_5ss_pos': np.array(...),   # shape: (1000, 101)
                'anno_3ss_neg': np.array(...),   # shape: (1000, 101)
                'anno_5ss_neg': np.array(...),   # shape: (1000, 101)
                'ref_coord': np.array(...),      # shape: (1000, 101)
                'metadata': [dict1, dict2, ...]  # length: 1000
            }
            
            results = scorer.score_batch(batch_data)
        """
        # Get predictions for both strands
        y_pos, y_neg = self.predict_batch(batch_data['x_ref'], batch_data['x_alt'])
        
        # Calculate scores for positive strand
        pos_scores = self.calculate_splicing_scores(
            y_pos, batch_data['anno_3ss_pos'], batch_data['anno_5ss_pos']
        )
        pos_results = self.get_max_scores_and_coords(pos_scores, batch_data['ref_coord'])
        
        # Calculate scores for negative strand
        neg_scores = self.calculate_splicing_scores(
            y_neg, batch_data['anno_3ss_neg'], batch_data['anno_5ss_neg']
        )
        neg_results = self.get_max_scores_and_coords(neg_scores, batch_data['ref_coord'])

        # Find overall maximum across all 8 score types
        max_delta_results = self.get_overall_max_delta(pos_results, neg_results)
        
        # Combine results with strand prefixes
        results = {}
        for key, value in pos_results.items():
            results[f'pos_{key}'] = value
        for key, value in neg_results.items():
            results[f'neg_{key}'] = value

        # Add overall maximum results
        results.update(max_delta_results)
        
        # Add metadata if provided
        if 'metadata' in batch_data:
            results['metadata'] = batch_data['metadata']
            
        return results
    
    def time_scoring_breakdown(self, batch_data: Dict) -> Dict[str, float]:
        """
        Time different parts of the scoring pipeline without using profiler
        
        Args:
            batch_data: Same as score_batch input
            
        Returns:
            Dictionary with timing for each step in seconds
        """
        import time
        
        timings = {}
        
        # Time prediction
        start = time.perf_counter()
        y_pos, y_neg = self.predict_batch(batch_data['x_ref'], batch_data['x_alt'])
        timings['prediction'] = time.perf_counter() - start
        
        # Time positive strand scoring
        start = time.perf_counter()
        pos_scores = self.calculate_splicing_scores(
            y_pos, batch_data['anno_3ss_pos'], batch_data['anno_5ss_pos']
        )
        pos_results = self.get_max_scores_and_coords(pos_scores, batch_data['ref_coord'])
        timings['pos_strand_scoring'] = time.perf_counter() - start
        
        # Time negative strand scoring
        start = time.perf_counter()
        neg_scores = self.calculate_splicing_scores(
            y_neg, batch_data['anno_3ss_neg'], batch_data['anno_5ss_neg']
        )
        neg_results = self.get_max_scores_and_coords(neg_scores, batch_data['ref_coord'])
        timings['neg_strand_scoring'] = time.perf_counter() - start
        
        # Time final aggregation
        start = time.perf_counter()
        max_delta_results = self.get_overall_max_delta(pos_results, neg_results)
        
        # Combine results
        results = {}
        for key, value in pos_results.items():
            results[f'pos_{key}'] = value
        for key, value in neg_results.items():
            results[f'neg_{key}'] = value
        results.update(max_delta_results)
        
        if 'metadata' in batch_data:
            results['metadata'] = batch_data['metadata']
        
        timings['aggregation'] = time.perf_counter() - start
        timings['total'] = sum(timings.values())
        
        # Print breakdown
        print("\nScoring Time Breakdown:")
        print("-" * 40)
        for step, duration in timings.items():
            percentage = (duration / timings['total'] * 100) if step != 'total' else 100
            print(f"{step:20s}: {duration:6.3f}s ({percentage:5.1f}%)")
        
        batch_size = batch_data['x_ref'].shape[0]
        print(f"\nThroughput: {batch_size / timings['total']:.1f} sequences/second")
        
        try:
            memory_info = tf.config.experimental.get_memory_info('GPU:0')
            memory_gb = memory_info.get('current', 0) / 1e9
            memory_gb_peak = memory_info.get('peak', 0) / 1e9
        except:
            memory_gb = 0
            memory_gb_peak = 0
            
        print(f"{memory_gb:16.2f} | {memory_gb_peak:13.2f}")
        
        return timings
    
    def benchmark_performance(self, test_batch_sizes: List[int] = [100, 500, 1000, 5000]):
        """
        Benchmark performance with different batch sizes
        """
        print("\nBenchmarking SpliceAI performance on H100...")
        print("Batch Size | Time (s) | Sequences/sec | Current Mem (GB) | Peak Mem (GB)")
        print("-" * 75)
        
        # Create dummy data for benchmarking
        context_length = 10000
        pred_length = 101
        
        for batch_size in test_batch_sizes:
            try:
                # Create test data
                x_shape = (batch_size, context_length + pred_length, 4)
                test_data = {
                    'x_ref': np.random.rand(*x_shape).astype(np.float32),
                    'x_alt': np.random.rand(*x_shape).astype(np.float32),
                    'anno_3ss_pos': np.zeros((batch_size, pred_length), dtype=np.float32),
                    'anno_5ss_pos': np.zeros((batch_size, pred_length), dtype=np.float32),
                    'anno_3ss_neg': np.zeros((batch_size, pred_length), dtype=np.float32),
                    'anno_5ss_neg': np.zeros((batch_size, pred_length), dtype=np.float32),
                    'ref_coord': np.arange(pred_length)[None, :].repeat(batch_size, axis=0)
                }
                
                # Warm up
                _ = self.score_batch(test_data)
                
                # Reset memory stats
                tf.config.experimental.reset_memory_stats('GPU:0')
                
                # Time the operation
                start_time = time.perf_counter()
                _ = self.score_batch(test_data)
                end_time = time.perf_counter()
                
                elapsed = end_time - start_time
                throughput = batch_size / elapsed
                
                # Get memory usage
                try:
                    memory_info = tf.config.experimental.get_memory_info('GPU:0')
                    memory_gb = memory_info.get('current', 0) / 1e9
                    memory_gb_peak = memory_info.get('peak', 0) / 1e9
                except:
                    memory_gb = 0
                    memory_gb_peak = 0
                    
                print(f"{batch_size:10d} | {elapsed:8.3f} | {throughput:13.1f} | "
                      f"{memory_gb:16.2f} | {memory_gb_peak:13.2f}")
                
            except tf.errors.ResourceExhaustedError:
                print(f"{batch_size:10d} | OOM - batch size too large")
                break
            except Exception as e:
                print(f"{batch_size:10d} | Error: {str(e)}")

def process_spliceai_batch_direct(batch_data, chromosome='chr22'):
    """
    Process batch data with corrected track dimensions
    """
    # Unpack
    (ref_allele, var_allele, tracks, variant_idx, sample_idx, 
     hap_idx, var_pos, var_nt, ref_nt, ref_index, flag) = batch_data
        
    batch_size = ref_allele.shape[0]
    print(f"Processing batch of size: {batch_size}")
    
    # One-hot encode sequences
    x_ref = one_hot_encode_batch(ref_allele)
    x_alt = one_hot_encode_batch(var_allele)
    print(f"Encoded sequences: ref {x_ref.shape}, alt {x_alt.shape}")
    
    # FIX: Transpose tracks to get correct dimension order
    print(f"Original tracks shape: {tracks.shape}")
    
    if tracks.shape[1] == 4 and tracks.shape[2] > tracks.shape[1]:
        # Shape is (batch_size, 4_channels, seq_length) - need to transpose
        tracks_transposed = np.transpose(tracks, (0, 2, 1))  # -> (batch_size, seq_length, 4_channels)
        print(f"Transposed tracks shape: {tracks_transposed.shape}")
        tracks = tracks_transposed
    elif tracks.shape[2] == 4:
        # Already in correct format (batch_size, seq_length, 4_channels)
        print("Tracks already in correct format")
    else:
        raise ValueError(f"Unexpected tracks shape: {tracks.shape}. Expected either (batch, 4, seq_len) or (batch, seq_len, 4)")
    
    # Extract splice site annotations from corrected tracks
    anno_3ss_pos = tracks[:, :, 0].astype(np.float32)
    anno_5ss_pos = tracks[:, :, 1].astype(np.float32)
    anno_3ss_neg = tracks[:, :, 2].astype(np.float32)
    anno_5ss_neg = tracks[:, :, 3].astype(np.float32)
    
    print(f"Annotation shapes: 3ss_pos {anno_3ss_pos.shape}, 5ss_pos {anno_5ss_pos.shape}")
    
    # Create metadata for each variant-haplotype
    metadata = []
    for i in range(batch_size):
        # Handle different nucleotide formats (your data shows both byte strings and regular strings)
        if hasattr(ref_nt[i], 'decode'):
            ref_str = ref_nt[i].decode('utf-8')
        else:
            ref_str = str(ref_nt[i])
            
        if hasattr(var_nt[i], 'decode'):
            alt_str = var_nt[i].decode('utf-8')
        else:
            alt_str = str(var_nt[i])
        
        meta = {
            'variant_idx': int(variant_idx[i]),
            'chromosome': chromosome,
            'position': int(var_pos[i]),
            'ref_allele': ref_str,
            'alt_allele': alt_str,
            'sample_id': int(sample_idx[i]),
            'hap_id': int(hap_idx[i]),
            'flag': int(flag[i])
        }
        metadata.append(meta)
    
    # Format for scorer
    batch_dict = {
        'x_ref': x_ref,
        'x_alt': x_alt,
        'anno_3ss_pos': anno_3ss_pos,
        'anno_5ss_pos': anno_5ss_pos,
        'anno_3ss_neg': anno_3ss_neg,
        'anno_5ss_neg': anno_5ss_neg,
        'ref_coord': ref_index,
        'metadata': metadata
    }
    
    return batch_dict


def run_spliceai_scoring(dataset, scorer, output_path: str, batch_size: int = 1000, chromosome=None):
    """
    Run SpliceAI scoring and save to parquet
    
    Args:
        dataset: SpliceHapDataset instance
        scorer: OptimizedSpliceAIScorer instance
        output_path: Path for output parquet file
        batch_size: Batch size for processing
    """
    # Calculate total batches for progress bar
    total_batches = ceil(len(dataset) / batch_size)
    
    print(f"Processing {len(dataset)} variant-haplotype combinations")
    print(f"Total batches: {total_batches}")
    print(f"Batch size: {batch_size}")
    
    # Initialize results list
    all_results = []
    
    # Process batches using dataset's batch_iter
    start_time = time.time()
    
    for batch_data in tqdm(dataset.batch_iter(batch_size), 
                          desc="Processing batches", 
                          total=total_batches):
        # Process batch through dataset
        processed_batch = process_spliceai_batch_direct(batch_data, chromosome=chromosome)
        
        # Score with SpliceAI
        results = scorer.score_batch(processed_batch)
        
        # Convert to flat format
        batch_df = results_to_dataframe(results)
        all_results.append(batch_df)
    
    # Combine all results
    print("\nCombining results...")
    final_df = pd.concat(all_results, ignore_index=True)
    
    # Save to parquet
    print(f"Saving to {output_path}")
    final_df.to_parquet(output_path, index=False, compression='snappy')
    
    end_time = time.time()
    total_time = end_time - start_time
    
    print(f"\nComplete!")
    print(f"Total records: {len(final_df)}")
    print(f"Total time: {total_time/60:.2f} minutes")
    print(f"File size: {final_df.memory_usage(deep=True).sum() / 1e6:.1f} MB (in memory)")
    
    return final_df

def results_to_dataframe(results: Dict) -> pd.DataFrame:
    """Convert scorer results to DataFrame with coordinates for each score type"""
    
    metadata = results['metadata']

    df = pd.DataFrame({
        # Metadata
        'variant_idx': [m['variant_idx'] for m in metadata],
        'chromosome': [m['chromosome'] for m in metadata],
        'position': [m['position'] for m in metadata],
        'ref_allele': [m['ref_allele'] for m in metadata],
        'alt_allele': [m['alt_allele'] for m in metadata],
        'sample_id': [m['sample_id'] for m in metadata],
        'hap_id': [m['hap_id'] for m in metadata],
        'flag': [m['flag'] for m in metadata],
        
        # Main results  
        'max_delta_score': results['max_delta_score'],
        'max_delta_coord': results['max_delta_coord'],
        'max_delta_score_type': results['max_delta_score_type'],
        'max_delta_strand': results['max_delta_strand'],
        
        # Positive strand scores and coordinates
        'pos_acceptor_gain_max': results['pos_acceptor_gain_max'],
        'pos_acceptor_gain_max_coord': results['pos_acceptor_gain_max_coord'],
        'pos_acceptor_loss_max': results['pos_acceptor_loss_max'],
        'pos_acceptor_loss_max_coord': results['pos_acceptor_loss_max_coord'],
        'pos_donor_gain_max': results['pos_donor_gain_max'],
        'pos_donor_gain_max_coord': results['pos_donor_gain_max_coord'],
        'pos_donor_loss_max': results['pos_donor_loss_max'],
        'pos_donor_loss_max_coord': results['pos_donor_loss_max_coord'],
        
        # Negative strand scores and coordinates
        'neg_acceptor_gain_max': results['neg_acceptor_gain_max'],
        'neg_acceptor_gain_max_coord': results['neg_acceptor_gain_max_coord'],
        'neg_acceptor_loss_max': results['neg_acceptor_loss_max'],
        'neg_acceptor_loss_max_coord': results['neg_acceptor_loss_max_coord'],
        'neg_donor_gain_max': results['neg_donor_gain_max'],
        'neg_donor_gain_max_coord': results['neg_donor_gain_max_coord'],
        'neg_donor_loss_max': results['neg_donor_loss_max'],
        'neg_donor_loss_max_coord': results['neg_donor_loss_max_coord'],
    })
    
    return df