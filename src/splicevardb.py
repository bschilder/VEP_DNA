import os
import requests
import pooch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import src.utils as utils

# Get your token from: https://compbio.ccia.org.au/splicevardb/
token = os.getenv('SPLICEVARDB_TOKEN')

CACHE_DIR = pooch.os_cache('splicevardb')


def get_variants(base_url=" https://compbio.ccia.org.au/splicevardb-api",
                  endpoint="/variants/",
                  page_size=0,
                  offset=0,
                  cache=CACHE_DIR,
                  verbose=True):
    if token is None:
        raise ValueError("No token found. Get your token from: https://compbio.ccia.org.au/splicevardb/ and copy into your .env file as SPLICEVARDB_TOKEN={your_token}")

    print(f"Getting variants from {url}")
    
    url = f"{base_url}{endpoint}?page_size={page_size}&offset={offset}"
    headers = {
        'accept': 'application/json',
        "Authorization": f"Bearer {token}"
    }
    response = requests.get(url, headers=headers)
    res = response.json()

    if cache:
        save_path = os.path.join(cache, f"variants_{page_size}_{offset}.json.gz")
        utils.save_json(obj=res, save_path=save_path, verbose=verbose)
    return res


def load_and_merge_data(processed_results_path, metadata_path):
    """
    Load results (our predictions) and merge with metadata (from clinvar and splicevardb)
    
    Args:
        processed_results_path: Path to results parquet file
        metadata_path: Path to metadata CSV file


    Example:
        >>> merged_df, results_df, full_metadata_df= load_and_merge_data(
            processed_results_path='../data/1000_Genomes_on_GRCh38/SpliceAI/out/chr7_results_processed.parquet', 
            metadata_path = '../data/1000_Genomes_on_GRCh38/SpliceAI/splicevardb_x_clinvar_snv.csv')
        >>> merged_df.head(10)
    """
    
    print("Loading results data...")
    
    # Load results parquet file
    if isinstance(processed_results_path, pd.DataFrame):
        results_df = processed_results_path
    else:
        results_df = pd.read_parquet(processed_results_path)
    
    # Optimize data types for memory efficiency
    results_dtypes = {
        'variant_idx': 'int16',        # max=940, fits in int16 (max 32,767)
        'chromosome': 'category',      # Categorical for efficiency
        'position': 'int32',           # Genomic positions fit in int32
        'ref_allele': 'category',      # Categorical for efficiency
        'alt_allele': 'category',      # Categorical for efficiency
        'sample': 'category',          # Keep original for reference
        'population': 'category',      # Population as category
        'superpopulation': 'category', # Super population as category
        'hap_id': 'int8',              # Binary 0/1 - perfect for int8
        'flag': 'int8',                # Small integers fit in int8
        'max_delta_score': 'float16',  # Reduced precision for memory
        'max_delta_coord': 'int32',    # Genomic coordinates
        'max_delta_score_type': 'category',
        'max_delta_strand': 'category',
        
        # Score columns as float16
        'pos_acceptor_gain_max': 'float16',
        'pos_acceptor_loss_max': 'float16', 
        'pos_donor_gain_max': 'float16',
        'pos_donor_loss_max': 'float16',
        'neg_acceptor_gain_max': 'float16',
        'neg_acceptor_loss_max': 'float16',
        'neg_donor_gain_max': 'float16',
        'neg_donor_loss_max': 'float16',
        
        # Coordinate columns as int32
        'pos_acceptor_gain_max_coord': 'int32',
        'pos_acceptor_loss_max_coord': 'int32', 
        'pos_donor_gain_max_coord': 'int32',
        'pos_donor_loss_max_coord': 'int32',
        'neg_acceptor_gain_max_coord': 'int32',
        'neg_acceptor_loss_max_coord': 'int32',
        'neg_donor_gain_max_coord': 'int32',
        'neg_donor_loss_max_coord': 'int32'
    }
    
    print("Optimizing results DataFrame dtypes...")
    original_memory = results_df.memory_usage(deep=True).sum() / (1024**2)
    print(f"Original memory: {original_memory:.2f} MB")
    
    # Apply dtype optimizations
    for col, dtype in results_dtypes.items():
        if col in results_df.columns:
            results_df[col] = results_df[col].astype(dtype)
    
    optimized_memory = results_df.memory_usage(deep=True).sum() / (1024**2)
    savings = ((original_memory - optimized_memory) / original_memory) * 100
    
    print(f"Optimized memory: {optimized_memory:.2f} MB")
    print(f"Memory savings: {savings:.1f}%")
    
    # Load metadata with optimized dtypes
    print("Loading metadata...")
    metadata_dtypes = {
        'chr': 'category', 'pos': 'int32', 'ref': 'category', 'alt': 'category', 
        'CLNSIG': 'category', 'CLNSIG_simplified': 'category',
        'method': 'category', 'spliceogenicity': 'category', 'GENEINFO': 'category',
        'clinvar_id': 'int32', 'svdb_id': 'int32', 'CLNREVSTAT': 'int8',
        'hgvs': 'string', 'gene': 'category',
        # Binary 0/1 columns
        '3ss_can': 'int8', '3ss_eprox': 'int8', '3ss_iprox': 'int8',
        '5ss_can': 'int8', '5ss_eprox': 'int8', '5ss_iprox': 'int8',
        'bp_region': 'int8', 'exon_core': 'int8'
    }
    
    try:
        metadata_df = pd.read_csv(metadata_path, dtype=metadata_dtypes)
        print("✅ Loaded metadata with optimized dtypes")
    except Exception as e:
        print(f"⚠️  Dtype optimization failed ({e}), loading normally...")
        metadata_df = pd.read_csv(metadata_path)

    # Keep full metadata for return
    full_metadata_df = metadata_df.copy()

    # Only keep essential columns for merged df
    metadata_df = metadata_df[['chr', 'pos', 'clinvar_id', 'ref', 'alt',
        '3ss_can', '3ss_eprox', '3ss_iprox', 'bp_region', '5ss_can', '5ss_eprox', '5ss_iprox', 'exon_core',
        'svdb_id', 'hgvs', 'method', 'spliceogenicity', 'gene', 'GENEINFO', 'CLNSIG',
        'CLNREVSTAT', 'CLNSIG_simplified']]
    
    # Merge with metadata
    print("Merging with metadata...")
    merged_df = results_df.merge(
        metadata_df, 
        left_on=['chromosome', 'position', 'ref_allele', 'alt_allele'],
        right_on=['chr', 'pos', 'ref', 'alt'],
        how='left'
    )
    
    # Report results
    memory_mb = merged_df.memory_usage(deep=True).sum() / (1024**2)
    successful_merges = merged_df['clinvar_id'].notna().sum()
    merge_rate = successful_merges / len(merged_df) * 100
    
    print(f"📊 Final dataset: {merged_df.shape}, Memory: {memory_mb:.1f} MB")
    print(f"🎯 Merge success: {successful_merges:,} / {len(merged_df):,} ({merge_rate:.1f}%)")
    
    return merged_df, results_df, full_metadata_df

def plot_variant_score_distributions_with_pop(df, variant_id_col='clinvar_id', score_col='max_delta_score', 
                                   threshold=0.2, n_variants=20, figsize=(20, 18),
                                   population_col='super_population', selection_method='crossing_first',
                                   use_log_scale=True, show_pop_legend=True, save_path=None,
                                   ref_pred_df=None, show_pop_proportions=True, 
                                   show_individual_pops=True, individual_pop_log_scale=True,
                                   main_to_prop_ratio=2.5, variant_height=8.0):
    """
    Optimized plot of score distributions for splice variants across haplotypes with population split bars
    
    Args:
        df: DataFrame with variant and score data
        variant_id_col: Column name for variant identifier
        score_col: Column name for splice scores
        threshold: SpliceAI threshold for clinical significance (default 0.2)
        n_variants: Number of variants to plot (will select most variable)
        figsize: Figure size tuple
        population_col: Column name for population information (default 'super_population')
        selection_method: Method to select variants ('std', 'range', 'count', 'mean', 'crossing_threshold', 'crossing_first')
        use_log_scale: Whether to use log scale for y-axis (helps with low counts)
        show_pop_legend: Whether to show population legend
        save_path: Path to save plot (optional)
        ref_pred_df: Optional DataFrame with reference predictions containing max_delta_score for each variant
        show_pop_proportions: Whether to show population proportions as connected subplot below main plot
        show_individual_pops: Whether to show individual population distribution subplots
        individual_pop_log_scale: Whether to use log scale for individual population subplots
        main_to_prop_ratio: Height ratio between main plot and proportion plot (default 2.5)
        variant_height: Total height allocated to each variant's subplot group (default 4.0 inches)
    
    Returns:
        Dictionary with analysis results

    Example:
        >>> _ = plot_variant_score_distributions_with_pop(merged_df, variant_id_col='clinvar_id', score_col='max_delta_score', 
        >>>     threshold=0.2, n_variants=10, figsize=(22, 80),
        >>>     population_col='super_population', selection_method='crossing_first',
        >>>     use_log_scale=True, show_pop_legend=True, save_path='./plots/test.pdf',
        >>>     ref_pred_df=ref_pred_file, show_pop_proportions=True, 
        >>>     show_individual_pops=True, individual_pop_log_scale=True,
        >>>     main_to_prop_ratio=2.0, variant_height=10.0)
    """
    
    print(f"=== VARIANT SCORE DISTRIBUTION ANALYSIS WITH POPULATION SPLIT ===")
    print(f"Using threshold: {threshold} | Selection: {selection_method} | Log scale: {use_log_scale}")
    print(f"Show proportions: {show_pop_proportions} | Show individual pops: {show_individual_pops}")
    if ref_pred_df is not None:
        print(f"Reference predictions provided: {len(ref_pred_df)} variants")
    
    # Check required columns exist
    required_cols = [variant_id_col, score_col, population_col]
    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        print(f"❌ Missing columns: {missing_cols}")
        return None
    
    # Get unique populations once and create color map
    unique_populations = df[population_col].dropna().unique()
    n_pops = len(unique_populations)
    print(f"Found {n_pops} unique populations: {sorted(unique_populations)}")
    
    # Efficient color mapping
    import matplotlib.cm as cm
    colors = cm.Set3(np.linspace(0, 1, n_pops))
    pop_color_map = dict(zip(unique_populations, colors))
    
    # Vectorized variant statistics calculation
    print("Calculating variant statistics...")
    variant_stats = df.groupby(variant_id_col)[score_col].agg([
        'count', 'mean', 'std', 'min', 'max'
    ]).reset_index()
    variant_stats['range'] = variant_stats['max'] - variant_stats['min']
    
    # Filter to variants with sufficient data (≥3 haplotypes)
    multi_hap_variants = variant_stats[variant_stats['count'] >= 3].copy()
    
    if len(multi_hap_variants) == 0:
        print("❌ No variants with ≥3 haplotypes found")
        return None
    
    print(f"Found {len(multi_hap_variants)} variants with ≥3 haplotypes")
    
    # Vectorized threshold crossing analysis
    if selection_method in ['crossing_threshold', 'crossing_first']:
        print("Analyzing threshold crossing...")
        # More efficient: group once and apply vectorized operations
        variant_groups = df.groupby(variant_id_col)[score_col]
        crossing_stats = variant_groups.agg(
            min_score='min',
            max_score='max',
            above_threshold=lambda x: (x >= threshold).sum(),
            below_threshold=lambda x: (x < threshold).sum()
        ).reset_index()
        crossing_stats['crosses_threshold'] = (
            (crossing_stats['min_score'] < threshold) & 
            (crossing_stats['max_score'] > threshold)
        )
        
        # Merge efficiently
        multi_hap_variants = multi_hap_variants.merge(
            crossing_stats, on=variant_id_col, how='left'
        )
    
    # Efficient variant selection
    if selection_method == 'std':
        top_variants = multi_hap_variants.nlargest(n_variants, 'std')
    elif selection_method == 'range':
        top_variants = multi_hap_variants.nlargest(n_variants, 'range')
    elif selection_method == 'count':
        top_variants = multi_hap_variants.nlargest(n_variants, 'count')
    elif selection_method == 'mean':
        top_variants = multi_hap_variants.nlargest(n_variants, 'mean')
    elif selection_method == 'crossing_threshold':
        crossing_variants = multi_hap_variants[multi_hap_variants['crosses_threshold']]
        if len(crossing_variants) == 0:
            print("❌ No threshold-crossing variants found, using highest std")
            top_variants = multi_hap_variants.nlargest(n_variants, 'std')
        else:
            top_variants = crossing_variants.nlargest(n_variants, 'std')
            print(f"Selected {len(top_variants)} threshold-crossing variants")
    elif selection_method == 'crossing_first':
        multi_hap_variants['selection_priority'] = (
            multi_hap_variants['crosses_threshold'].astype(int) * 100 + 
            multi_hap_variants['std']
        )
        top_variants = multi_hap_variants.nlargest(n_variants, 'selection_priority')
        crossing_count = top_variants['crosses_threshold'].sum()
        print(f"Selected {len(top_variants)} variants: {crossing_count} crossing, {len(top_variants) - crossing_count} others")
    else:
        raise ValueError(f"Unknown selection_method: {selection_method}")
    
    # Get additional variant info efficiently (single merge)
    info_cols = ['chr', 'pos', 'ref', 'alt', 'gene', 'spliceogenicity', 'CLNSIG',
                 '3ss_can', '3ss_eprox', '3ss_iprox', 'bp_region', 
                 '5ss_can', '5ss_eprox', '5ss_iprox', 'exon_core']
    available_info_cols = [col for col in info_cols if col in df.columns]
    
    if available_info_cols:
        variant_info = df.drop_duplicates(variant_id_col)[[variant_id_col] + available_info_cols]
        top_variants = top_variants.merge(variant_info, on=variant_id_col, how='left')
    
    # Prepare reference prediction lookup if provided
    ref_lookup_info = None
    if ref_pred_df is not None:
        print(f"Reference predictions provided: {len(ref_pred_df)} variants")
        
        # Determine which columns to use for lookup
        if all(col in df.columns for col in ['chr', 'pos', 'ref', 'alt']):
            ref_lookup_info = {
                'df_cols': ['chr', 'pos', 'ref', 'alt'],
                'ref_cols': ['chromosome', 'position', 'ref_allele', 'alt_allele'],
                'method': 'metadata_cols'
            }
        elif all(col in df.columns for col in ['chromosome', 'position', 'ref_allele', 'alt_allele']):
            ref_lookup_info = {
                'df_cols': ['chromosome', 'position', 'ref_allele', 'alt_allele'],
                'ref_cols': ['chromosome', 'position', 'ref_allele', 'alt_allele'],
                'method': 'original_cols'
            }
        else:
            print("⚠️  Warning: Cannot match with ref_pred_df - missing required columns")
            ref_lookup_info = None
    
    # Prepare data for plotting (filter once)
    selected_variant_ids = top_variants[variant_id_col].tolist()
    plot_data = df[df[variant_id_col].isin(selected_variant_ids)].copy()
    
    # Pre-compute shared bins
    shared_bins = np.linspace(0, 1, 71)  # 70 bins
    
    # Create plots with dynamic subplot structure
    n_cols = 4
    n_rows = int(np.ceil(len(top_variants) / n_cols))
    n_plot_variants = len(top_variants)  # Plot all variants
    
    # Calculate subplot layout based on options
    subplot_rows_per_variant = 1  # Main plot
    if show_pop_proportions:
        subplot_rows_per_variant += 1
    if show_individual_pops:
        subplot_rows_per_variant += n_pops
    
    # Create figure with proper size
    total_height = variant_height * n_rows  # Use variant_height parameter
    fig = plt.figure(figsize=(figsize[0], total_height))
    
    # Create subplot structure
    if show_pop_proportions or show_individual_pops:
        # Use GridSpec for connected subplots
        import matplotlib.gridspec as gridspec
        
        # Calculate total number of subplot rows needed
        total_subplot_rows = n_rows * subplot_rows_per_variant
        
        # Create height ratios
        height_ratios = []
        for row in range(n_rows):
            height_ratios.append(main_to_prop_ratio)  # Main plot (customizable ratio)
            if show_pop_proportions:
                height_ratios.append(1)  # Proportion plot
            if show_individual_pops:
                height_ratios.extend([1] * n_pops)  # Individual population plots
        
        gs = gridspec.GridSpec(total_subplot_rows, n_cols, 
                              height_ratios=height_ratios, 
                              hspace=0.02,  # Minimal spacing for connected plots
                              wspace=0.25,  # Reduced horizontal spacing
                              top=0.95,     # Leave space at top
                              bottom=0.05,  # Leave space at bottom
                              figure=fig)
        
        axes = []
        
        # Create axes for each variant
        for variant_idx in range(len(top_variants)):
            variant_row = variant_idx // n_cols  # Which row of variants (0, 1, 2, ...)
            variant_col = variant_idx % n_cols   # Which column (0, 1, 2, 3)
            
            # Calculate the GridSpec row for this variant's main plot
            gs_row = variant_row * subplot_rows_per_variant
            
            # Main plot
            main_ax = fig.add_subplot(gs[gs_row, variant_col])
            axes.append(main_ax)
            
    else:
        # Simple layout for main plots only
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = np.atleast_1d(axes).flatten()
        gs = None
    
    # Plot each variant (all variants, not just first row)
    matched_ref_count = 0
    variant_plot_data = []  # Store data for additional subplots
    
    for i, (_, variant_row) in enumerate(top_variants.iterrows()):
        if i >= len(axes):
            break
            
        var_id = variant_row[variant_id_col]
        var_data = plot_data[plot_data[variant_id_col] == var_id]
        
        ax = axes[i]
        
        # Look up reference score for this specific variant
        ref_score = None
        if ref_lookup_info is not None:
            # Get variant coordinates from current variant_row
            try:
                df_values = [variant_row[col] for col in ref_lookup_info['df_cols']]
                ref_values = ref_lookup_info['ref_cols']
                
                # Find matching row in ref_pred_df
                mask = True
                for ref_col, df_val in zip(ref_values, df_values):
                    mask = mask & (ref_pred_df[ref_col] == df_val)
                
                matching_rows = ref_pred_df[mask]
                if len(matching_rows) > 0:
                    ref_score = matching_rows['max_delta_score'].iloc[0]
                    matched_ref_count += 1
                    
            except (KeyError, IndexError) as e:
                # Silently skip if variant info not available
                pass
        
        # Efficient population-wise data preparation
        pop_scores_list = []
        pop_colors = []
        pop_names = []
        pop_data_dict = {}
        
        for pop in unique_populations:
            pop_scores = var_data[var_data[population_col] == pop][score_col].values
            if len(pop_scores) > 0:
                pop_scores_list.append(pop_scores)
                pop_colors.append(pop_color_map[pop])
                pop_names.append(pop)
                pop_data_dict[pop] = pop_scores
        
        # Store data for additional subplots
        variant_plot_data.append({
            'var_id': var_id,
            'variant_row': variant_row,
            'pop_data_dict': pop_data_dict,
            'pop_names': pop_names,
            'pop_colors': pop_colors,
            'ref_score': ref_score
        })
        
        # Create main histogram
        if pop_scores_list:
            counts, bins, patches = ax.hist(
                pop_scores_list, 
                bins=shared_bins, 
                stacked=True,
                density=False,
                alpha=0.8, 
                edgecolor='black', 
                linewidth=0.5,
                color=pop_colors,
                label=pop_names
            )
            
            # Handle log scale
            if use_log_scale:
                ax.set_yscale('log')
                if len(counts) > 0:
                    # Calculate max from stacked bars
                    max_count = max([sum(count_vals) for count_vals in zip(*counts)]) if len(counts) > 1 else max(counts[0])
                    ax.set_ylim(0.5, max_count * 2)
        
        # Formatting
        ax.set_ylabel('Frequency', fontsize=8)
        
        # Add threshold line
        ax.axvline(threshold, color='black', linestyle='--', linewidth=1.4, alpha=0.8)
        
        # Add reference prediction line if available
        if ref_score is not None and pd.notna(ref_score):
            ax.axvline(ref_score, color='red', linestyle='--', linewidth=1.4, alpha=0.9, label='Ref')
            
            # Add reference score text
            if i == 0:  # Only add text label to first plot to avoid clutter
                ax.text(ref_score + 0.01, ax.get_ylim()[1] * 0.7, f'Ref\n({ref_score:.3f})', 
                       fontsize=7, color='red', ha='left')
        
        ax.set_xlim(0, 1)
        
        # Legend
        if show_pop_legend and pop_names:
            legend = ax.legend(fontsize=7, loc='upper right', bbox_to_anchor=(0.98, 0.98), 
                              frameon=True, fancybox=True, shadow=True, framealpha=0.95)
            legend.get_frame().set_linewidth(0.5)
        
        # Create title with correct format (like your reference)
        title_parts = []
        
        # First line: chr:pos ref>alt
        if 'chr' in variant_row and 'pos' in variant_row and 'ref' in variant_row and 'alt' in variant_row:
            title_parts.append(f"{variant_row['chr']}:{variant_row['pos']} {variant_row['ref']}>{variant_row['alt']}")
        
        # Second line: gene | genomic_region
        gene_info = []
        if 'gene' in variant_row and pd.notna(variant_row['gene']):
            gene_info.append(str(variant_row['gene']))
        
        # Add genomic region information - show column names where value = 1
        region_columns = ['3ss_can', '3ss_eprox', '3ss_iprox', 'bp_region', 
                         '5ss_can', '5ss_eprox', '5ss_iprox', 'exon_core']
        
        active_regions = []
        for region_col in region_columns:
            if region_col in variant_row:
                try:
                    # Handle different data types that might represent 1
                    value = variant_row[region_col]
                    if pd.notna(value) and (value == 1 or value == 1.0 or value == '1' or value == True):
                        active_regions.append(region_col)
                except (KeyError, TypeError):
                    continue
        
        if active_regions:
            gene_info.append(' | '.join(active_regions))
        else:
            # If all region columns are 0 or missing, label as deep_intronic
            has_region_data = any(region_col in variant_row for region_col in region_columns)
            if has_region_data:  # Only add deep_intronic if we have region data but all are 0
                gene_info.append('deep_intronic')
            
        if gene_info:
            title_parts.append(' | '.join(gene_info))
        
        # Third line: SpliceVarDB info
        if 'spliceogenicity' in variant_row and pd.notna(variant_row['spliceogenicity']):
            try:
                splice_val = float(variant_row['spliceogenicity'])
                splice_category = 'Splice-altering'  # You might want to determine this based on value
                title_parts.append(f"SpliceVarDB: {splice_category}")
            except (ValueError, TypeError):
                title_parts.append(f"SpliceVarDB: {variant_row['spliceogenicity']}")
        
        # Fourth line: ClinVar info
        if 'CLNSIG' in variant_row and pd.notna(variant_row['CLNSIG']):
            clnsig = str(variant_row['CLNSIG'])
            title_parts.append(f"ClinVar: {clnsig}")
        
        # Create final title
        title = '\n'.join(title_parts) if title_parts else f"Variant {var_id}"
        
        ax.set_title(title, fontsize=10, pad=12, ha='center')  # Centered title with more padding
        
        # Only add x-axis label if no proportion subplot
        if not show_pop_proportions:
            ax.set_xlabel('Max Delta Score', fontsize=9)
        else:
            ax.set_xlabel('')  # Remove to connect with proportion plot
            ax.tick_params(labelbottom=False)
            
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)
        
        # Add threshold label to first plot only
        if i == 0:
            ax.text(threshold + 0.01, ax.get_ylim()[1] * 0.9, f'Threshold\n({threshold})', 
                   fontsize=7, color='black', ha='left')
            
        # Create connected proportion subplot if requested
        if show_pop_proportions and gs is not None:
            # Calculate subplot position for this variant's proportion plot
            variant_row_idx = i // n_cols
            variant_col_idx = i % n_cols
            
            # Calculate the GridSpec row for proportion plot (directly below main plot)
            prop_gs_row = variant_row_idx * subplot_rows_per_variant + 1
            
            # Create proportion subplot directly below main plot
            prop_ax = fig.add_subplot(gs[prop_gs_row, variant_col_idx])
            
            # Calculate proportions for each bin
            bin_proportions = []
            for pop in unique_populations:
                if pop in pop_data_dict:
                    pop_hist, _ = np.histogram(pop_data_dict[pop], bins=shared_bins)
                    bin_proportions.append(pop_hist)
                else:
                    bin_proportions.append(np.zeros(len(shared_bins)-1))
            
            # Convert to proportions
            bin_proportions = np.array(bin_proportions)
            total_counts = bin_proportions.sum(axis=0)
            
            # Avoid division by zero
            proportions = np.divide(bin_proportions, total_counts, 
                                  out=np.zeros_like(bin_proportions, dtype=float), 
                                  where=total_counts!=0)
            
            # Create stacked bar plot for proportions
            bin_centers = (shared_bins[:-1] + shared_bins[1:]) / 2
            bin_width = shared_bins[1] - shared_bins[0]
            
            bottom = np.zeros(len(bin_centers))
            for j, pop in enumerate(unique_populations):
                if pop in pop_data_dict:
                    prop_ax.bar(bin_centers, proportions[j], 
                              bottom=bottom, width=bin_width*0.8,
                              color=pop_color_map[pop], alpha=0.8, 
                              edgecolor='black', linewidth=0.3)
                    bottom += proportions[j]
            
            prop_ax.set_xlim(0, 1)
            prop_ax.set_ylim(0, 1)
            prop_ax.set_ylabel('Proportion', fontsize=9)  # Clearer font size
            prop_ax.tick_params(labelsize=8)
            prop_ax.grid(True, alpha=0.3)
            
            # Add x-axis label only to proportion plot
            prop_ax.set_xlabel('Max Delta Score', fontsize=9)
            
            # No vertical lines in proportion plot for cleaner appearance
        
        # Add individual population subplots if requested  
        if show_individual_pops and gs is not None:
            variant_row_idx = i // n_cols
            variant_col_idx = i % n_cols
            
            start_row_offset = 2 if show_pop_proportions else 1
            
            for pop_idx, pop in enumerate(unique_populations):
                # Calculate the GridSpec row for this population's subplot
                pop_gs_row = variant_row_idx * subplot_rows_per_variant + start_row_offset + pop_idx
                
                pop_ax = fig.add_subplot(gs[pop_gs_row, variant_col_idx])
                
                if pop in pop_data_dict and len(pop_data_dict[pop]) > 0:
                    pop_hist, bins, patches = pop_ax.hist(
                        pop_data_dict[pop], 
                        bins=shared_bins,
                        color=pop_color_map[pop], 
                        alpha=0.8,
                        edgecolor='black', 
                        linewidth=0.3
                    )
                    
                    if individual_pop_log_scale and len(pop_hist) > 0 and pop_hist.max() > 0:
                        pop_ax.set_yscale('log')
                        pop_ax.set_ylim(0.5, pop_hist.max() * 2)
                
                pop_ax.set_xlim(0, 1)
                pop_ax.set_ylabel(f'{pop}\nCount', fontsize=9)
                pop_ax.tick_params(labelsize=8)
                pop_ax.grid(True, alpha=0.3)
                
                # # Add threshold and ref lines
                # pop_ax.axvline(threshold, color='black', linestyle='--', linewidth=1.4, alpha=0.8)
                # if ref_score is not None and pd.notna(ref_score):
                #     pop_ax.axvline(ref_score, color='red', linestyle='--', linewidth=1.4, alpha=0.9)
                    
                # Only add x-label to bottom subplot
                if pop_idx == len(unique_populations) - 1:
                    pop_ax.set_xlabel('Max Delta Score', fontsize=9)
                else:
                    pop_ax.tick_params(labelbottom=False)
    
    # Hide unused axes if using simple layout
    if gs is None:
        for i in range(len(top_variants), len(axes)):
            if i < len(axes):
                axes[i].set_visible(False)
    
    # Print final matching stats
    if ref_lookup_info is not None:
        print(f"📊 Found reference predictions for {matched_ref_count}/{len(variant_plot_data)} plotted variants")
    
    # Use constrained layout instead of tight_layout for better spacing
    try:
        fig.set_constrained_layout(True)
    except:
        # Fallback to tight layout if constrained layout fails
        plt.tight_layout(pad=0.5)
    
    # Save or show
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(f"📁 Plot saved to: {save_path}")
    else:
        plt.show()
    
    return {
        'variant_stats': variant_stats,
        'top_variants': top_variants,
        'population_info': {
            'unique_populations': unique_populations,
            'population_colors': pop_color_map,
            'n_populations': n_pops
        },
        'reference_predictions': {
            'provided': ref_pred_df is not None,
            'matched_variants': matched_ref_count if ref_lookup_info is not None else 0,
            'lookup_method': ref_lookup_info['method'] if ref_lookup_info is not None else None
        }
    }
