import numpy as np
import pandas as pd
import src.utils as utils
import seaborn as sns
import matplotlib.pyplot as plt
 

def wtvariants_to_vep_linear_model(
        Xwt,
        y_vep, 
        model_type="ridge",  # "ridge" or "lasso"
        alpha=1.0,
        random_state=42, 
        model_kwargs={}, 
        add_positions=True,
        verbose=True,
    ):
    """
    Fit a Ridge or Lasso regression model to predict VEP values from wt_variant features,
    and compute input-output (wt_variant-site) interaction strengths, including directionality.

    Parameters
    ----------
    vep_df : pd.DataFrame
        DataFrame containing variant effect predictor (VEP) data.
    target : str, default="VEP"
        Name of the column in vep_df containing the target VEP values.
    haplotype_col : str, default="haplotype"
        Name of the column in vep_df identifying haplotypes.
    site_col : str, default="site"
        Name of the column in vep_df identifying sites.
    wt_variant_split : str, default="[,]|[|]"
        Regular expression used to split wt_variant strings.
    model_type : {"ridge", "lasso"}, default="ridge"
        Type of linear model to fit. "ridge" for Ridge regression, "lasso" for Lasso regression.
    alpha : float, default=1.0 (same default as sklearn)
        Regularization strength; must be a positive float. Larger values specify stronger regularization.
        For Ridge regression, this corresponds to the L2 penalty term, and for Lasso regression, to the L1 penalty term.
    random_state : int, default=42
        Random seed for reproducibility.
    pivot_table_kwargs : dict, default={}
        Additional keyword arguments to pass to the pivot table function.
    add_positions : bool, default=True
        If True, add positions to the interaction_df.
    verbose : bool, default=True
        If True, print progress.

    Returns
    -------
    interaction_df : pd.DataFrame
        DataFrame with columns ['wt_variant', 'site', 'interaction_strength', 'interaction_strength_signed', 'n_haplotypes', 'interaction_strength_weighted']
    model : fitted sklearn model
    """
    from sklearn.linear_model import Ridge, Lasso

    np.random.seed(random_state)
 
    # Remove any rows with NaNs in X or y and report how many rows were dropped
    nan_mask = (~Xwt.isna().any(axis=1)) & (~y_vep.isna().any(axis=1))
    n_dropped = (~nan_mask).sum()
    if n_dropped > 0 and verbose:
        print(f"Dropped {n_dropped} haplotypes due to NaNs in input or target matrices.")
    Xwt_clean = Xwt.loc[nan_mask]
    y_vep_clean = y_vep.loc[nan_mask]

    # Fit model
    if model_type == "ridge":
        model = Ridge(alpha=alpha, random_state=random_state, **model_kwargs)
    elif model_type == "lasso":
        model = Lasso(alpha=alpha, random_state=random_state, **model_kwargs)
    else:
        raise ValueError("model_type must be 'ridge' or 'lasso'")

    # Ridge/Lasso in sklearn supports multi-target regression (haplotype x site)
    model.fit(Xwt_clean.values, y_vep_clean.values)

    # Attribution: use both signed and absolute value of coefficients as interaction strength
    # model.coef_ shape: (n_sites, n_wt_variants)
    # We want (n_wt_variants, n_sites)
    coef_matrix_signed = model.coef_.T  # shape: (n_wt_variants, n_sites)
    coef_matrix_abs = np.abs(coef_matrix_signed)

    input_features = Xwt_clean.columns
    output_sites = y_vep_clean.columns
    interaction_df = pd.DataFrame(
        coef_matrix_abs,
        index=input_features,
        columns=output_sites
    )
    interaction_df = interaction_df.stack().reset_index()
    interaction_df.columns = ['wt_variant', 'site', 'interaction_strength']

    # Add signed interaction strength
    interaction_df_signed = pd.DataFrame(
        coef_matrix_signed,
        index=input_features,
        columns=output_sites
    ).stack().reset_index(drop=True)
    interaction_df["interaction_strength_signed"] = interaction_df_signed
    interaction_df["outlier_type"] = interaction_df["interaction_strength_signed"].apply(
        lambda x: "more pathogenic" if x < 0 else ("neutral" if x == 0 else "more benign")
    )

    # Add info about how many haplotypes each wt_variant is present in
    wt_variant_counts = Xwt_clean.sum(axis=0).to_dict()
    interaction_df['n_haplotypes'] = interaction_df['wt_variant'].map(wt_variant_counts)
    # Compute a score that gives equal weight to interaction strength and number of haplotypes
    interaction_df["interaction_strength_weighted"] = np.sqrt(
        interaction_df["interaction_strength"] * interaction_df["n_haplotypes"]
    )
    interaction_df = interaction_df.sort_values('interaction_strength', ascending=False)
    interaction_df["clinical_variant"] = interaction_df["site"]

    if add_positions: 
        interaction_df["wt_position"] = interaction_df["wt_variant"].str.split(":").str[1].str.split("-").str[0].astype(int)
        interaction_df["clinical_position"] = interaction_df["clinical_variant"].str.split(":").str[1].str.split("-").str[0].astype(int)
        interaction_df["position_distance"] = abs(interaction_df["wt_position"] - interaction_df["clinical_position"])
        
    # Turn coef_matrix_signed and coef_matrix_abs into DataFrames with correct row/col names
    coef_matrix_signed_df = pd.DataFrame(
        coef_matrix_signed,
        index=input_features,
        columns=output_sites
    )
    coef_matrix_abs_df = pd.DataFrame(
        coef_matrix_abs,
        index=input_features,
        columns=output_sites
    )

    return {"interaction_df": interaction_df, "model": model, 
            "Xwt_clean": Xwt_clean, "y_vep_clean": y_vep_clean,
            "coef_matrix_signed": coef_matrix_signed_df, 
            "coef_matrix_abs": coef_matrix_abs_df}



def plot_clinsig_interaction_strength(
    ridge_df,
    annot_df=None,
    site_col="mutant",
    agg_func="mean",
    x="clinsig",    
    y="interaction_strength",
    palette=utils.get_clinsig_palette(),
    title="Marginal Effect Size per Clinical Variant",
    xlabel="Clinical Significance",
    ylabel="Marginal Effect Size",
    # clinsig_sort_order = ["benign", "VUS", "pathogenic", ],
    figsize=(5, 5),
    text_format="star",
    show_test_name=False,
    loc='inside',
    verbose=0,
    pvalue_format_string=" ({:.2g})",
    test='Mann-Whitney',
    annotator_kwargs=None,
    xlabel_rotation=None,  # New argument for label rotation
    facet=None,  # New argument for faceting
):
    """
    Create boxplots showing interaction strength by clinical significance with optional faceting.
    
    This function generates boxplots to visualize the distribution of interaction strength
    (or other metrics) across different clinical significance categories. It supports both
    single plots and faceted plots for comparing across additional categorical variables.
    
    Parameters
    ----------
    ridge_df : pandas.DataFrame
        DataFrame containing the ridge regression results with interaction strength data.
        Must contain columns for clinical variants and the y-axis metric.
    annot_df : pandas.DataFrame, optional
        Annotation DataFrame containing clinical significance information.
        Required when x is not in ridge_df.columns.
    site_col : str, default "mutant"
        Column name in annot_df that corresponds to the site identifier.
    agg_func : str or callable, default "mean"
        Aggregation function to apply when grouping data. Can be "mean", "median", "sum", etc.
    x : str, default "clinsig"
        Column name for x-axis grouping (clinical significance categories).
    y : str, default "interaction_strength"
        Column name for y-axis values (the metric to plot).
    palette : dict, optional
        Color palette for different clinical significance categories.
        Defaults to utils.get_clinsig_palette().
    title : str, default "Marginal Effect Size per Clinical Variant"
        Plot title.
    xlabel : str, default "Clinical Significance"
        X-axis label.
    ylabel : str, default "Marginal Effect Size"
        Y-axis label.
    figsize : tuple, default (5, 5)
        Figure size as (width, height) in inches.
    text_format : str, default "star"
        Format for statistical annotation text. Options: "star", "simple", "full".
    show_test_name : bool, default False
        Whether to show the statistical test name in annotations.
    loc : str, default 'inside'
        Location for statistical annotations. Options: 'inside', 'outside'.
    verbose : int, default 0
        Verbosity level for statistical annotations.
    pvalue_format_string : str, default " ({:.2g})"
        Format string for p-value display in annotations.
    test : str, default 'Mann-Whitney'
        Statistical test to use for pairwise comparisons.
    annotator_kwargs : dict, optional
        Additional keyword arguments for the statistical annotator.
    xlabel_rotation : float, optional
        Rotation angle in degrees for x-axis labels. If None, no rotation is applied.
    facet : str, optional
        Column name to use for faceting (creating separate subplots for each unique value).
        If None, creates a single plot. If specified, creates faceted plots using seaborn's FacetGrid.
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'fig': matplotlib.figure.Figure or seaborn.FacetGrid.fig
            The figure object
        - 'ax': matplotlib.axes.Axes or numpy.ndarray of axes
            The axes object(s). For faceted plots, this is an array of axes.
        - 'data': pandas.DataFrame
            The processed data used for plotting
    
    Examples
    --------
    >>> # Basic plot
    >>> result = plot_clinsig_interaction_strength(ridge_df)
    
    >>> # With custom labels and rotation
    >>> result = plot_clinsig_interaction_strength(
    ...     ridge_df, 
    ...     title="Custom Title",
    ...     xlabel_rotation=45
    ... )
    
    >>> # Faceted plot by gene
    >>> result = plot_clinsig_interaction_strength(
    ...     ridge_df, 
    ...     facet='gene',
    ...     figsize=(12, 4)
    ... )
    
    >>> # With annotation DataFrame
    >>> result = plot_clinsig_interaction_strength(
    ...     ridge_df, 
    ...     annot_df=clinvar_df,
    ...     site_col='variant_id'
    ... )
    
    Notes
    -----
    - Clinical significance labels are automatically cleaned (underscores replaced with newlines,
      "path" expanded to "pathogenic", "vus" to "VUS")
    - Statistical annotations are added for pairwise comparisons between clinical significance groups
    - For faceted plots, the same styling is applied to all subplots
    - The function automatically handles data aggregation and clinical significance sorting
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statannotations.Annotator import Annotator
    from itertools import combinations

    if annotator_kwargs is None:
        annotator_kwargs = {}

    # Prepare bar_df
    if x in ridge_df.columns:
        groupby_cols = [x for x in ["clinical_variant",x,facet] if x is not None]
        bar_df = ridge_df.groupby(groupby_cols)[y].agg(agg_func).reset_index()
    else:
        if annot_df is None:
            raise ValueError("annot_df is required when x is not in ridge_df.columns")
        bar_df = ridge_df.groupby("clinical_variant")[y].agg(agg_func).reset_index().merge(
            annot_df[[site_col, x]].drop_duplicates().rename(columns={site_col: "clinical_variant"})
        )

    bar_df = utils.sort_by_clinsig(bar_df, clinsig_col=x)

    # Standardize clinsig labels: replace underscores with spaces, expand "path" and "likely_path"
    def clean_clinsig(clinsig):
        clinsig = clinsig.replace("_", "\n")
        if clinsig == "path":
            return "pathogenic"
        elif clinsig == "likely\npath":
            return "likely\npathogenic"
        elif clinsig == "vus":
            return "VUS"
        return clinsig

    bar_df[x] = bar_df[x].astype(str).apply(clean_clinsig)
 
    # Remap palette keys to match cleaned clinsig labels
    palette_cleaned = {}
    for k, v in palette.items():
        k_clean = k.replace("_", "\n")
        if k_clean == "path":
            k_clean = "pathogenic"
        elif k_clean == "likely\npath":
            k_clean = "likely\npathogenic"
        elif k_clean == "vus":
            k_clean = "VUS"
        palette_cleaned[k_clean] = v

    # Sort bar_df by clinsig order: "VUS", "pathogenic", "benign" 
    # bar_df[x] = pd.Categorical(bar_df[x], categories=clinsig_sort_order, ordered=True)
    # bar_df = bar_df.sort_values(by=x)
    

    # Handle faceting
    if facet is not None:
        # Create faceted plot
        g = sns.FacetGrid(bar_df, col=facet, height=figsize[1], aspect=figsize[0]/figsize[1])
        g.map_dataframe(sns.boxplot, x=x, y=y, hue=x, palette=palette_cleaned, showfliers=False)
        
        # Set title and labels for each subplot
        for ax in g.axes.flat:
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # Optionally rotate x-axis labels and adjust anchor
            if xlabel_rotation is not None:
                for label in ax.get_xticklabels():
                    label.set_rotation(xlabel_rotation)
                    # Adjust horizontal alignment based on rotation
                    if xlabel_rotation > 0:
                        label.set_ha('right')
                        label.set_va('top')
                    elif xlabel_rotation < 0:
                        label.set_ha('left')
                        label.set_va('top')
                    else:
                        label.set_ha('center')
                        label.set_va('top')
            
            # Remove the top and right spines (lines) from the plot margin
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
        
        plt.tight_layout()
        return {'fig': g.fig, 'ax': g.axes, 'data': bar_df}
    else:
        # Original non-faceted plot
        plt.figure(figsize=figsize)

        # Draw the boxplot
        ax = sns.boxplot(
            data=bar_df,
            x=x,
            y=y,
            hue=x,
            palette=palette_cleaned,
            showfliers=False
        )

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Optionally rotate x-axis labels and adjust anchor
        if xlabel_rotation is not None:
            for label in ax.get_xticklabels():
                label.set_rotation(xlabel_rotation)
                # Adjust horizontal alignment based on rotation
                if xlabel_rotation > 0:
                    label.set_ha('right')
                    label.set_va('top')
                elif xlabel_rotation < 0:
                    label.set_ha('left')
                    label.set_va('top')
                else:
                    label.set_ha('center')
                    label.set_va('top')

        # Get the order of clinsig groups as plotted
        clinsig_order = [t.get_text() for t in ax.get_xticklabels()]

        # Compute mean interaction_strength for each clinsig group
        means = bar_df.groupby(x)[y].mean()
        # Sort clinsig groups by mean interaction_strength
        sorted_clinsig = means.sort_values().index.tolist()

        # Generate all pairwise combinations, sorted by distance between means (descending)
        pairwise = list(combinations(sorted_clinsig, 2))
        pairwise_sorted = sorted(pairwise, key=lambda pair: abs(means[pair[0]] - means[pair[1]]), reverse=True)

        # Add statistical annotations
        annotator = Annotator(
            ax,
            pairs=pairwise_sorted,
            data=bar_df,
            x=x,
            y=y,
            order=clinsig_order
        )
        annotator.configure(
            test=test,
            text_format=text_format,
            loc=loc,
            verbose=verbose,
            show_test_name=show_test_name,
            pvalue_format_string=pvalue_format_string,
            **annotator_kwargs
        )
        annotator.apply_and_annotate()

        plt.tight_layout()
        # Remove the top and right spines (lines) from the plot margin
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        return {'fig': plt.gcf(), 'ax': ax, 'data': bar_df}


def plot_merged_clustermaps(
    matrices,
    log_color_scale=True,
    log_eps=1e-6,
    cmap="viridis",
    fillna=0,
    figsize=(16, 8),
    title="WT-Clinical Variant Joint Effect Size by Region",
    col_colors_palette="binary",
    cbar_kws={"orientation": "horizontal"},
    annotate_values=False,
    exclude_rows=None,
    exclude_cols=None,
    mismatched_legend_kws={"bbox_to_anchor": (0.5, 1.25), "loc": "upper left", "title": "", "frameon": False},
    cbar_title=r"$log_{10}(|\text{Mean Joint Effect}|)$",
    linewidths=0.5,
    linecolor="#cccccc",
    fmt=".3f", 
    xlabel="Clinical Variant Region",
    ylabel="WT Variant Region",
    cbar_position=(.275, 0.82, .2, 0.025),
    **kwargs
):
    """
    Plot merged clustermaps of multiple matrices side by side, with column colors indicating the matrix name.

    Parameters
    ----------
    matrices : dict of pd.DataFrame
        Dictionary of matrices to merge and plot. Keys are matrix names.
    log_color_scale : bool, default True
        If True, use logarithmic color scale for the heatmap.
    log_eps : float, default 1e-6
        Small value to add before log10 to avoid log(0).
    cmap : str, default "viridis"
        Colormap for the heatmap.
    figsize : tuple, default (16, 8)
        Figure size for the clustermap.
    title : str, default "Merged WT-Clinical Variant Interaction Matrices"
        Title for the plot.
    exclude_rows : list or set, optional
        List or set of row labels to exclude from the plot.
    exclude_cols : list or set, optional
        List or set of column labels to exclude from the plot.
    cbar_position : tuple, optional
        Position of the colorbar as (x, y, width, height) in figure coordinates.
    mismatched_legend_kws : dict, optional
        Additional keyword arguments to pass to the legend.
    cbar_title : str, optional
        Title for the colorbar.
    linewidths : float, optional
        Width of the lines separating the matrices.
    linecolor : str, optional
        Color of the lines separating the matrices.
    fmt : str, optional
        Format for the values in the heatmap.
    xlabel : str, optional
        Label for the x-axis.
    ylabel : str, optional
        Label for the y-axis.
    cbar_position : tuple, optional
        Position of the colorbar as (x, y, width, height) in figure coordinates.
    mismatched_legend_kws : dict, optional
        Additional keyword arguments to pass to the legend.
    **kwargs : dict, optional
        Additional keyword arguments to pass to sns.clustermap.
    """

    from matplotlib.colors import to_hex
    from matplotlib.patches import Patch
    from matplotlib.colors import LogNorm

    # Handle exclude_rows and exclude_cols
    exclude_rows = set(exclude_rows) if exclude_rows is not None else set()
    exclude_cols = set(exclude_cols) if exclude_cols is not None else set()

    matrix_names = list(matrices.keys())
    orig_matrices = []
    col_colors = []

    # Assign a unique color to each matrix name
    palette = sns.color_palette(col_colors_palette, n_colors=len(matrix_names))
    name_to_color = {name: to_hex(color) for name, color in zip(matrix_names, palette)}

    for name in matrix_names:
        X = matrices[name]
        if fillna is not None:
            X = X.fillna(fillna)
        # Exclude specified rows and columns
        if exclude_rows:
            X = X.loc[~X.index.isin(exclude_rows)]
        if exclude_cols:
            X = X.loc[:, ~X.columns.isin(exclude_cols)]
        orig_matrices.append(X)
        # For each column in this matrix, assign the color for this matrix name
        col_colors.append([name_to_color[name]] * X.shape[1])

    # Concatenate matrices horizontally (columns)
    if len(orig_matrices) == 0 or any(m.shape[0] == 0 for m in orig_matrices):
        raise ValueError("After excluding rows, at least one matrix has no rows left.")
    if any(m.shape[1] == 0 for m in orig_matrices):
        raise ValueError("After excluding columns, at least one matrix has no columns left.")

    orig_concat = np.concatenate([m.values for m in orig_matrices], axis=1)
    # Build the full index and columns
    row_labels = orig_matrices[0].index
    col_labels = []
    for name, mat in zip(matrix_names, orig_matrices):
        col_labels.extend([c for c in mat.columns])

    # Concatenate col_colors
    col_colors_flat = sum(col_colors, [])

    # Create DataFrame for seaborn
    orig_concat_df = pd.DataFrame(orig_concat, index=row_labels, columns=col_labels)

    # Remove any columns or rows that are in exclude_cols or exclude_rows (in case they slipped through)
    if exclude_rows:
        orig_concat_df = orig_concat_df.loc[~orig_concat_df.index.isin(exclude_rows)]
    if exclude_cols:
        orig_concat_df = orig_concat_df.loc[:, ~orig_concat_df.columns.isin(exclude_cols)]
        # Also need to update col_colors_flat to match the new columns
        # Rebuild col_colors_flat based on the columns that remain
        new_col_colors_flat = []
        col_idx = 0
        for name, mat in zip(matrix_names, orig_matrices):
            for c in mat.columns:
                if c not in exclude_cols:
                    new_col_colors_flat.append(name_to_color[name])
                col_idx += 1
        col_colors_flat = new_col_colors_flat

    # Determine vmin and vmax for LogNorm
    # Avoid zeros for LogNorm: set minimum to smallest positive value
    if log_color_scale:
        # Masked array to ignore zeros, negatives, and nans
        masked = np.ma.masked_where((orig_concat_df.values <= 0) | np.isnan(orig_concat_df.values), orig_concat_df.values)
        if masked.count() == 0:
            vmin = 1e-6
            vmax = 1
        else:
            vmin = masked.min()
            vmax = masked.max() 
        # Use actual data range for LogNorm, ensuring vmin is at least log_eps
        norm = LogNorm(vmin=max(vmin, log_eps), vmax=vmax)
    else:
        norm = None

    # Plot the clustermap
    g = sns.clustermap(
        orig_concat_df,
        cmap=cmap,
        mask=np.isnan(orig_concat_df),
        annot=orig_concat_df.round(3) if annotate_values else None,
        fmt=fmt,
        linewidths=linewidths,
        linecolor=linecolor,
        col_colors=col_colors_flat,
        xticklabels=True,
        yticklabels=True,
        figsize=figsize, 
        cbar_kws=cbar_kws,
        norm=norm,
        **kwargs
    )

    # If col_cluster is False, draw a white line splitting the two submatrices
    if not kwargs.get("col_cluster", True) and len(matrix_names) == 2:
        # Find the split index (number of columns in the first matrix, after exclusion)
        split_idx = orig_matrices[0].shape[1]
        # Adjust for excluded columns in the first matrix
        if exclude_cols:
            split_idx = sum([1 for c in orig_matrices[0].columns if c not in exclude_cols])
        # Draw a vertical white line at the split
        ax = g.ax_heatmap
        ax.axvline(split_idx, color="white", linewidth=3, zorder=10)
    if log_color_scale:
        g.ax_cbar.set_title(cbar_title)
    else:
        g.ax_cbar.set_title(cbar_title)
    g.ax_heatmap.set_xlabel(xlabel)
    g.ax_heatmap.set_ylabel(ylabel)
    
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    # Reduce padding around the heatmap
    g.ax_heatmap.figure.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10)
    # Set the title over the entire plot (including dendrograms and heatmap)
    g.fig.suptitle(title, y=0.95, fontsize='large')

    # Add a legend for the matrix names/colors
    handles = [Patch(facecolor=name_to_color[name], label=name.replace("_", " ")) for name in matrix_names]
    g.ax_heatmap.legend(handles=handles, **mismatched_legend_kws) 
    
    # Set the position of the colorbar: (x, y, width, height) in figure coordinates
    g.ax_cbar.set_position(cbar_position)
    plt.show()
    return {'fig': g.fig, 'axes': {'heatmap': g.ax_heatmap}, 'data': orig_concat_df}


def read_bed_files(dir_path,
                    sep="\t",
                    header=None, 
                    index_col=False,
                    names = [
                        "Chromosome",    # chrom
                        "Start",         # start
                        "End",           # end
                        "Name",          # name
                        "Score",         # score
                        "Strand",        # strand
                        "Type",          # type
                        "gene_name",     # gene_name
                        "tx_id", # transcript_id 
                        "exon_number",
                        "exon_id"
                    ]): 
    import glob
    import os
    import pandas as pd
    from tqdm.auto import tqdm
    bed_paths = glob.glob(dir_path)
    bed_dict = {} 
    for path in tqdm(bed_paths):
        key = os.path.basename(path).split(".")[0]
        df = pd.read_csv(path, sep=sep, header=header, names=names, index_col=index_col)  
        bed_dict[key] = df
    return bed_dict


REGION_ANNOTATION_MAP = {
    "5ss_can": "5′ss canonical",
    "5ss_iprox": "5′ss intronic proximal",
    "5ss_eprox": "5′ss exonic proximal",
    "3ss_can": "3′ss canonical",
    "3ss_iprox": "3′ss intronic proximal",
    "bp_region": "Branchpoint region",
    "3ss_eprox": "3′ss exonic proximal",
    "exon_core": "Exonic core", # (Exon body excluding 5′ss/3′ss exonic proximal)
    # "35ss_eprox": "3/5′ss exonic proximal",
    "intron_dist": "Intronic distal", 
    "intron_prox": "Intronic proximal",
    "mane_introns": "MANE introns",
    "mane_exons": "MANE exons",
    "3prime_UTR": "3′ UTR",
    "5prime_UTR": "5′ UTR",
}


def annotate_variants_with_bed(
    ridge_df, 
    bed_dict, 
    region_annotation_map=REGION_ANNOTATION_MAP, 
    variant_types=("wt", "clinical"),
    verbose=True
):
    """
    Annotate variants in ridge_df with overlap to regions in bed_dict.
    Adds columns for each region indicating overlap, and merges exon info.
    
    Parameters
    ----------
    ridge_df : pd.DataFrame
        DataFrame containing at least columns for variant IDs (e.g. 'wt_variant', 'clinical_variant').
    bed_dict : dict
        Dictionary mapping region short names to BED DataFrames.
    region_full_map : dict or None
        Mapping from region short names to full region names. If None, uses default.
    variant_types : tuple
        Tuple of variant types to annotate (default: ("wt", "clinical")).
    verbose : bool
        Whether to print progress.
        
    Returns
    -------
    pd.DataFrame
        Annotated DataFrame.
    """
    import pyranges as pr 
    import pandas as pd
    from tqdm import tqdm
 
    df = ridge_df.copy()

    exon_info_all = {}
    for variant_type in variant_types:
        if verbose:
            print(variant_type)
        id_col = f"{variant_type}_variant"
        exon_info = []
        
        # Reset variant coordinates
        for col in ['Chromosome', 'Start', 'End']:
            if col in df.columns:
                df.drop(columns=[col], inplace=True)
        df.insert(0, "Chromosome", df[id_col].str.split(":").str[0])
        df.insert(1, "Start", df[id_col].str.split(":").str[1].str.split("-").str[0].astype(int))
        df.insert(2, "End", df[id_col].str.split(":").str[1].str.split("-").str[1].str.split("_").str[0].astype(int))

        for key, bed_df in tqdm(bed_dict.items(), disable=not verbose): 
            key_full = region_annotation_map[key]
            overlap = pr.PyRanges(df).overlap(pr.PyRanges(bed_df))
            if len(overlap) > 0:
                exon_tmp = pr.PyRanges(df).join(pr.PyRanges(bed_df), apply_strand_suffix=False).df[[id_col,"exon_number","exon_id"]].drop_duplicates()
                exon_tmp.rename(columns={col: f"{variant_type}_{col}" for col in exon_tmp.columns[1:]}, inplace=True)
                exon_info.append(exon_tmp)

            if len(overlap) > 0:
                df[f"{variant_type}_{key_full}"] = df[id_col].isin(overlap.df[id_col])
            else:
                df[f"{variant_type}_{key_full}"] = False  

        # Merge in exon info
        if exon_info:
            exon_info_all[variant_type] = pd.concat(exon_info, axis=0).drop_duplicates().dropna()
            df = df.merge(pd.concat(exon_info, axis=0).drop_duplicates().dropna(), on=id_col, how="left")
    
    df["exon_number_distance"] = df["wt_exon_number"] - df["clinical_exon_number"]
    df["exon_number_distance_abs"] = df["exon_number_distance"].abs()
    return df

def plot_splicing_tracks(
    bed_dict,
    ridge_df,
    chrom,
    gene_min_pos,
    gene_max_pos,
    interaction_threshold=0.01,
    exclude_keys=None,
    fig_track_height=0.3,
    feature_track_ratio=2,
    region_name=None,
):
    """
    Visualize splicing region annotation tracks and mean interaction scores using pygenomeviz.

    Parameters
    ----------
    bed_dict : dict
        Dictionary of region_name -> BED dataframe.
    ridge_df : pd.DataFrame
        DataFrame with columns including 'wt_position', 'clinical_position', 'interaction_strength'.
    chrom : str
        Chromosome name (e.g., 'chr17').
    gene_min_pos : int
        Start coordinate of the region.
    gene_max_pos : int
        End coordinate of the region.
    interaction_threshold : float, optional
        Minimum interaction_strength to include in mean tracks.
    exclude_keys : list, optional
        List of region keys to exclude from annotation tracks.
    fig_track_height : float, optional
        Height of each track in the figure.
    feature_track_ratio : float, optional
        Ratio for feature track height.
    region_name : str, optional
        Name for the region segment (default: "{chrom}_region").
    Returns
    -------
    fig : matplotlib.figure.Figure
        The generated figure.
    gv : GenomeViz
        The GenomeViz object (for further customization).
    """
    from pygenomeviz import GenomeViz
    import matplotlib.pyplot as plt
 
    target_chrom = chrom
    target_start = int(gene_min_pos)
    target_end = int(gene_max_pos)
    region_name = region_name or f"{target_chrom}_region"
    
    # Validate coordinates
    if target_start >= target_end:
        raise ValueError(f"gene_min_pos ({target_start}) must be less than gene_max_pos ({target_end})")

    # Assign a color to each annotation track using a colormap
    cmap = plt.get_cmap('tab20')
    track_colors = {key: cmap(i) for i, key in enumerate(bed_dict.keys())}

    # Sort bed_dict so that any keys with "exon" come last, and "intron_dist" comes just before exons
    def custom_sort_key(key):
        if "exon" in key:
            return (2, key)
        elif key == "intron_dist":
            return (1, key)
        else:
            return (0, key)
    bed_dict_sorted = dict(sorted(bed_dict.items(), key=lambda x: custom_sort_key(x[0])))

    # Create GenomeViz object
    gv = GenomeViz(fig_track_height=fig_track_height, feature_track_ratio=feature_track_ratio)
    gv.set_scale_xticks(start=target_start, unit="Kb")

    # Add annotation tracks
    for i, (key, bed_df) in enumerate(bed_dict_sorted.items()):
        if key in exclude_keys:
            continue
        region_segments = {region_name: (target_start, target_end)}
        track = gv.add_feature_track(key, region_segments)
        overlap = bed_df[
            (bed_df["Chromosome"] == target_chrom) &
            (bed_df["Start"] < target_end) &
            (bed_df["End"] > target_start)
        ]
        color = track_colors[key]
        if key in ["exon_core", "mane_exons"]:
            exon_regions = [(int(row["Start"]), int(row["End"])) for _, row in overlap.iterrows()]
            if exon_regions:
                track.add_exon_feature(
                    exon_regions,
                    strand=1,
                    plotstyle="box",
                    label="",
                    text_kws=dict(rotation=0, hpos="center"),
                )
            continue
        for _, row in overlap.iterrows():
            strand = 1
            if "Strand" in row and row["Strand"] in ("+", "-"):
                strand = 1 if row["Strand"] == "+" else -1
            track.add_feature(
                int(row["Start"]), int(row["End"]), strand,
                label="",
                plotstyle="box",
                facecolor=color,
                edgecolor=color,
                lw=1.0,
                alpha=0.8
            )

    # Add mean interaction tracks (clinical and wt)
    def add_mean_interaction_track(
        df, pos_col, track_label, region_name, start, end, gv, cmap, norm
    ):
        track = gv.add_feature_track(track_label, {region_name: (start, end)})
        for _, row in df.iterrows():
            pos = int(row[pos_col])
            box_start = pos - 5
            box_end = pos + 5
            color = cmap(norm(row["interaction_strength"]))
            alpha = 0.1 + 0.9 * norm(row["interaction_strength"])
            track.add_feature(
                box_start, box_end, 1,
                label="",
                plotstyle="box",
                facecolor=color,
                edgecolor=color,
                lw=1.0,
                alpha=alpha
            )
        return track

    # Clinical mean interaction
    mean_interaction = (
        ridge_df.loc[ridge_df["interaction_strength"] >= interaction_threshold]
        .groupby("clinical_position")
        .agg({"interaction_strength": "mean"})
        .reset_index()
    )
    cmap_interaction = plt.get_cmap("viridis")
    norm = plt.Normalize(mean_interaction["interaction_strength"].min(), mean_interaction["interaction_strength"].max())
    add_mean_interaction_track(
        mean_interaction, "clinical_position", "Clinical mean interaction",
        region_name, target_start, target_end, gv, cmap_interaction, norm
    )

    # WT mean interaction
    mean_interaction_wt = (
        ridge_df.loc[ridge_df["interaction_strength"] >= interaction_threshold]
        .groupby("wt_position")
        .agg({"interaction_strength": "mean"})
        .reset_index()
    )
    cmap_interaction_wt = plt.get_cmap("viridis")
    norm_wt = plt.Normalize(mean_interaction_wt["interaction_strength"].min(), mean_interaction_wt["interaction_strength"].max())
    add_mean_interaction_track(
        mean_interaction_wt, "wt_position", "WT mean interaction",
        region_name, target_start, target_end, gv, cmap_interaction_wt, norm_wt
    )

    fig = gv.plotfig()
    return fig, gv


def ridge_df_annot_to_matrices(
    ridge_df_annot, 
    region_annotation_map=REGION_ANNOTATION_MAP,
    check_exon_match=True,
    exon_distances = []
):
    """
    Compute variant sensitization matrices for WT and clinical variant region overlaps.
    If check_exon_match is True, returns separate matrices for exon-matched and exon-mismatched cases.
    Otherwise, returns a single overlap-only matrix.

    Parameters
    ----------
    ridge_df_annot : pd.DataFrame
        Annotated DataFrame with region overlap columns and exon_id columns.
    bed_dict : dict
        Dictionary of region short names to BED DataFrames (for keys).
    region_annotation_map : dict
        Mapping from region short names to full region names.
    check_exon_match : bool, default True
        Whether to compute exon-matched/mismatched matrices.
    exon_distances : list, default []
        List of exon distances to compute matrices for.

    Returns
    -------
    dict or pd.DataFrame
        If check_exon_match is True, returns dict of matrices for each case.
        Otherwise, returns a single DataFrame.
    """
    import pandas as pd
    import numpy as np
    from tqdm import tqdm
    wt_keys = ["wt_" + v for v in region_annotation_map.values()]
    clinical_keys = ["clinical_" + v for v in region_annotation_map.values()]

    if not check_exon_match:
        # Standard matrix: overlap only
        matrix = pd.DataFrame(
            np.nan,
            index=wt_keys,
            columns=clinical_keys
        )
        for wt_col in tqdm(wt_keys):
            for clinical_col in clinical_keys:
                # Check if both columns exist in ridge_df_annot
                if wt_col in ridge_df_annot.columns and clinical_col in ridge_df_annot.columns:
                    mask = ridge_df_annot[wt_col] & ridge_df_annot[clinical_col]
                    if mask.any():
                        matrix.loc[wt_col, clinical_col] = ridge_df_annot.loc[mask, "interaction_strength"].mean()
                    else:
                        matrix.loc[wt_col, clinical_col] = np.nan
                else:
                    # If either column is missing, set as NaN
                    matrix.loc[wt_col, clinical_col] = np.nan
        # Optionally, rename the index/columns to just the key names (remove prefix)
        matrix.index = [c.replace("wt_", "") for c in matrix.index]
        matrix.columns = [c.replace("clinical_", "") for c in matrix.columns]
        return matrix
    else:
        # When check_exon_match is True, create separate matrices for:
        # 1. Overlap & exon_id match
        # 2. Overlap & exon_id mismatch

        matrices = {} 
        cases = [
            ("exon_matched", lambda m: m & (ridge_df_annot["wt_exon_id"] == ridge_df_annot["clinical_exon_id"])),
            ("exon_mismatched", lambda m: m & (ridge_df_annot["wt_exon_id"].notna()) & (ridge_df_annot["clinical_exon_id"].notna()) & (ridge_df_annot["wt_exon_id"] != ridge_df_annot["clinical_exon_id"])),
        ]
        # Programmatically add exon_distance_n cases
        for dist in exon_distances: 
            cases.append((
                f"exon_distance_{dist}",
                lambda m, d=dist: m & (ridge_df_annot["exon_number_distance_abs"].notna()) & (ridge_df_annot["exon_number_distance_abs"] == d)
            ))
        for case_name, case_mask_func in tqdm(cases):
            mat = pd.DataFrame(
                np.nan,
                index=wt_keys,
                columns=clinical_keys
            )
            for wt_col in tqdm(wt_keys, desc=f"Computing matrices for case: {case_name}"):
                for clinical_col in clinical_keys:
                    # Check if both columns exist in ridge_df_annot
                    if wt_col in ridge_df_annot.columns and clinical_col in ridge_df_annot.columns:
                        overlap_mask = ridge_df_annot[wt_col] & ridge_df_annot[clinical_col]
                        mask = case_mask_func(overlap_mask)
                        if mask.any():
                            mat.loc[wt_col, clinical_col] = ridge_df_annot.loc[mask, "interaction_strength"].mean()
                        else:
                            mat.loc[wt_col, clinical_col] = np.nan
                    else:
                        # If either column is missing, set as NaN
                        mat.loc[wt_col, clinical_col] = np.nan
            # Optionally, rename the index/columns to just the key names (remove prefix)
            mat.index = [c.replace("wt_", "") for c in mat.index]
            mat.columns = [c.replace("clinical_", "") for c in mat.columns]
            mat.sort_index(axis=0, inplace=True)
            mat.sort_index(axis=1, inplace=True)
            matrices[case_name] = mat
        return matrices





def plot_paired_violin_with_stats(
    matrices,
    save_path=None,
    fig_save_kwargs=utils.FIG_SAVE_KWARGS,
    show=True,
    hue_col="matrix",
    group_color_palette="binary",
    title="Paired Distributions of Joint Effect Sizes",
    ylabel="Mean Joint Effect Size",
    figsize=(4, 6),
    point_size=3,
    line_alpha=0.2,
    line_color="gray",
    annotator_kwargs={"test":"t-test_paired", 
                     "text_format":"full", 
                     "loc":"inside", 
                     "verbose":2, 
                     "pvalue_format_string":"{:.3f}"},
):
    """
    Plot paired violin plots for the first two matrices in the matrices dict,
    perform paired t-test and Wilcoxon signed-rank test, and annotate the plot.

    Parameters
    ----------
    matrices : dict of pd.DataFrame
        Dictionary of DataFrames of the same shape, e.g. {"exon_matched": df1, "exon_mismatched": df2}
    save_path : str
        Path to save the figure.
    fig_save_kwargs : dict or None
        Additional kwargs for plt.savefig.
    show : bool
        Whether to call plt.show().
    title : str
        Title for the plot.
    ylabel : str
        Y-axis label.
    Returns
    -------
    dict
        Dictionary with test statistics and p-values.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns
    import pandas as pd
    import numpy as np
    from scipy.stats import ttest_rel, wilcoxon
    from statannotations.Annotator import Annotator

    # Support any number of matrices, as long as they are all the same shape
    matrix_names = list(matrices.keys())
    mats = [matrices[name] for name in matrix_names]
    # Flatten all matrices
    flattened = [mat.values.flatten() for mat in mats]
    # Align by index/column order and mask out any pair where at least one is nan
    # Create an array where each row is one "pair", each column is one matrix
    stacked = np.vstack(flattened)
    # Now, only keep rows with no NaNs anywhere (all matrices must have a value for the pair)
    mask = ~np.isnan(stacked).any(axis=0)
    filtered = [arr[mask] for arr in flattened]
    n_pairs = filtered[0].shape[0]

    # Build the melted plotting DataFrame
    df_plot = pd.DataFrame({
        'value': np.concatenate(filtered),
        'matrix': np.concatenate([
            [matrix_names[i]] * n_pairs for i in range(len(matrix_names))
        ]),
        'pair_id': np.tile(np.arange(n_pairs), len(matrix_names)),
    })

    # Run paired statistical tests for all pairs
    paired_stats = {}
    for i in range(len(matrix_names)):
        for j in range(i + 1, len(matrix_names)):
            name_i = matrix_names[i]
            name_j = matrix_names[j]
            vals_i = filtered[i]
            vals_j = filtered[j]
            # t-test
            try:
                t_stat, t_pval = ttest_rel(vals_i, vals_j)
            except Exception as e:
                t_stat, t_pval = np.nan, np.nan
            # Wilcoxon signed-rank
            try:
                w_stat, w_pval = wilcoxon(vals_i, vals_j)
            except Exception as e:
                w_stat, w_pval = np.nan, np.nan
            paired_stats[(name_i, name_j)] = {
                "t_stat": t_stat,
                "t_pval": t_pval,
                "w_stat": w_stat,
                "w_pval": w_pval
            }

    # Make the violin plot for all matrices
    g = plt.figure(figsize=figsize)
    ax = sns.violinplot(
        x='matrix',
        y='value',
        data=df_plot,
        inner=None,
        palette=group_color_palette,
        hue=hue_col,
        cut=0
    )
    # Overlay the data points
    sns.stripplot(x='matrix', y='value', data=df_plot, jitter=True, color='k', alpha=0.5, zorder=2, size=point_size)

    # Draw lines between matched pairs, connecting each pair across all groups
    # To avoid overplotting, fade the lines more as there are more matrices 
    matrix_columns = {name: i for i, name in enumerate(matrix_names)}
    for pair_idx in range(n_pairs):
        # Get y values for this pair in each group
        yvals = [filtered[i][pair_idx] for i in range(len(matrix_names))]
        # X positions are integers (0, 1, ..., k-1)
        plt.plot(
            np.arange(len(matrix_names)), yvals,
            color=line_color, alpha=line_alpha, zorder=1
        )

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel("")
    sns.despine(ax=ax, top=True, right=True)
    # Replace underscores with spaces in x-axis tick labels
    ax.set_xticklabels([label.get_text().replace("_", " ") for label in ax.get_xticklabels()])

    # Use statannotations for pairwise annotation
    pairs = []
    for i in range(len(matrix_names)):
        for j in range(i + 1, len(matrix_names)):
            pairs.append((matrix_names[i], matrix_names[j]))
    annotator = Annotator(ax, pairs, data=df_plot, x='matrix', y='value', order=matrix_names)
    annotator.configure(**annotator_kwargs)
    annotator.apply_and_annotate()

    plt.tight_layout()
    if fig_save_kwargs is not None and save_path is not None:
        plt.savefig(save_path, **fig_save_kwargs)
    if show:
        plt.show()
    plt.close()

    return {
        "fig": g.figure,
        "axes": g.axes,
        "data": df_plot,
        "t_stat": t_stat,
        "t_pval": t_pval,
        "w_stat": w_stat,
        "w_pval": w_pval,
        "matrix_names": matrix_names
    }