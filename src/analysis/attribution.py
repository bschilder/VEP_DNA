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
):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from statannotations.Annotator import Annotator
    from itertools import combinations

    if annotator_kwargs is None:
        annotator_kwargs = {}

    # Prepare bar_df
    if x in ridge_df.columns:
        bar_df = ridge_df.groupby(["clinical_variant",x])[y].agg(agg_func).reset_index()
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
    **kwargs : dict, optional
        Additional keyword arguments to pass to sns.clustermap.
    """

    from matplotlib.colors import to_hex
    from matplotlib.patches import Patch
    from matplotlib.colors import LogNorm
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
        orig_matrices.append(X)
        # For each column in this matrix, assign the color for this matrix name
        col_colors.append([name_to_color[name]] * X.shape[1])

    # Concatenate matrices horizontally (columns)
    orig_concat = np.concatenate([m.values for m in orig_matrices], axis=1)
    # Build the full index and columns
    row_labels = matrices[matrix_names[0]].index
    col_labels = []
    for name, mat in zip(matrix_names, orig_matrices):
        col_labels.extend([c for c in mat.columns])

    # Concatenate col_colors
    col_colors_flat = sum(col_colors, [])

    # Create DataFrame for seaborn
    orig_concat_df = pd.DataFrame(orig_concat, index=row_labels, columns=col_labels)

    # Determine vmin and vmax for LogNorm
    # Avoid zeros for LogNorm: set minimum to smallest positive value
    if log_color_scale:
        # Masked array to ignore zeros and nans
        masked = np.ma.masked_where((orig_concat_df.values < 0) | np.isnan(orig_concat_df.values), orig_concat_df.values)
        if masked.count() == 0:
            vmin = 1e-6
            vmax = 1
        else:
            vmin = masked.min()
            vmax = masked.max()
        norm = LogNorm(vmin=max(vmin, log_eps), vmax=max(vmax, log_eps*10))
    else:
        norm = None

    # Plot the clustermap
    g = sns.clustermap(
        orig_concat_df,
        cmap=cmap,
        mask=np.isnan(orig_concat_df),
        annot=orig_concat_df.round(3),
        fmt=".3f",
        linewidths=0.5,
        linecolor="#cccccc",
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
        # Find the split index (number of columns in the first matrix)
        split_idx = orig_matrices[0].shape[1]
        # Draw a vertical white line at the split
        ax = g.ax_heatmap
        ax.axvline(split_idx, color="white", linewidth=3, zorder=10)
    if log_color_scale:
        g.ax_cbar.set_title("Log-scaled Mean Joint Effect Size")
    else:
        g.ax_cbar.set_title("Mean Joint Effect Size")
    g.ax_heatmap.set_xlabel("Clinical Variant Region")
    g.ax_heatmap.set_ylabel("WT Variant Region")
    
    g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=45, ha="right")
    g.ax_heatmap.set_yticklabels(g.ax_heatmap.get_yticklabels(), rotation=0)
    # Reduce padding around the heatmap
    g.ax_heatmap.figure.subplots_adjust(left=0.10, right=0.98, top=0.92, bottom=0.10)
    # Set the title over the entire plot (including dendrograms and heatmap)
    g.fig.suptitle(title, y=0.95, fontsize='large')

    # Add a legend for the matrix names/colors
    handles = [Patch(facecolor=name_to_color[name], label=name.replace("_", " ")) for name in matrix_names]
    g.ax_heatmap.legend(handles=handles, title="", bbox_to_anchor=(0.5, 1.2), loc='upper left') 
    
    # Set the position of the colorbar: (x, y, width, height) in figure coordinates
    g.ax_cbar.set_position((.275, 0.82, .2, 0.025))
    plt.show()
    return {'fig': g.fig, 'axes': {'heatmap': g.ax_heatmap}, 'data': orig_concat_df}
