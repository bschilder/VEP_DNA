import os
import glob
from re import T
import pandas as pd
import numpy as np
from tqdm import tqdm
import pooch
import seaborn as sns
import matplotlib.pyplot as plt


import src.utils as utils

# Categories:
# - pathogenicity
# - splicing 
# - protein
# - conservation
# - MPRA
# - population genetics
# - regulatory
# - gene constraint
# - clinical
# - other

ANNOT_DICT = {
    # Pathogenicity predictors
    "CADD_PHRED": "pathogenicity",
    "CADD_RAW": "pathogenicity",
    "ClinPred": "pathogenicity", 
    "BayesDel_addAF_score": "pathogenicity",
    "BayesDel_noAF_score": "pathogenicity",
    "DANN_score": "pathogenicity",
    "Eigen-PC-phred_coding": "pathogenicity",
    "Eigen-PC-raw_coding": "pathogenicity",
    "Eigen-phred_coding": "pathogenicity",
    "Eigen-raw_coding": "pathogenicity",
    "MPC_score": "pathogenicity",
    "MetaLR_score": "pathogenicity",
    "MetaRNN_score": "pathogenicity",
    "MetaSVM_score": "pathogenicity",
    "MutationTaster_converted_rankscore": "pathogenicity",
    "Reliability_index": "pathogenicity",

    # Splicing
    "rf_score": "splicing",
    "ada_score": "splicing",
    "MaxEntScan_ref": "splicing",
    "MaxEntScan_alt": "splicing",
    "MaxEntScan_diff": "splicing",
    "SpliceAI_pred_DP_AG": "splicing",
    "SpliceAI_pred_DP_AL": "splicing",
    "SpliceAI_pred_DP_DG": "splicing",
    "SpliceAI_pred_DP_DL": "splicing",
    "SpliceAI_pred_DS_AG": "splicing",
    "SpliceAI_pred_DS_AL": "splicing",
    "SpliceAI_pred_DS_DG": "splicing",
    "SpliceAI_pred_DS_DL": "splicing",

    # Protein 

    "gMVP_score": "protein",
    "fathmm-XF_coding_score": "protein",
    "PROVEAN_converted_rankscore": "protein",
    "PROVEAN_pred": "protein",
    "PROVEAN_score": "protein",
    "MutationAssessor_score": "protein",
    "MVP_score": "protein",
    "LIST-S2_score": "protein",
    "VARITY_ER_LOO_score": "protein",
    "VARITY_ER_score": "protein",
    "VARITY_R_LOO_score": "protein",
    "VARITY_R_score": "protein",
    "DEOGEN2_score": "protein",
    "PrimateAI_pred": "protein",
    "PrimateAI_score": "protein",
    "MutFormer_score": "protein",
    "ESM1b_score": "protein",
    "REVEL": "protein",
    "EVE_SCORE": "protein",
    "am_pathogenicity": "protein",
    "SIFT_score": "protein",
    "PolyPhen_score": "protein",
    "BLOSUM62": "protein",
    "MOTIF_SCORE_CHANGE": "protein",
    "mutfunc_exp": "protein",
    "mutfunc_int": "protein",
    "mutfunc_mod": "protein",
    "mutfunc_motif": "protein",

    # Conservation
    "GERP++_NR": "conservation",
    "GERP++_RS": "conservation",
    "GERP_91_mammals": "conservation",
    "phastCons100way_vertebrate": "conservation",
    "phastCons17way_primate": "conservation",
    "phastCons470way_mammalian": "conservation",
    "phyloP100way_vertebrate": "conservation",
    "phyloP17way_primate": "conservation",
    "phyloP470way_mammalian": "conservation",

    # MPRA/experimental
    "MaveDB_score_mean": "MPRA",
    "MaveDB_score_abs_mean": "MPRA",
    # "MaveDB_score_min": "MPRA",
    # "MaveDB_score_max": "MPRA",

    # Regulatory
    "OpenTargets_l2g": "regulatory",
    "Enformer_SAD": "regulatory",
    "Enformer_SAR": "regulatory",

    # Population genetics/allele frequency
    "AF": "population genetics",
    "AFR_AF": "population genetics",
    "AMR_AF": "population genetics",
    "EAS_AF": "population genetics",
    "EUR_AF": "population genetics",
    "SAS_AF": "population genetics",
    "gnomADe_AF": "population genetics",
    "gnomADe_AFR_AF": "population genetics",
    "gnomADe_AMR_AF": "population genetics",
    "gnomADe_ASJ_AF": "population genetics",
    "gnomADe_EAS_AF": "population genetics",
    "gnomADe_FIN_AF": "population genetics",
    "gnomADe_MID_AF": "population genetics",
    "gnomADe_NFE_AF": "population genetics",
    "gnomADe_REMAINING_AF": "population genetics",
    "gnomADe_SAS_AF": "population genetics",
    "gnomADg_AF": "population genetics",
    "gnomADg_AFR_AF": "population genetics",
    "gnomADg_AMI_AF": "population genetics",
    "gnomADg_AMR_AF": "population genetics",
    "gnomADg_ASJ_AF": "population genetics",
    "gnomADg_EAS_AF": "population genetics",
    "gnomADg_FIN_AF": "population genetics",
    "gnomADg_MID_AF": "population genetics",
    "gnomADg_NFE_AF": "population genetics",
    "gnomADg_REMAINING_AF": "population genetics",
    "gnomADg_SAS_AF": "population genetics",
    "AF_TGP": "population genetics",
    "AllOfUs_gvs_all_af": "population genetics",
    "AllOfUs_gvs_max_af": "population genetics",
    "AllOfUs_gvs_afr_af": "population genetics",
    "AllOfUs_gvs_amr_af": "population genetics",
    "AllOfUs_gvs_eas_af": "population genetics",
    "AllOfUs_gvs_eur_af": "population genetics",
    "AllOfUs_gvs_mid_af": "population genetics",
    "AllOfUs_gvs_oth_af": "population genetics",
    "AllOfUs_gvs_sas_af": "population genetics",
    "1000Gp3_AF": "population genetics",
    "1000Gp3_AFR_AF": "population genetics",
    "1000Gp3_AMR_AF": "population genetics",
    "1000Gp3_EAS_AF": "population genetics",
    "1000Gp3_EUR_AF": "population genetics",
    "1000Gp3_SAS_AF": "population genetics",
    "ALFA_African_AF": "population genetics",
    "ALFA_African_American_AF": "population genetics",
    "ALFA_African_Others_AF": "population genetics",
    "ALFA_Asian_AF": "population genetics",
    "ALFA_East_Asian_AF": "population genetics",
    "ALFA_European_AF": "population genetics",
    "ALFA_Latin_American_1_AF": "population genetics",
    "ALFA_Latin_American_2_AF": "population genetics",
    "ALFA_Other_AF": "population genetics",
    "ALFA_Other_Asian_AF": "population genetics",
    "ALFA_South_Asian_AF": "population genetics",
    "ALFA_Total_AF": "population genetics",
    "TOPMed_frz8_AF": "population genetics",

    # Gene constraint
    "LOEUF": "gene constraint",
    "pHaplo": "gene constraint",
    "pTriplo": "gene constraint",
    "bStatistic": "gene constraint",
    "bStatistic_converted_rankscore": "gene constraint", 

    # Clinical/phenotype
    "Geno2MP_HPO_count": "clinical",

    # Other
    # (add more as needed)
}

ANNOT_COLS = list(ANNOT_DICT.keys())

def filter_annotations(
    df,
    df_filters={"Location": "Location"},
    # cache_search=os.path.join(pooch.os_cache("pooch"), "clinvar_by_chrom", "clinvarVEP*"),
    cache_search=os.path.join(os.path.expanduser("~/projects/data/ensemblVEP"), "clinvarVEP*"),
    cached_file=None,
    force=False
):
    """
    Filter and annotate [Ensembl VEP (Variant Effect Predictor)](https://useast.ensembl.org/Tools/VEP) 
    annotation files for ClinVar variants.

    This function loads, filters, and processes VEP annotation files for ClinVar variants,
    optionally caching the merged and filtered results for future use. It also extracts
    and computes additional annotation columns such as SIFT/PolyPhen scores and MaveDB statistics.

    Args:
        df (pd.DataFrame): DataFrame containing the variants of interest. Used to filter the VEP annotations.
        df_filters (dict, optional): Dictionary mapping column names in `df` to column names in the VEP annotation files
            for filtering. Default is {"Location": "Location"}.
        cached_file (str, optional): Path to the cached, merged, and filtered annotation file. If the file exists and
            `force` is False, it will be loaded instead of recomputing. Default is a file in the pooch cache directory.
        force (bool, optional): If True, forces regeneration of the merged and filtered annotation file even if the
            cached file exists. Default is False.

    Returns:
        pd.DataFrame: The merged, filtered, and annotated DataFrame containing VEP annotations for the variants of interest.

    Notes:
        - The function expects VEP annotation files to be located in the "clinvar_by_chrom" subdirectory of the pooch cache.
        - Additional columns are computed:
            - "mutant": Concatenation of reference amino acid, protein position, and alternate amino acid.
            - "ENST": Transcript ID without version.
            - "SIFT_score" and "PolyPhen_score": Extracted numeric scores from the respective columns.
            - "MaveDB_score_mean", "MaveDB_score_abs_mean", "MaveDB_score_min", "MaveDB_score_max": Statistics computed from comma-separated MaveDB scores.
    """
    if cached_file is not None and os.path.exists(cached_file) and not force:
        print(f"Reading from {cached_file}")
        cv_annot = pd.read_parquet(cached_file)
    else:
        # Get all VEP annotation files in the cache directory
        cv_vep_files = glob.glob(cache_search)
        if len(cv_vep_files) == 0:
            raise FileNotFoundError(f"No VEP annotation files found in {pooch.os_cache('pooch')}/clinvar_by_chrom")
        
        # Check that each of the filters are present in the VEP annotation files
        for col, filter_col in df_filters.items():
            if col not in df.columns:
                raise ValueError(f"Filter column {col} not found in `df`")

        cv_annot = []
        for f in tqdm(cv_vep_files, desc="Processing VEP annotation files"):
            cv_tmp = pd.read_csv(f, sep="\t", 
                                 low_memory=False, 
                                 na_values=['-'],
                                #  dtype={"Amino_acids": str,
                                #         "Protein_position": str}
                                 ) 
            
            # Filter according to df_filters
            for col, filter_col in df_filters.items():
                cv_tmp = cv_tmp.loc[
                    cv_tmp[filter_col].isin(df[col].unique().tolist())
                ]

            # Add to list
            cv_annot.append(cv_tmp)

        # Merge and save all filtered annotations
        cv_annot = pd.concat(cv_annot)
        if cached_file is not None:
            print(f"Caching merged and filtered annotations --> {cached_file}")
            cv_annot.to_parquet(cached_file)

    # Extract numeric values from SIFT and PolyPhen columns using regex
    print("Extracting numeric values from SIFT and PolyPhen columns")
    cv_annot['SIFT_score'] = cv_annot['SIFT'].str.extract(r'\(([\d.]+)\)').astype(float)
    cv_annot['PolyPhen_score'] = cv_annot['PolyPhen'].str.extract(r'\(([\d.]+)\)').astype(float)

    # Compute statistics from MaveDB_score column (comma-separated values)
    print("Computing statistics from MaveDB_score column")
    cv_annot['MaveDB_score'] = cv_annot['MaveDB_score'].astype(str)
    def safe_float(x):
        try:
            if x in [None, 'None', 'nan', 'NaN', '']:
                return np.nan
            return float(x)
        except Exception:
            return np.nan

    cv_annot['MaveDB_score_mean'] = cv_annot['MaveDB_score'].str.split(',').apply(
        lambda x: np.nanmean([safe_float(i) for i in x]) if isinstance(x, list) else np.nan
    )
    cv_annot['MaveDB_score_abs_mean'] = cv_annot['MaveDB_score'].str.split(',').apply(
        lambda x: np.nanmean([abs(safe_float(i)) for i in x]) if isinstance(x, list) else np.nan
    )
    cv_annot['MaveDB_score_min'] = cv_annot['MaveDB_score'].str.split(',').apply(
        lambda x: np.nanmin([safe_float(i) for i in x]) if isinstance(x, list) else np.nan
    )
    cv_annot['MaveDB_score_max'] = cv_annot['MaveDB_score'].str.split(',').apply(
        lambda x: np.nanmax([safe_float(i) for i in x]) if isinstance(x, list) else np.nan
    )


    # Create a new column using string operations in a more efficient way
    print("Creating chrom and chromStart columns")
    cv_annot["chrom"] = "chr" + cv_annot["Location"].str.split(":").str[0]
    cv_annot["chromStart"] = cv_annot["Location"].str.split(":").str[1].str.split("-").str[0]  
    
    print("Adding variant name column")
    cv_annot = utils.add_variant_name(cv_annot,  
                                      chrom_col='chrom',
                                    start_col='chromStart',
                                    end_col=None,
                                    ref_col='REF_ALLELE',
                                    alt_col='Allele',
                                    alias='site')
    
   
    #   Add mutant column: ref_aa + position + alt_aa
    if "Amino_acids" in cv_annot.columns:
        print("Adding mutant column")
        cv_annot.loc[:, "mutant"] = (
            cv_annot["Amino_acids"].str.split("/").str[0]
            + cv_annot["Protein_position"].astype(str)
            + cv_annot["Amino_acids"].str.split("/").str[1]
        ) 
    
    # Add ENST column: transcript ID without version
    print("Adding ENST column")
    cv_annot.loc[:, "ENST"] = cv_annot["Feature"].str.split(".").str[0]
    
    print(cv_annot.shape)
    return cv_annot

def run_correlation_analysis(vep_annot, 
                             group_col="is_ref",
                             vep_col="VEP",
                             ANNOT_COLS=ANNOT_COLS,
                             ref_vs_all=False,
                             method="spearman",
                             transform=None,
                             transform_kwargs={},
                             leave=False,
                             verbose=True):
    """
    Perform Spearman correlation analysis between VEP scores and annotation columns.

    For each annotation column in ANNOT_COLS, this function computes the Spearman correlation
    coefficient (rho) between the 'VEP' column and the annotation column, separately for
    reference ('is_ref' == True) and non-reference ('is_ref' == False) variants. The absolute
    difference in correlation coefficients between non-reference and reference is also computed.

    Parameters
    ----------
    vep_annot : pd.DataFrame
        DataFrame containing VEP scores, annotation columns, and an 'is_ref' boolean column.
    group_col : str, optional
        Name of the column containing the group to split the data into. Default is "is_ref".
    vep_col : str, optional
        Name of the column containing the VEP scores. Default is "VEP".
    ANNOT_COLS : list, optional
        List of annotation columns to include in the analysis. Default is ANNOT_COLS.
    ref_vs_all : bool, optional
        If False (default), compares the difference in correlation between reference VEPs and non-reference (personalized) VEPs.
        If True, compares the difference in correlation between reference VEPs and all VEPs.
    verbose : bool, optional
        If True, prints a summary of the correlation results. Default is True.

    Returns
    -------
    pd.DataFrame
        DataFrame summarizing the Spearman correlation coefficients for each annotation column,
        including the reference, non-reference, and their absolute difference.

    Notes
    -----
    - The function skips annotation columns not present in the input DataFrame.
    - Only rows with non-missing values in both the annotation column and 'VEP' are used.
    - Set the 'plot' variable to True to enable plotting (currently disabled).
    """
    if method == "spearman":
        from scipy.stats import spearmanr as corr
    elif method == "pearson":
        from scipy.stats import pearsonr as corr
    else:
        raise ValueError(f"Invalid method: {method}")
    
    import warnings
 
    r2_results = []
    if group_col not in vep_annot.columns:
        raise ValueError(f"Group column {group_col} not found in vep_annot dataframe")
    
    for col in tqdm(ANNOT_COLS, 
                    desc="Calculating Spearman r", 
                    leave=leave):
        # print(col)
        

        if col not in vep_annot.columns:
            warnings.warn(f"Skipping {col} because it is not in the vep_annot dataframe")
            continue

        vep_annot_sub = vep_annot.dropna(subset=[col, vep_col]) 

        if vep_annot_sub.empty:
            warnings.warn(f"Skipping {col} because it has no non-missing values")
            continue
    
        # Apply transformation to VEP and annotation columns
        try:
            vep_annot_sub.loc[:, vep_col] = vep_annot_sub[vep_col].astype(float)
            vep_annot_sub.loc[:, col] = vep_annot_sub[col].astype(float)
        except Exception:
            warnings.warn(f"Error converting {vep_col} and {col} to float")
            continue

        if transform is not None:
            vep_annot_sub.loc[:, vep_col] = vep_annot_sub[vep_col].transform(transform, **transform_kwargs)
            vep_annot_sub.loc[:, col] = vep_annot_sub[col].transform(transform, **transform_kwargs) 

        # Calculate Spearman r for each facet
        try:
            ref_n = vep_annot_sub[vep_annot_sub[group_col]].shape[0]
            ref_r, ref_p = corr(
                vep_annot_sub[vep_annot_sub[group_col]][vep_col],
                vep_annot_sub[vep_annot_sub[group_col]][col]
            )
        except Exception:
            warnings.warn(f"Error calculating ref_r for {col}")
            continue

        # Calculate Spearman r for all VEPs
        try:
            if ref_vs_all:
                nonref_n = vep_annot_sub.shape[0]
                nonref_r, nonref_p = corr(
                    vep_annot_sub[vep_col],
                    vep_annot_sub[col]
                )
            # Calculate Spearman r for non-reference VEPs
            else:
                nonref_n = vep_annot_sub[~vep_annot_sub[group_col]].shape[0]
                nonref_r, nonref_p = corr(
                    vep_annot_sub[~vep_annot_sub[group_col]][vep_col],
                    vep_annot_sub[~vep_annot_sub[group_col]][col]
                )
        except Exception:
            warnings.warn(f"Error calculating nonref_r for {col}")
            continue

        # Compute the difference in Rho
        r_diff = nonref_r - ref_r 
        rabs_diff = abs(nonref_r) - abs(ref_r)
        r2_diff =  nonref_r**2 - ref_r**2 

        r2_results.append({
            'annotation': col,
            'ref_r': ref_r,
            'nonref_r': nonref_r,
            'r_diff': r_diff,
            'rabs_diff': rabs_diff,
            'r2_diff': r2_diff,
            "ref_n": ref_n,
            "nonref_n": nonref_n,
            "ref_p": ref_p,
            "nonref_p": nonref_p
        }) 

    # Convert results to DataFrame for easy viewing
    r2_df = pd.DataFrame(r2_results)


    # Calculate FDR (q-values) for non-reference p-values using Benjamini-Hochberg
    from statsmodels.stats.multitest import multipletests
    r2_df["ref_fdr"] = multipletests(r2_df["ref_p"], method="fdr_bh")[1]
    r2_df["nonref_fdr"] = multipletests(r2_df["nonref_p"], method="fdr_bh")[1]

    # Calculate scores that weight the correlation by the p-value, then calclate the difference between groups
    r2_df["combined_diff"] = (r2_df["nonref_r"] *(1-r2_df["nonref_p"])) - (r2_df["ref_r"] *(1-r2_df["ref_p"]))
    r2_df["combined_abs_diff"] = (r2_df["nonref_r"].abs() *(1-r2_df["nonref_p"])) - (r2_df["ref_r"].abs() *(1-r2_df["ref_p"]))
    r2_df["combined2_diff"] = (r2_df["nonref_r"] *(1-r2_df["nonref_p"])).pow(2) - (r2_df["ref_r"] *(1-r2_df["ref_p"])).pow(2)

    
    if verbose:
        print("\nR2 Results Summary:")
        print(r2_df.sort_values('r_diff', ascending=False))
    return r2_df


def plot_correlation_analysis(
    r2_df,
    x_var="r2_diff",
    y_var="annotation",
    figsize=(6, 7),
    ylabel="Annotation",
    xlabel="Difference in Correlation Between Non-ref vs. Ref\n"
           r"($\rho_{\text{non-ref}} \text{ }  \Delta \text{ } \rho_{\text{ref}}$)",
    legend_title="non-ref\ncorrelation\n"
                 r"$(|ρ_{\text{non-ref}}|)$",
    title=None,
    add_summary_subtitle=True,
    hue="nonref_r_abs",
    palette="flare_r",
    # Filtering options
    min_n=None,
    max_p=None,
    min_diff=None,
    annotations=None,
    # Faceting option
    facet_col=None,
    facet_col_wrap=2,
    facet_sharex=True,
    facet_sharey=True,
    facet_height=None,
    facet_aspect=1.2,
    facet_legend="auto",
    # New argument for category column
    show_category_column=False,
    category_dict=None,
    category_palette=None,
    category_legend_title="Category",
    category_legend_loc='center left',
    category_legend_bbox_to_anchor=(0.8, 0.25),
):
    """
    Plot the correlation analysis results, with optional faceting by a column.
    Optionally adds a column to the left of the y-axis with boxes indicating annotation category.
    """
    import src.utils as utils
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Use global ANNOT_DICT if not provided
    global ANNOT_DICT
    if category_dict is None:
        category_dict = ANNOT_DICT

    r2_df = r2_df.copy()

    if min_n is not None:
        r2_df = r2_df.loc[(r2_df["ref_n"] > min_n)
                          & (r2_df["nonref_n"] > min_n)]

    if max_p is not None:
        r2_df = r2_df.loc[(r2_df["ref_p"] < max_p)
                          & (r2_df["nonref_p"] < max_p)]

    if min_diff is not None:
        r2_df = r2_df.loc[(r2_df[x_var].abs() > min_diff)]

    if annotations is not None:
        annotations = utils.as_list(annotations)
        r2_df = r2_df.loc[r2_df["annotation"].isin(annotations)]

    r2_df.dropna(inplace=True)
    r2_df.sort_values(x_var, ascending=False, inplace=True)
    r2_df.loc[:, "nonref_r_abs"] = r2_df["nonref_r"].abs()

    # --- Consistent category palette setup ---
    # Always determine all possible categories from category_dict (including "other")
    all_possible_categories = list(sorted(set(category_dict.values()) | {"other"}))
    if show_category_column:
        r2_df["category"] = r2_df["annotation"].map(category_dict)
        r2_df["category"] = r2_df["category"].fillna("other")
        if category_palette is None:
            # Use a consistent palette for all possible categories, not just those present in the data
            palette_colors = sns.color_palette("tab10", n_colors=len(all_possible_categories))
            category_palette = dict(zip(all_possible_categories, palette_colors))
        else:
            # If user provides a palette, ensure it covers all possible categories
            # Avoid item assignment on string
            if isinstance(category_palette, dict):
                for cat in all_possible_categories:
                    if cat not in category_palette:
                        # Assign a default color if missing
                        category_palette = dict(category_palette)  # Make a copy if not already
                        category_palette[cat] = sns.color_palette("tab10", n_colors=len(all_possible_categories))[all_possible_categories.index(cat)]
            else:
                # If category_palette is not a dict, ignore and use default
                palette_colors = sns.color_palette("tab10", n_colors=len(all_possible_categories))
                category_palette = dict(zip(all_possible_categories, palette_colors))

    if add_summary_subtitle:
        nonref_win_pct = (r2_df[x_var] > 0).sum() / len(r2_df.loc[r2_df[x_var] != 0])
        if title is None:
            title = f"Non-ref win rate: {nonref_win_pct:.1%}"
        else:
            title = f"{title}\nNon-ref win rate: {nonref_win_pct:.1%}"

    # Facet by a column if requested
    if facet_col is not None:
        import seaborn as sns
        import matplotlib.pyplot as plt

        # Set facet height if not provided
        if facet_height is None:
            # Try to scale height by number of y categories per facet
            n_facets = r2_df[facet_col].nunique()
            n_y = r2_df[y_var].nunique()
            facet_height = max(figsize[1] / n_facets, 3)

        g = sns.catplot(
            data=r2_df,
            kind="bar",
            x=x_var,
            y=y_var,
            hue=hue,
            palette=palette,
            col=facet_col,
            col_wrap=facet_col_wrap,
            sharex=facet_sharex,
            sharey=facet_sharey,
            height=facet_height,
            aspect=facet_aspect,
            legend=facet_legend,
        )
        g.set_axis_labels(xlabel, ylabel)
        g.set_titles(col_template="{col_name}")
        if title is not None:
            plt.subplots_adjust(top=0.85)
            g.fig.suptitle(title)
        # Replace underscores with spaces in y-tick labels for each facet
        for ax in g.axes.flatten():
            ax.set_yticklabels([
                label.get_text().replace('_af', '_AF') .replace('_', ' ')
                for label in ax.get_yticklabels()
            ])
        # Set legend title
        if legend_title is not None:
            g._legend.set_title(legend_title)
        # Add category column if requested (not supported for facet for now)
        if show_category_column:
            import warnings
            warnings.warn("Category column not currently supported for faceted plots.")
        return {'fig': g.fig, 'axes': g.axes, 'data': r2_df, 'facet': g}
    else:
        if show_category_column:
            # --- Custom plotting with category column ---
            import matplotlib.patches as mpatches

            # Prepare data for plotting
            y_labels = r2_df[y_var].tolist()
            y_pos = range(len(y_labels))
            # Map annotation to category and color
            categories = r2_df["category"].tolist()
            # Use the consistent palette for all possible categories
            cat_color_map = {cat: category_palette[cat] for cat in all_possible_categories}
            cat_colors = [cat_color_map[cat] for cat in categories]

            # Set up figure with two axes: one for category, one for barplot
            fig = plt.figure(figsize=figsize)
            # Gridspec: left for category, right for barplot
            from matplotlib.gridspec import GridSpec
            gs = GridSpec(1, 2, width_ratios=[0.05, 0.95], wspace=0.05)
            ax_cat = fig.add_subplot(gs[0, 0])
            ax_bar = fig.add_subplot(gs[0, 1], sharey=ax_cat)

            # Draw category boxes
            for i, (cat, color) in enumerate(zip(categories, cat_colors)):
                ax_cat.add_patch(
                    mpatches.Rectangle(
                        (0.01, i - 0.4), 1, 0.8, color=color, ec='black', linewidth=2
                    )
                )
            ax_cat.set_ylim(-0.5, len(y_labels) - 0.5)
            ax_cat.set_xlim(0, 1)
            ax_cat.set_xticks([])
            ax_cat.set_yticks(y_pos)
            # Show y-tick labels only on barplot axis
            ax_cat.set_yticklabels([])
            ax_cat.tick_params(left=False, labelleft=False, right=False)
            ax_cat.set_frame_on(False)

            # Draw barplot
            sns.barplot(
                data=r2_df,
                x=x_var,
                y=y_var,
                hue=hue,
                palette=palette,
                ax=ax_bar
            )
            ax_bar.set_ylabel(ylabel)
            ax_bar.set_xlabel(xlabel)
            if title is not None:
                ax_bar.set_title(title)

            # Move y-tick labels to the left axis (category)
            ax_bar.set_yticklabels([])
            ax_bar.tick_params(left=False, labelleft=False)
            # Set legend in the lower right
            handles, labels = ax_bar.get_legend_handles_labels()
            if legend_title is not None:
                ax_bar.legend(
                    handles=handles,
                    title=legend_title,
                    frameon=False,
                    loc='lower right'
                )

            # Replace underscores with spaces in y-tick labels
            ax_bar.set_yticklabels([label.get_text().replace('_', ' ').replace(' af', ' AF') for label in ax_bar.get_yticklabels()])

            # Add category legend, sorted alphabetically
            sorted_categories = sorted(all_possible_categories)
            cat_legend_handles = [
                mpatches.Patch(color=cat_color_map[cat], label=cat)
                for cat in sorted_categories
            ]
            # Place category legend to the left of the plot
            fig.legend(
                handles=cat_legend_handles,
                title=category_legend_title,
                loc=category_legend_loc,
                bbox_to_anchor=category_legend_bbox_to_anchor,
                frameon=False
            )
            # Add y-tick labels to the left of the category boxes
            for i, label in enumerate(y_labels):
                ax_cat.text(-0.05, i, label.replace('_', ' '), va='center', ha='right', fontsize=plt.rcParams.get("ytick.labelsize", 10))
            plt.subplots_adjust(left=0.22, right=0.98, wspace=0.02)

            # Move the y-axis label further to the left to avoid overlap with annotation names
            ax_bar.yaxis.set_label_coords(-0.75, 0.5)

            # Remove top and right spines (margin lines) for both axes
            ax_cat.spines['top'].set_visible(False)
            ax_cat.spines['right'].set_visible(False)
            ax_bar.spines['top'].set_visible(False)
            ax_bar.spines['right'].set_visible(False)

            return {'fig': fig, 'axes': (ax_cat, ax_bar), 'data': r2_df}
        else:
            plt.figure(figsize=figsize)
            g = sns.barplot(
                data=r2_df,
                x=x_var,
                y=y_var,
                hue=hue,
                palette=palette
            )
            plt.legend(title=legend_title)
            plt.ylabel(ylabel)
            plt.xlabel(xlabel)
            if title is not None:
                plt.title(title)
            # Replace underscores with spaces in y-tick labels
            _ = plt.gca().set_yticklabels([label.get_text().replace('_', ' ') for label in plt.gca().get_yticklabels()])
            
            # Remove top and right spines (margin lines) for both axes
            # Fix: g.axes is a numpy array of axes, not a string, so we should not assign to g.axes[0] as a string
            # Instead, check if g.axes is an array or a single axis
            axes = g.axes if hasattr(g, "axes") else [g]
            if hasattr(axes, "__iter__"):
                for ax in axes:
                    if hasattr(ax, "spines"):
                        ax.spines['top'].set_visible(False)
                        ax.spines['right'].set_visible(False)
            else:
                if hasattr(axes, "spines"):
                    axes.spines['top'].set_visible(False)
                    axes.spines['right'].set_visible(False)
            
            return {'fig': g.figure, 'axes': g.axes, 'data': r2_df}
