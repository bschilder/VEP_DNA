import numpy as np
import pandas as pd
import src.utils as utils

 

def wtvariants_to_vep_linear_model(
        Xwt,
        y_vep, 
        model_type="ridge",  # "ridge" or "lasso"
        alpha=1.0,
        random_state=42, 
        model_kwargs={}, 
        add_positions=True,
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
    if n_dropped > 0:
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
        interaction_df["wt_position"] = interaction_df["wt_variant"].str.split(">").str[0].astype(int)
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