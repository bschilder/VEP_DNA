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
        test_epistasis=False,
        epistasis_alpha=None,
        epistasis_pvalue_threshold=0.05,
        verbose=True,
    ):
    """
    Fit a Ridge or Lasso regression model to predict VEP values from wt_variant features,
    and compute input-output (wt_variant-site) interaction strengths, including directionality.

    Parameters
    ----------
    Xwt : pd.DataFrame
        Input matrix (haplotype x wt_variant: binary matrix).
    y_vep : pd.DataFrame
        Target matrix (haplotype x site: VEP values).
    model_type : {"ridge", "lasso"}, default="ridge"
        Type of linear model to fit. "ridge" for Ridge regression, "lasso" for Lasso regression.
    alpha : float, default=1.0 (same default as sklearn)
        Regularization strength; must be a positive float. Larger values specify stronger regularization.
        For Ridge regression, this corresponds to the L2 penalty term, and for Lasso regression, to the L1 penalty term.
    random_state : int, default=42
        Random seed for reproducibility.
    model_kwargs : dict, default={}
        Additional keyword arguments to pass to the model constructor.
    add_positions : bool, default=True
        If True, add positions to the interaction_df.
    test_epistasis : bool, default=False
        If True, compute epistasis within this model. This tests whether each WT variant-site
        interaction is truly epistatic (non-additive) rather than simply additive.
        When False, use the separate function `test_epistasis_across_models()` after training
        all models for cross-model epistasis testing.
    epistasis_alpha : float, optional
        Regularization strength for epistasis testing models. If None, uses alpha.
    epistasis_pvalue_threshold : float, default=0.05
        P-value threshold for determining epistatic vs additive interactions.
    verbose : bool, default=True
        If True, print progress.

    Returns
    -------
    dict with keys:
        interaction_df : pd.DataFrame
            DataFrame with columns ['wt_variant', 'site', 'interaction_strength', 'interaction_strength_signed', 
            'n_haplotypes', 'interaction_strength_weighted']
            If test_epistasis=True, also includes epistasis columns: 'is_epistatic', 'epistasis_pvalue', 
            'epistasis_fstat', 'additive_r2', 'interaction_r2', 'delta_r2', 'additive_mse', 'interaction_mse',
            'epistasis_coefficient', 'joint_effect', 'expected_additive_effect', 'deviation_from_additive',
            'wt_individual_effect', 'clinical_individual_effect'
        model : fitted sklearn model
        Xwt_clean : pd.DataFrame
            Cleaned input matrix (haplotype x wt_variant: binary matrix)
        y_vep_clean : pd.DataFrame
            Cleaned target matrix (haplotype x site: VEP)
        coef_matrix_signed : pd.DataFrame
            Signed interaction strength matrix (wt_variant x site)
        coef_matrix_abs : pd.DataFrame
            Absolute interaction strength matrix (wt_variant x site)
        epistasis_results : dict, optional
            If test_epistasis=True, contains summary statistics: 'n_tested', 'n_epistatic', 
            'n_additive', 'epistasis_rate'
    
    Note
    ----
    For epistasis testing, there are two options:
    1. Within-model epistasis (test_epistasis=True): Computes epistasis within a single model,
       comparing observed joint effects to expected additive effects based on individual WT and
       clinical variant effects computed from the same model.
    2. Cross-model epistasis (test_epistasis=False, then use test_epistasis_across_models()): 
       Computes epistasis after training all models, using cross-model WT effect estimates.
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

    return_dict = {"interaction_df": interaction_df, "model": model, 
            "Xwt_clean": Xwt_clean, "y_vep_clean": y_vep_clean,
            "coef_matrix_signed": coef_matrix_signed_df, 
            "coef_matrix_abs": coef_matrix_abs_df}
    
    # Compute epistasis within model if requested
    if test_epistasis:
        if verbose:
            print("Computing epistasis within model...")
        
        from scipy.stats import f as f_distribution
        from sklearn.metrics import r2_score, mean_squared_error
        from tqdm import tqdm
        
        epistasis_alpha = epistasis_alpha or alpha
        ModelClass = Ridge if model_type == "ridge" else Lasso
        model_kwargs_epi = dict(alpha=epistasis_alpha, random_state=random_state, fit_intercept=True, **model_kwargs)
        
        # Step 1: Compute individual WT variant effects (averaging across all clinical variants)
        wt_individual_effects = {}
        for wt_var in input_features:
            wt_vals = Xwt_clean[wt_var].values.astype(float)
            effects = []
            for site in output_sites:
                y_vals = y_vep_clean[site].values
                mask = ~(np.isnan(wt_vals) | np.isnan(y_vals))
                if mask.sum() < 2:
                    continue
                wt_clean, y_clean = wt_vals[mask], y_vals[mask]
                if wt_clean.std() < 1e-10:
                    continue
                wt_mask = wt_clean > 0.5
                if wt_mask.any() and (~wt_mask).any():
                    effect = y_clean[wt_mask].mean() - y_clean[~wt_mask].mean()
                    if not np.isnan(effect):
                        effects.append(effect)
            wt_individual_effects[wt_var] = np.mean(effects) if effects else 0.0
        
        # Step 2: Compute individual clinical variant effects (relative to baseline)
        baseline_vep = y_vep_clean.values.mean()
        clinical_individual_effects = {}
        for site in output_sites:
            y_vals = y_vep_clean[site].values
            mask = ~np.isnan(y_vals)
            if mask.sum() > 0:
                clinical_individual_effects[site] = y_vals[mask].mean() - baseline_vep
            else:
                clinical_individual_effects[site] = 0.0
        
        # Step 3: Test epistasis for each WT variant-site pair
        epistasis_cols = ['is_epistatic', 'epistasis_pvalue', 'epistasis_fstat', 'additive_r2', 
                         'interaction_r2', 'delta_r2', 'additive_mse', 'interaction_mse', 
                         'epistasis_coefficient', 'joint_effect', 'expected_additive_effect',
                         'deviation_from_additive', 'wt_individual_effect', 'clinical_individual_effect']
        for col in epistasis_cols:
            interaction_df[col] = np.nan if col != 'is_epistatic' else False
        
        for idx, row in tqdm(interaction_df.iterrows(), total=len(interaction_df), desc="Epistasis", disable=not verbose):
            wt_var = row['wt_variant']
            site = row['site']
            
            if wt_var not in Xwt_clean.columns or site not in y_vep_clean.columns:
                continue
            
            wt_vals = Xwt_clean[wt_var].values.astype(float)
            y_vals = y_vep_clean[site].values
            mask = ~(np.isnan(wt_vals) | np.isnan(y_vals))
            
            if mask.sum() < 10:
                continue
            
            wt_clean, y_clean = wt_vals[mask], y_vals[mask]
            if wt_clean.std() < 1e-10 or len(set(wt_clean.astype(int))) < 2:
                continue
            
            # Get effects
            joint_effect = row['interaction_strength_signed']
            wt_effect = wt_individual_effects.get(wt_var, 0.0)
            clinical_effect = clinical_individual_effects.get(site, 0.0)
            expected_additive = wt_effect + clinical_effect
            
            # Fit additive model: VEP = beta_0 + beta_1 * WT
            X_add = wt_clean.reshape(-1, 1)
            model_add = ModelClass(**model_kwargs_epi).fit(X_add, y_clean)
            y_pred_add = model_add.predict(X_add)
            
            # Fit interaction model: VEP = beta_0 + beta_1 * WT + beta_2 * (WT * deviation)
            # The deviation term captures how much the joint effect deviates from additivity
            # Note: Methods doc says Delta_deviation = Effect_additive - Effect_WT, but we use
            # the observed deviation (joint_effect - expected_additive) which is more appropriate
            # for testing epistasis as it directly measures non-additivity
            deviation = joint_effect - expected_additive
            epistasis_term = wt_clean * deviation
            X_int = np.column_stack([wt_clean, epistasis_term])
            model_int = ModelClass(**model_kwargs_epi).fit(X_int, y_clean)
            y_pred_int = model_int.predict(X_int)
            
            # Metrics
            r2_add = r2_score(y_clean, y_pred_add)
            r2_int = r2_score(y_clean, y_pred_int)
            mse_add = mean_squared_error(y_clean, y_pred_add)
            mse_int = mean_squared_error(y_clean, y_pred_int)
            
            # F-test
            n = len(y_clean)
            rss_add, rss_int = mse_add * n, mse_int * n
            df_add, df_int = n - 2, n - 3
            
            if rss_int > 1e-10 and df_int > 0 and (df_add - df_int) > 0:
                f_stat = ((rss_add - rss_int) / (df_add - df_int)) / (rss_int / df_int)
                f_stat = max(0, f_stat)
                try:
                    pvalue = 1 - f_distribution.cdf(f_stat, df_add - df_int, df_int)
                except:
                    pvalue = np.nan
            else:
                f_stat = pvalue = np.nan
            
            # Store results
            interaction_df.at[idx, 'is_epistatic'] = pvalue < epistasis_pvalue_threshold if not np.isnan(pvalue) else False
            interaction_df.at[idx, 'epistasis_pvalue'] = pvalue
            interaction_df.at[idx, 'epistasis_fstat'] = f_stat
            interaction_df.at[idx, 'additive_r2'] = r2_add
            interaction_df.at[idx, 'interaction_r2'] = r2_int
            interaction_df.at[idx, 'delta_r2'] = r2_int - r2_add
            interaction_df.at[idx, 'additive_mse'] = mse_add
            interaction_df.at[idx, 'interaction_mse'] = mse_int
            interaction_df.at[idx, 'epistasis_coefficient'] = model_int.coef_[1] if len(model_int.coef_) > 1 else np.nan
            interaction_df.at[idx, 'joint_effect'] = joint_effect
            interaction_df.at[idx, 'expected_additive_effect'] = expected_additive
            interaction_df.at[idx, 'deviation_from_additive'] = joint_effect - expected_additive
            interaction_df.at[idx, 'wt_individual_effect'] = wt_effect
            interaction_df.at[idx, 'clinical_individual_effect'] = clinical_effect
        
        # Summary statistics
        n_tested = (~np.isnan(interaction_df['epistasis_pvalue'])).sum()
        n_epistatic = interaction_df['is_epistatic'].sum()
        
        epistasis_results = {
            'n_tested': n_tested,
            'n_epistatic': n_epistatic,
            'n_additive': n_tested - n_epistatic,
            'epistasis_rate': n_epistatic / n_tested if n_tested > 0 else 0.0,
        }
        
        return_dict['epistasis_results'] = epistasis_results
        
        if verbose:
            print(f"Epistasis testing complete: {n_tested} tested, {n_epistatic} epistatic (p < {epistasis_pvalue_threshold}), rate: {epistasis_results['epistasis_rate']:.4f}")
    
    return return_dict


def test_epistasis_across_models(
    models_dict,
    Xwt_full,
    y_vep_full,
    model_type="ridge",
    epistasis_alpha=None,
    epistasis_pvalue_threshold=0.05,
    verbose=True,
    random_state=42,
    model_kwargs={},
):
    """
    Test epistasis across all trained ridge regression models.
    
    This function computes epistasis testing after all models have been trained.
    It computes the independent effect of each WT variant across ALL clinical variants,
    then tests whether each (WT variant, clinical variant) interaction is truly epistatic
    (non-additive) rather than simply additive.
    
    Parameters
    ----------
    models_dict : dict
        Dictionary mapping clinical variant names to model result dictionaries.
        Each model result dict should have keys: 'interaction_df', 'Xwt_clean', 'y_vep_clean'
    Xwt_full : pd.DataFrame
        Full WT variant matrix (haplotype x wt_variant: binary matrix) for all haplotypes.
    y_vep_full : pd.DataFrame
        Full VEP target matrix (haplotype x clinical_variant: VEP values) for all haplotypes.
    model_type : {"ridge", "lasso"}, default="ridge"
        Type of linear model used in training. Must match the models in models_dict.
    epistasis_alpha : float, optional
        Regularization strength for epistasis testing models. If None, uses alpha=1.0.
    epistasis_pvalue_threshold : float, default=0.05
        P-value threshold for determining epistatic vs additive interactions.
    verbose : bool, default=True
        If True, print progress.
    random_state : int, default=42
        Random seed for reproducibility.
    model_kwargs : dict, default={}
        Additional keyword arguments to pass to the model constructor.
    
    Returns
    -------
    dict
        Dictionary with keys:
        - 'epistasis_df': pd.DataFrame with all original and epistasis results combined
        - 'epistasis_results': dict with summary statistics
        - 'updated_models': dict with updated model dictionaries (including epistasis results)
    """
    from sklearn.linear_model import Ridge, Lasso
    from scipy.stats import f as f_distribution
    from sklearn.metrics import r2_score, mean_squared_error
    from tqdm import tqdm
    
    epistasis_alpha = epistasis_alpha or 1.0
    ModelClass = Ridge if model_type == "ridge" else Lasso
    model_kwargs_full = dict(alpha=epistasis_alpha, random_state=random_state, fit_intercept=True, **model_kwargs)
    
    if verbose:
        print(f"Computing epistasis across {len(models_dict)} models...")
    
    # Step 1: Compute WT individual effects (vectorized where possible)
    all_wt_variants = {wt for mr in models_dict.values() 
                      for wt in mr.get('Xwt_clean', pd.DataFrame()).columns}
    
    wt_individual_effects = {}
    for wt_var in tqdm(all_wt_variants, desc="WT effects", disable=not verbose):
        effects = []
        for clinical_var, mr in models_dict.items():
            Xwt_clean = mr.get('Xwt_clean')
            y_vep_clean = mr.get('y_vep_clean')
            if Xwt_clean is None or wt_var not in Xwt_clean.columns:
                continue
            if y_vep_clean is None or clinical_var not in y_vep_clean.columns:
                continue
            
            wt_vals = Xwt_clean[wt_var].values.astype(float)
            y_vals = y_vep_clean[clinical_var].values
            mask = ~(np.isnan(wt_vals) | np.isnan(y_vals))
            
            if mask.sum() < 2:
                continue
            
            wt_clean, y_clean = wt_vals[mask], y_vals[mask]
            if wt_clean.std() < 1e-10:
                continue
            
            wt_mask = wt_clean > 0.5
            if wt_mask.any() and (~wt_mask).any():
                effect = y_clean[wt_mask].mean() - y_clean[~wt_mask].mean()
                if not np.isnan(effect):
                    effects.append(effect)
        
        wt_individual_effects[wt_var] = np.mean(effects) if effects else 0.0
    
    if verbose:
        print(f"Computed effects for {len(wt_individual_effects)} WT variants")
    
    # Step 2: Process each model and store results directly in dataframes
    all_dfs = []
    updated_models = {}
    
    for clinical_var, mr in tqdm(models_dict.items(), desc="Epistasis", disable=not verbose):
        interaction_df = mr.get('interaction_df')
        Xwt_clean = mr.get('Xwt_clean')
        y_vep_clean = mr.get('y_vep_clean')
        
        if interaction_df is None or Xwt_clean is None or y_vep_clean is None:
            continue
        if clinical_var not in y_vep_clean.columns:
            continue
        
        df = interaction_df.copy()
        y_values = y_vep_clean[clinical_var].values
        
        # Initialize epistasis columns
        epistasis_cols = ['is_epistatic', 'epistasis_pvalue', 'epistasis_fstat', 'additive_r2', 
                         'interaction_r2', 'delta_r2', 'additive_mse', 'interaction_mse', 
                         'epistasis_coefficient', 'joint_effect', 'expected_additive_effect',
                         'deviation_from_additive', 'wt_individual_effect']
        for col in epistasis_cols:
            df[col] = np.nan if col != 'is_epistatic' else False
        
        # Process each row
        for idx, row in df.iterrows():
            wt_var = row['wt_variant']
            if wt_var not in Xwt_clean.columns:
                continue
            
            wt_ind = Xwt_clean[wt_var].values.astype(float)
            mask = ~(np.isnan(wt_ind) | np.isnan(y_values))
            
            if mask.sum() < 10:
                continue
            
            wt_clean, y_clean = wt_ind[mask], y_values[mask]
            if wt_clean.std() < 1e-10 or len(set(wt_clean.astype(int))) < 2:
                continue
            
            joint_effect = row['interaction_strength_signed']
            wt_effect = wt_individual_effects.get(wt_var, 0.0)
            expected_additive = wt_effect
            
            # Fit models
            X_add = wt_clean.reshape(-1, 1)
            model_add = ModelClass(**model_kwargs_full).fit(X_add, y_clean)
            y_pred_add = model_add.predict(X_add)
            fitted_wt_coef = model_add.coef_[0]
            
            epistasis_coef = fitted_wt_coef - wt_effect
            expected_dev = joint_effect - expected_additive
            epistasis_term = wt_clean * expected_dev
            
            X_int = np.column_stack([wt_clean, epistasis_term])
            model_int = ModelClass(**model_kwargs_full).fit(X_int, y_clean)
            y_pred_int = model_int.predict(X_int)
            
            # Metrics
            r2_add = r2_score(y_clean, y_pred_add)
            r2_int = r2_score(y_clean, y_pred_int)
            mse_add = mean_squared_error(y_clean, y_pred_add)
            mse_int = mean_squared_error(y_clean, y_pred_int)
            
            # F-test
            n = len(y_clean)
            rss_add, rss_int = mse_add * n, mse_int * n
            df_add, df_int = n - 2, n - 3
            
            if rss_int > 1e-10 and df_int > 0 and (df_add - df_int) > 0:
                f_stat = ((rss_add - rss_int) / (df_add - df_int)) / (rss_int / df_int)
                f_stat = max(0, f_stat)
                try:
                    pvalue = 1 - f_distribution.cdf(f_stat, df_add - df_int, df_int)
                except:
                    pvalue = np.nan
            else:
                f_stat = pvalue = np.nan
            
            # Store results
            df.at[idx, 'is_epistatic'] = pvalue < epistasis_pvalue_threshold if not np.isnan(pvalue) else False
            df.at[idx, 'epistasis_pvalue'] = pvalue
            df.at[idx, 'epistasis_fstat'] = f_stat
            df.at[idx, 'additive_r2'] = r2_add
            df.at[idx, 'interaction_r2'] = r2_int
            df.at[idx, 'delta_r2'] = r2_int - r2_add
            df.at[idx, 'additive_mse'] = mse_add
            df.at[idx, 'interaction_mse'] = mse_int
            df.at[idx, 'epistasis_coefficient'] = epistasis_coef
            df.at[idx, 'joint_effect'] = joint_effect
            df.at[idx, 'expected_additive_effect'] = expected_additive
            df.at[idx, 'deviation_from_additive'] = joint_effect - expected_additive
            df.at[idx, 'wt_individual_effect'] = wt_effect
        
        all_dfs.append(df)
        updated_models[clinical_var] = {**mr, 'interaction_df': df}
    
    epistasis_df = pd.concat(all_dfs, ignore_index=True)
    
    # Summary
    n_tested = (~np.isnan(epistasis_df['epistasis_pvalue'])).sum()
    n_epistatic = epistasis_df['is_epistatic'].sum()
    
    epistasis_results = {
        'n_tested': n_tested,
        'n_epistatic': n_epistatic,
        'n_additive': n_tested - n_epistatic,
        'epistasis_rate': n_epistatic / n_tested if n_tested > 0 else 0.0,
    }
    
    if verbose:
        print(f"\nTested: {n_tested}, Epistatic: {n_epistatic} (p < {epistasis_pvalue_threshold}), Rate: {epistasis_results['epistasis_rate']:.4f}")
    
    return {
        'epistasis_df': epistasis_df,
        'epistasis_results': epistasis_results,
        'updated_models': updated_models,
    }


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
    clinsig_order=None,  # Order for clinical significance categories
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
    use_abs=False,  # New argument to use absolute values
    yaxis_scientific=False,  # New argument to use scientific notation on y-axis
    show_xticklabels=True,  # New argument to show/hide x-axis tick labels
    yaxis_log=False,  # New argument to use log scale on y-axis
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
    use_abs : bool, default False
        If True, use absolute values of the y-axis metric for plotting.
    yaxis_scientific : bool, default False
        If True, format y-axis tick labels using scientific notation (e.g., 1e-3).
    show_xticklabels : bool, default True
        If False, hide x-axis tick labels.
    yaxis_log : bool, default False
        If True, use log scale on y-axis.
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
    
    >>> # Using absolute values
    >>> result = plot_clinsig_interaction_strength(
    ...     ridge_df,
    ...     use_abs=True,
    ...     ylabel="Absolute Marginal Effect Size"
    ... )
    
    >>> # With scientific notation on y-axis
    >>> result = plot_clinsig_interaction_strength(
    ...     ridge_df,
    ...     yaxis_scientific=True
    ... )
    
    >>> # Hide x-axis tick labels
    >>> result = plot_clinsig_interaction_strength(
    ...     ridge_df,
    ...     show_xticklabels=False
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
    import matplotlib.ticker as ticker
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
        bar_df = ridge_df.groupby("clinical_variant", observed=True)[y].agg(agg_func).reset_index().merge(
            annot_df[[site_col, x]].drop_duplicates().rename(columns={site_col: "clinical_variant"})
        )

    # Apply absolute value if requested
    if use_abs:
        bar_df[y] = bar_df[y].abs()

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

    # Use provided clinsig_order or default
    if clinsig_order is None:
        clinsig_order = utils.get_clinsig_order()
    
    # Sort by the provided order (using raw values)
    bar_df = utils.sort_by_clinsig(bar_df, clinsig_col=x, clinsig_order=clinsig_order)

    # Clean the labels in the dataframe
    bar_df[x] = bar_df[x].astype(str).apply(clean_clinsig)
    
    # Map the provided order to cleaned order to preserve the desired sequence
    clinsig_order_cleaned = [clean_clinsig(str(val)) for val in clinsig_order]
    
    # Only include categories that actually exist in the data, preserving the order
    categories_in_data = []
    for cat in clinsig_order_cleaned:
        if cat in bar_df[x].values and cat not in categories_in_data:
            categories_in_data.append(cat)
    
    # Convert to Categorical with the cleaned order to force seaborn to respect it
    bar_df[x] = pd.Categorical(bar_df[x], categories=categories_in_data, ordered=True)
 
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
        g.map_dataframe(
            sns.boxplot, 
            x=x, 
            y=y, 
            hue=x, 
            palette=palette_cleaned, 
            order=clinsig_order_cleaned, 
            showfliers=False,
            medianprops={'color': 'black', 'linewidth': 1},
            whiskerprops={'color': 'black', 'linewidth': 1},
            capprops={'color': 'black', 'linewidth': 1},
            boxprops={'edgecolor': 'black', 'linewidth': 1}
        )
        
        # Set title and labels for each subplot
        for ax in g.axes.flat:
            ax.set_title(title)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            
            # Apply scientific notation to y-axis if requested
            if yaxis_scientific:
                ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
                ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))
            
            # Apply log scale to y-axis if requested
            if yaxis_log:
                ax.set_yscale('log')
            
            # Hide x-axis tick labels if requested
            if not show_xticklabels:
                ax.set_xticklabels([])
            # Optionally rotate x-axis labels and adjust anchor
            elif xlabel_rotation is not None:
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
            order=clinsig_order_cleaned,
            showfliers=False,
            medianprops={'color': 'black', 'linewidth': 1},
            whiskerprops={'color': 'black', 'linewidth': 1},
            capprops={'color': 'black', 'linewidth': 1},
            boxprops={'edgecolor': 'black', 'linewidth': 1}
        )

        # Set title and labels
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)

        # Apply scientific notation to y-axis if requested
        if yaxis_scientific:
            ax.yaxis.set_major_formatter(ticker.ScalarFormatter(useMathText=True))
            ax.ticklabel_format(style='scientific', axis='y', scilimits=(0, 0))

        # Apply log scale to y-axis if requested
        if yaxis_log:
            ax.set_yscale('log')

        # Hide x-axis tick labels if requested
        if not show_xticklabels:
            ax.set_xticklabels([])
        # Optionally rotate x-axis labels and adjust anchor
        elif xlabel_rotation is not None:
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
        # Use categorical categories if available (most reliable), otherwise get from tick labels
        if isinstance(bar_df[x].dtype, pd.CategoricalDtype):
            clinsig_order = list(bar_df[x].cat.categories)
        elif not show_xticklabels:
            # If labels are hidden, use the cleaned order we computed
            clinsig_order = clinsig_order_cleaned
        else:
            # Get from tick labels, filtering out empty strings
            clinsig_order = [t.get_text() for t in ax.get_xticklabels() if t.get_text()]
            # Fallback to cleaned order if we got empty strings
            if not clinsig_order or all(not text for text in clinsig_order):
                clinsig_order = clinsig_order_cleaned

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
    drop_empty_rows=False,
    drop_empty_cols=False,
    mismatched_legend_kws={"bbox_to_anchor": (0.5, 1.25), "loc": "upper left", "title": "", "frameon": False},
    tick_params_kwargs = {'pad': 1},
    cbar_title=r"$log_{10}(|\text{Mean Joint Effect}|)$",
    cbar_tick_labelsize=None,
    flip_xticklabels=False,
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
    drop_empty_rows : bool, default False
        If True, automatically drop rows where all values are NaN or zero.
    drop_empty_cols : bool, default False
        If True, automatically drop columns where all values are NaN or zero.
    cbar_position : tuple, optional
        Position of the colorbar as (x, y, width, height) in figure coordinates.
    mismatched_legend_kws : dict, optional
        Additional keyword arguments to pass to the legend.
    cbar_title : str, optional
        Title for the colorbar.
    cbar_tick_labelsize : str or float, optional
        Size of the colorbar tick labels. Can be a string like 'small', 'medium', 'large', or a float.
    flip_xticklabels : bool, default False
        If True, flip xtick labels to angle to the right (-45 degrees) with left side facing the heatmap.
        If False, labels are rotated 45 degrees with right side facing the heatmap (default).
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
    
    # Drop empty rows/columns if requested
    # Empty means all values are NaN or zero (or both)
    if drop_empty_rows:
        # A row is empty if all values are NaN or zero
        empty_rows = orig_concat_df.index[
            (orig_concat_df.isna().all(axis=1)) | 
            ((orig_concat_df.fillna(0) == 0).all(axis=1))
        ]
        if len(empty_rows) > 0:
            orig_concat_df = orig_concat_df.drop(index=empty_rows)
    
    if drop_empty_cols:
        # A column is empty if all values are NaN or zero
        empty_cols = orig_concat_df.columns[
            (orig_concat_df.isna().all(axis=0)) | 
            ((orig_concat_df.fillna(0) == 0).all(axis=0))
        ]
        if len(empty_cols) > 0:
            orig_concat_df = orig_concat_df.drop(columns=empty_cols)
            # Update col_colors_flat to match remaining columns
            remaining_cols = orig_concat_df.columns.tolist()
            new_col_colors_flat = []
            for name, mat in zip(matrix_names, orig_matrices):
                for c in mat.columns:
                    if c in remaining_cols:
                        new_col_colors_flat.append(name_to_color[name])
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
    g.ax_heatmap.set_ylabel(ylabel, rotation=270, va="bottom")
    
    # Set xtick label rotation and alignment
    if flip_xticklabels:
        g.ax_heatmap.set_xticklabels(g.ax_heatmap.get_xticklabels(), rotation=-45, ha="left")
    else:
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
    # Set colorbar tick label size and padding
    
    if cbar_tick_labelsize is not None:
        tick_params_kwargs['labelsize'] = cbar_tick_labelsize
    g.ax_cbar.tick_params(**tick_params_kwargs)
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
    value_col="interaction_strength",
    exon_distances = [],
    abs_before_agg=False
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
    value_col : str, default "interaction_strength"
        Column to use for the value in the matrix.
    exon_distances : list, default []
        List of exon distances to compute matrices for.
    abs_before_agg : bool, default False
        If True, take absolute value of values before aggregating (taking mean).

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
                        values = ridge_df_annot.loc[mask, value_col]
                        if abs_before_agg:
                            values = np.abs(values)
                        matrix.loc[wt_col, clinical_col] = values.mean()
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
                            values = ridge_df_annot.loc[mask, value_col]
                            if abs_before_agg:
                                values = np.abs(values)
                            mat.loc[wt_col, clinical_col] = values.mean()
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
    log10_transform_y=False,
    log_transform_y=False,
    wrap_xlabels=None,
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
    log10_transform_y : bool, default=False
        If True, apply -log10 transform to y-axis values. Values <= 0 are set to NaN.
    log_transform_y : bool, default=False
        If True, apply log10 transform (without negative) to y-axis values. Values <= 0 are set to NaN.
        Mutually exclusive with log10_transform_y.
    wrap_xlabels : int or None, default=None
        If provided, wrap x-axis tick labels at this character width. None means no wrapping.
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
    
    # Apply log transform if requested
    if log10_transform_y and log_transform_y:
        raise ValueError("Cannot use both log10_transform_y and log_transform_y simultaneously")
    elif log10_transform_y:
        flattened = [np.where(arr > 0, -np.log10(arr), np.nan) for arr in flattened]
        if ylabel == "Mean Joint Effect Size":
            ylabel = "-log$_{10}$(Mean Joint Effect Size)"
    elif log_transform_y:
        flattened = [np.where(arr > 0, np.log10(arr), np.nan) for arr in flattened]
        if ylabel == "Mean Joint Effect Size":
            ylabel = "log$_{10}$(Mean Joint Effect Size)"
    
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
    labels = [label.get_text().replace("_", " ") for label in ax.get_xticklabels()]
    # Wrap labels if requested
    if wrap_xlabels is not None:
        import textwrap
        labels = [textwrap.fill(label, width=wrap_xlabels) for label in labels]
    ax.set_xticklabels(labels)

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

 
def plot_epistasis_pvalue_barplot(
    ridge_df_annot,
    y="epistasis_pvalue",
    x="exon_number_distance",
    figsize=(4,3),
    only_nearest_neighbor_comparisons=True,
    correction_method="bonferroni",
    test="Mann-Whitney",
    comparison_text_format="star",
    log_scale_y=False,
    showplot=True,
    return_fig_ax=False
):
    """
    Visualize -log10(epistasis_pvalue) for each exon_number_distance group using a barplot,
    and annotate significance with statistical tests.

    Parameters
    ----------
    ridge_df_annot : pd.DataFrame
        Dataframe to use for plotting.
    y : str
        Column in ridge_df_annot with the p-values to plot (will plot -log10).
    x : str
        Name of column with exon_number_distance (should be int or convertible to int + NaN).
    figsize : tuple
        Figure size.
    only_nearest_neighbor_comparisons : bool
        If True (default), only compare adjacent/existing exon_number_distance categories.
        If False, compare all pairwise combinations among present groups.
    correction_method : str
        Statistical correction method, e.g., bonferroni, fdr_bh, etc.
    test : str
        Statistical test name for statannotations.
    comparison_text_format : str
        Format for significance (e.g., "star" or "simple").
    log_scale_y : bool
        If True, use log-scale for y-axis (neglog10 values).
    showplot : bool
        Whether to display the plot (plt.show()).
    return_fig_ax : bool
        If True, return fig, ax. If False, return nothing.

    Returns
    -------
    (fig, ax) if return_fig_ax else None
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    # Only plot if there's more than one value (otherwise barplot/statannotations will fail)
    exon_num_dist = ridge_df_annot[x].dropna().astype(int)
    present_values = sorted(exon_num_dist.unique())
    if len(present_values) <= 1:
        print(f"Not enough {x} groups for comparison.")
        return (None, None) if return_fig_ax else None
    else:
        # Assign NaN for missing values, convert to int only for valid ones, and to string
        col_str = f"{x}_str"
        ridge_df_annot[col_str] = ridge_df_annot[x].apply(
            lambda x: str(int(x)) if pd.notnull(x) else np.nan
        )
        # Only keep categories that are present and order appropriately
        categories = [str(x) for x in present_values]
        ridge_df_annot[col_str] = pd.Categorical(
            ridge_df_annot[col_str], categories=categories, ordered=True
        )

        # Statannotations imports (local to function)
        import statannotations
        from statannotations.Annotator import Annotator

        fig, ax = plt.subplots(figsize=figsize)

        neglog10_col = f"neglog10_{y}"
        ridge_df_annot[neglog10_col] = -np.log10(ridge_df_annot[y])

        # Barplot
        sns.barplot(
            data=ridge_df_annot,
            x=x,
            y=neglog10_col,
            ax=ax
        )
        ax.set_ylabel(f"-log10({y})")
        ax.set_xlabel(x)
        if log_scale_y:
            ax.set_yscale('log')

        # Build comparison pairs
        pairs = []
        if only_nearest_neighbor_comparisons:
            for i in range(len(categories)-1):
                pairs.append((categories[i], categories[i+1]))
        else:
            import itertools
            pairs = list(itertools.combinations(categories, 2))

        if len(pairs) > 0:
            annotator = Annotator(
                ax=ax,
                pairs=pairs,
                data=ridge_df_annot,
                x=x,
                y=neglog10_col
            )
            annotator.configure(
                test=test,
                text_format=comparison_text_format,
                comparisons_correction=correction_method
            )
            annotator.apply_and_annotate()

        plt.tight_layout()
        if showplot:
            plt.show()
        if return_fig_ax:
            return fig, ax
        return None



def plot_marginal_effect_by_region(
    ridge_df_annot, 
    y_var="interaction_strength",
    title="Marginal Effect Size by Annotation Overlap",
    x_label="Annotation Overlap",
    y_label="Marginal Effect Size",
    exclude_annot=None,
    include_annot=None,  # <--- NEW PARAMETER: list of annotation short names to include
    height=2,
    aspect=0.8,
    col_wrap=3,
    palette="binary",
    save_path=None,
    show=True,
    legend=True,
    show_xticklabels=True,
    additional_melt_id_vars=None,
    legend_title=None,  # <--- NEW PARAMETER
    legend_loc='upper center',  # <--- NEW PARAMETER: legend location
    legend_bbox_to_anchor=None,  # <--- NEW PARAMETER: legend bbox_to_anchor (x, y)
    legend_ncol=None,  # <--- NEW PARAMETER: number of columns in legend
    title_height=None,  # <--- NEW PARAMETER: controls suptitle y position
    y_label_pad=0.02,  # <--- NEW PARAMETER: controls padding between y-axis label and plots
    use_abs=False,  # <--- NEW PARAMETER: use absolute value of y_var
    subplot_title_size='small',  # <--- NEW PARAMETER: font size for subplot titles
    facets_as_ylabels=False,  # <--- NEW PARAMETER: use facet titles as y-axis labels instead
    center_ylabel=True,  # <--- NEW PARAMETER: center a single shared y-axis label on the figure
    facet_title_newlines=False,  # <--- NEW PARAMETER: split facet titles onto multiple lines by spaces
    facet_xy_paired=False,  # <--- NEW PARAMETER: pair facets into row/col labels instead of per-facet titles
    x_ticklabel_rotation=None,  # <--- NEW PARAMETER: rotate x-axis tick labels on bottom row
):
    """
    Plots marginal effect size by annotation region using a faceted barplot with statistical annotation.

    Parameters
    ----------
    ridge_df_annot : pd.DataFrame
        Annotation dataframe containing at least interaction_strength, and one-hot annotation columns.
    y : str, optional
        Column name to plot on y-axis.
    exclude_annot : list of str, optional
        List of regions (short names) to exclude from plot.
    include_annot : list of str, optional
        List of annotation short names to include. If None, includes all available annotations
        (after applying exclude_annot filter). If specified, only these annotations will be included.
        The order of subplots (top to bottom, left to right) will match the order in this list.
        Uses short names from REGION_ANNOTATION_MAP (e.g., "5ss_can", "3ss_can", "exon_core").
    aspect : float, optional
        Aspect ratio for seaborn facet grid.
    col_wrap : int, optional
        Number of columns/rows in facet grid.
    height : float, optional
        Height of facet grid plots.
    palette : str or list, optional
        Color palette to use.
    save_path : str, optional
        Path to save figure.
    show : bool, optional
        Whether to show the plot (default True).
    legend : bool, optional
        Whether to show the legend horizontally on top (default True).
    show_xticklabels : bool, optional
        Whether to show the x-axis tick labels (default True).
    additional_melt_id_vars : list of str, optional
        Additional columns to keep as id_vars in melt.
    legend_title : str, optional
        Title for the legend (default is None, which uses auto-title or no title).
    legend_loc : str or tuple, optional
        Location of the legend (default 'upper center'). Can be a string like 'upper center',
        'upper left', 'best', etc., or a tuple of coordinates.
    legend_bbox_to_anchor : tuple, optional
        Bbox_to_anchor for the legend in figure coordinates (default is None, which sets (0.5, 0.98)
        when legend_loc is 'upper center', otherwise uses default matplotlib behavior).
    legend_ncol : int, optional
        Number of columns in the legend (default is None, which uses the number of labels).
    title_height : float, optional
        Y position of the suptitle (default is None, which sets 0.98 if legend else 0.99).
        Values are in figure coordinates (0-1).
    y_label_pad : float, optional
        X position (padding) of the y-axis label in figure coordinates (default 0.02).
        Increase to add more space between the label and the plots.
    use_abs : bool, optional
        If True, use the absolute value of y_var for plotting (default False).
    subplot_title_size : str or float, optional
        Font size for subplot titles (default 'small'). Can be a string like 'small', 'medium', 'large'
        or a numeric value.
    facets_as_ylabels : bool, optional
        If True, move each facet title (region label with count) to be the y-axis label of that facet,
        and clear the facet titles. This is useful for stacked/one-column layouts where you prefer
        region labels along the y-axis instead of above each facet.
    center_ylabel : bool, optional
    facet_title_newlines : bool, optional
        If True, replace spaces in the region name part of each facet title with newline characters,
        so that multi-word region names (e.g., "3ss intronic proximal") are rendered one word per line.
        The "(n=...)" count is kept on its own line below.
    facet_xy_paired : bool, optional
        If True, suppress per-facet titles and instead derive column and row labels from the region
        names (e.g., "3′ss canonical" -> column label "3′ss", row label "canonical"). Column labels
        are drawn centered above each facet column, and row labels are drawn vertically on the
        right-most facet of each row. Intended for small, regular grids (e.g., 2×2) where regions
        can be interpreted as a Cartesian product of a few row/column categories.
    x_ticklabel_rotation : float, optional
        If provided, rotate the visible bottom-row x-axis tick labels by this many degrees and adjust
        their alignment (right-aligned for positive angles, left-aligned for negative, centered for 0).
        If True (default), draw a single shared y-axis label centered on the figure using ``y_label``.
        If False and ``facets_as_ylabels`` is also False, place the same ``y_label`` on the left-most
        facet of each row (i.e., each facet in the left-most column) instead of a global centered label.

    Returns
    -------
    g : sns.FacetGrid
        The created FacetGrid object.
    """
    from statannotations.Annotator import Annotator
    if exclude_annot is None:
        exclude_annot = ["MANE introns","MANE exons","5′ UTR","3′ UTR"]
    if additional_melt_id_vars is None:
        additional_melt_id_vars = []
    # Apply absolute value if requested
    working_df = ridge_df_annot.copy()
    if use_abs:
        working_df[y_var] = working_df[y_var].abs()
    
    # Melt the data to long format for easier plotting
    # Determine which annotations to include and their order
    if include_annot is not None:
        # Use only the specified annotations (convert short names to full names)
        # Preserve the order from include_annot
        annotation_values = [REGION_ANNOTATION_MAP[key] for key in include_annot if key in REGION_ANNOTATION_MAP]
    else:
        # Include all annotations from REGION_ANNOTATION_MAP
        annotation_values = list(REGION_ANNOTATION_MAP.values())
    
    # Filter by exclude_annot if specified
    if exclude_annot is not None:
        excluded_set = set(exclude_annot)
        annotation_values = [x for x in annotation_values if x not in excluded_set]
    
    # Get region columns that exist in the dataframe, preserving order
    region_columns = []
    available_annotation_values = []
    for x in annotation_values:
        col_name = "clinical_" + x
        if col_name in working_df.columns:
            region_columns.append(col_name)
            available_annotation_values.append(x)
    
    # Rebuild order_mapping based on what's actually available in the dataframe
    if include_annot is not None:
        order_mapping = {val: i for i, val in enumerate(available_annotation_values)}
    else:
        order_mapping = None
    
    melted = working_df.melt(
        id_vars=[y_var] + additional_melt_id_vars,
        value_vars=region_columns,
        var_name="region",
        value_name="in_region"
    )
    melted["region"] = melted["region"].str.replace("clinical_", "")
    
    # Sort by custom order if provided, otherwise by region name descending
    if order_mapping is not None:
        # Create a sort key based on the order mapping
        melted["_sort_order"] = melted["region"].map(order_mapping).fillna(float('inf'))
        melted.sort_values(by="_sort_order", ascending=True, inplace=True)
        melted.drop(columns=["_sort_order"], inplace=True)
    else:
        melted.sort_values(by="region", ascending=False, inplace=True)

    # Compute mean and standard error for each region/in_region group
    agg_df = melted.groupby(["region", "in_region"])[y_var].agg(['mean', 'count', 'std']).reset_index()
    agg_df["in_region"] = agg_df["in_region"].astype(bool).astype(str)
    agg_df["sem"] = agg_df["std"] / agg_df["count"].pow(0.5)
    
    # CRITICAL: Convert in_region to categorical with explicit order to ensure True comes before False
    # This matches statannotations order=[True, False] where True is first (left), False is second (right)
    agg_df["in_region"] = pd.Categorical(agg_df["in_region"], categories=["True", "False"], ordered=True)
    
    # Sort agg_df by the same order as melted
    if order_mapping is not None:
        agg_df["_sort_order"] = agg_df["region"].map(order_mapping).fillna(float('inf'))
        # Sort by region order, then by in_region (True before False due to categorical order)
        agg_df.sort_values(by=["_sort_order", "in_region"], ascending=[True, True], inplace=True)
        agg_df.drop(columns=["_sort_order"], inplace=True)
    else:
        # Original behavior: sort by region descending, then in_region (True before False)
        agg_df.sort_values(by=["region", "in_region"], ascending=[False, True], inplace=True)

    # --- Add count info for facet titles ---
    region_counts = agg_df.loc[agg_df["in_region"] == "True"].set_index("region")["count"].to_dict()
    region_with_count = {}
    for k, cnt in region_counts.items():
        region_label = k
        if facet_title_newlines:
            # Replace spaces in the region label with newlines, so each word appears on its own line
            region_label = region_label.replace(" ", "\n")
        region_with_count[k] = f"{region_label}\n(n={cnt})"
    agg_df["region_count"] = agg_df["region"].map(region_with_count)
    melted["region_count"] = melted["region"].map(region_with_count)

    g = sns.catplot(
        data=agg_df,
        x="in_region",
        y="mean",
        hue="in_region",
        kind="bar",
        col="region_count",
        palette=palette,
        col_wrap=col_wrap,
        sharey=True,
        sharex=True,
        height=height,
        aspect=aspect,
        legend=False,  # Disable automatic legend, we'll create our own
        errorbar=None,
        edgecolor="black",
        linewidth=1,
        # Order is now enforced by categorical type, but explicitly set for safety
        order=["True", "False"],  # True first (left), False second (right) - matches statannotations
        hue_order=["True", "False"],
    )

    # Add error bars manually
    axes_flat = [ax for ax in g.axes.flatten() if ax is not None]
    # Number of facets (one per unique region_count)
    grouped_regions = list(agg_df.groupby("region_count"))
    n_facets = len(grouped_regions)

    # Determine which facet indices are on the bottom row based purely on facet index
    # and col_wrap. Seaborn fills facets row-wise, so the bottom row consists of the
    # last `col_wrap` facets (or fewer if n_facets < col_wrap).
    if col_wrap is None or col_wrap <= 0:
        # No wrapping: single row, all facets are "bottom"
        bottom_start = 0
    else:
        bottom_start = max(0, n_facets - col_wrap)

    for idx, (ax, (region, subdf)) in enumerate(zip(axes_flat, grouped_regions)):
        for i, (idx, row) in enumerate(subdf.iterrows()):
            bars = ax.patches
            bar_idx = i
            if bar_idx < len(bars):
                bar = bars[bar_idx]
                x = bar.get_x() + bar.get_width() / 2
                y = bar.get_height()
                ax.errorbar(
                    x, y, yerr=row["sem"], fmt='none', c='black', capsize=4, lw=1.5
                )
        # (Defer x tick label visibility/rotation to a post-processing step below.)

    # Stat annotation: compare in_region True vs False per region
    for ax, region in zip(g.axes.flatten(), agg_df["region_count"].unique()):
        region_data = melted[melted["region_count"] == region]
        n_true = (region_data["in_region"] == True).sum()
        n_false = (region_data["in_region"] == False).sum()
        if n_true > 1 and n_false > 1:
            pairs = [(True, False)]
            y_offset = 0.05 / aspect if aspect > 0 else 0.05
            annotator = Annotator(
                ax=ax,
                pairs=pairs,
                data=region_data,
                x="in_region",
                y=y_var,
                order=[True, False],
            )
            annotator.configure(
                test='t-test_ind',
                text_format='star',
                loc='inside',
                verbose=0,
                line_offset_to_group=y_offset,
            )
            annotator.apply_and_annotate()

    # After all drawing/stat annotations, enforce x tick label visibility:
    # Only the bottom row (facet indices >= bottom_start) should display
    # tick labels when enabled.
    for idx, ax in enumerate(axes_flat):
        is_bottom = idx >= bottom_start
        if not show_xticklabels or not is_bottom:
            ax.tick_params(axis='x', labelbottom=False)
        else:
            ax.tick_params(axis='x', labelbottom=True)
            if x_ticklabel_rotation is not None:
                # Reduce distance between rotated labels and axis by ~half
                ax.tick_params(axis='x', pad=2)
                for label in ax.get_xticklabels():
                    label.set_rotation(x_ticklabel_rotation)
                    # Alignment: for ±90°, center; otherwise right for positive,
                    # left for negative, centered for 0.
                    if abs(x_ticklabel_rotation) == 90:
                        label.set_ha('center')
                        label.set_va('top')
                    elif x_ticklabel_rotation > 0:
                        label.set_ha('right')
                        label.set_va('top')
                    elif x_ticklabel_rotation < 0:
                        label.set_ha('left')
                        label.set_va('top')
                    else:
                        label.set_ha('center')
                        label.set_va('top')

    # Either keep facet titles above each subplot, convert them to y-axis labels,
    # or replace them with paired row/column labels.
    if facets_as_ylabels:
        # For each axis, take its current title (the region label with count),
        # clean off the seaborn "region_count = " prefix, move it to the y-axis
        # label, and clear the facet title.
        for ax in g.axes.flatten():
            current_title = ax.get_title()
            if current_title:
                # seaborn typically formats as "region_count = <label>"
                if "=" in current_title:
                    current_title = current_title.split("=", 1)[1].strip()
                ax.set_ylabel(current_title, fontsize=subplot_title_size)
            ax.set_title("")
            # Always clear the per-axis x label so only tick labels show
            ax.set_xlabel("")
    else:
        # Standard behavior: control how facet titles and y-axis labelling work
        g.set_axis_labels("", "")

        axes_array = np.asarray(g.axes)
        axes_flat = [ax for ax in axes_array.ravel() if ax is not None]

        if facet_xy_paired:
            # Suppress per-facet titles; we'll draw row/column labels instead.
            for ax in axes_flat:
                ax.set_title("")

            # Reconstruct the facet order and corresponding base region names.
            # region_count labels appear as column names in FacetGrid in the same
            # order as agg_df["region_count"].unique().
            facet_labels = list(agg_df["region_count"].unique())
            # Invert region_with_count to get base region name from label.
            label_to_region = {v: k for k, v in region_with_count.items()}
            base_regions = [label_to_region.get(lbl, lbl) for lbl in facet_labels]

            # Infer grid shape from col_wrap and number of facets
            n_facets = len(base_regions)
            if col_wrap is None or col_wrap <= 0:
                n_cols = n_facets
            else:
                n_cols = min(col_wrap, n_facets)

            # Derive column labels from the first row of facets
            col_labels = []
            for col_idx in range(n_cols):
                if col_idx >= n_facets:
                    break
                region_name = base_regions[col_idx]
                parts = region_name.split(" ", 1)
                col_label = parts[0]
                col_labels.append(col_label)

            # Derive row labels from the first facet in each row
            n_rows = (n_facets + n_cols - 1) // n_cols
            row_labels = []
            for row_idx in range(n_rows):
                facet_idx = row_idx * n_cols
                if facet_idx >= n_facets:
                    break
                region_name = base_regions[facet_idx]
                parts = region_name.split(" ", 1)
                row_label = parts[1] if len(parts) > 1 else parts[0]
                row_labels.append(row_label)

            # Draw column labels centered above the top row of facets
            for col_idx, col_label in enumerate(col_labels):
                if col_idx >= len(axes_flat):
                    break
                # Axis for this column in the first row
                ax = axes_flat[col_idx]
                pos = ax.get_position()
                x = pos.x0 + pos.width / 2.0
                # Add padding proportional to axis height so spacing scales with figure size
                y = pos.y1 + 0.2 * pos.height
                g.fig.text(x, y, col_label, ha="center", va="bottom", fontsize=subplot_title_size)

            # Draw row labels vertically on the right side of the last facet in each row
            for row_idx, row_label in enumerate(row_labels):
                start_idx = row_idx * n_cols
                end_idx = min(start_idx + n_cols, n_facets)
                if end_idx <= start_idx:
                    continue
                # Last facet in this row
                ax = axes_flat[end_idx - 1]
                pos = ax.get_position()
                # Add padding proportional to axis width so spacing scales with figure size
                x = pos.x1 + 0.2 * pos.width
                y = pos.y0 + pos.height / 2.0
                g.fig.text(
                    x,
                    y,
                    row_label,
                    ha="left",
                    va="center",
                    rotation=270,
                    fontsize=subplot_title_size,
                )
        else:
            # Default: each facet keeps its own title
            g.set_titles("{col_name}", size=subplot_title_size)

        # Handle y-axis labelling (global vs per-row left column)
        if y_label is not None:
            if center_ylabel:
                # Single shared centered y-axis label
                g.fig.text(y_label_pad, 0.5, y_label, va='center', rotation='vertical')
            else:
                # Put the same y-label on the left-most column facets instead.
                axes_all = [ax for ax in axes_array.ravel() if ax is not None]
                # Clear any existing per-axis y labels
                for ax in axes_all:
                    ax.set_ylabel("")
                if axes_all:
                    # Determine left-most column via axes positions
                    x0_vals = [ax.get_position().x0 for ax in axes_all]
                    min_x0 = min(x0_vals)
                    tol = 5e-3
                    left_axes = [ax for ax in axes_all if abs(ax.get_position().x0 - min_x0) < tol]
                else:
                    left_axes = []
                for ax in left_axes:
                    ax.set_ylabel(y_label)

        # Regardless of center_ylabel, only keep y tick labels on the left-most column;
        # hide them on other facets to reduce visual clutter.
        axes_all = [ax for ax in axes_array.ravel() if ax is not None]
        if axes_all:
            x0_vals = [ax.get_position().x0 for ax in axes_all]
            min_x0 = min(x0_vals)
            tol = 5e-3
            left_axes = [ax for ax in axes_all if abs(ax.get_position().x0 - min_x0) < tol]
            for ax in axes_all:
                if ax in left_axes:
                    ax.tick_params(axis='y', labelleft=True)
                else:
                    ax.tick_params(axis='y', labelleft=False)

    # Always keep the shared x-label at the bottom if provided, centered between
    # the left-most and right-most facet columns rather than the full figure.
    if x_label is not None:
        axes_all = [ax for ax in np.asarray(g.axes).ravel() if ax is not None]
        if axes_all:
            min_x = min(ax.get_position().x0 for ax in axes_all)
            max_x = max(ax.get_position().x1 for ax in axes_all)
            x_center = (min_x + max_x) / 2.0
        else:
            x_center = 0.5
        g.fig.text(x_center, .01, x_label, ha='center', va='center')
    # Adjust suptitle position to leave room for legend at top and keep it centered
    if title_height is None:
        # Set reasonable default based on whether legend is shown
        title_height = 0.98 if legend else 0.99
    g.fig.suptitle(title, y=title_height, x=0.5, ha="center")

    # Move the legend horizontally to the top
    # Remove any automatic legend from axes first
    for ax in g.axes.flatten():
        if hasattr(ax, "legend_") and ax.legend_ is not None:
            ax.legend_.remove()
    
    # Get handles and labels from plot elements directly
    # Since we're using hue="in_region", we need to get the unique values and their corresponding patches
    first_ax = g.axes.flatten()[0]
    patches = first_ax.patches
    
    # Get unique in_region values in the CORRECT order: True before False
    # Use categorical categories if available, otherwise explicitly order as ["True", "False"]
    if hasattr(agg_df["in_region"].dtype, 'categories'):
        # Categorical type - use its categories order
        unique_in_region = list(agg_df["in_region"].dtype.categories)
    else:
        # Not categorical - explicitly order as True, False
        unique_in_region = ["True", "False"]
        # Filter to only include values that actually exist
        existing_values = set(agg_df["in_region"].unique())
        unique_in_region = [val for val in unique_in_region if val in existing_values]
    
    # Create handles and labels from patches
    # Each facet has bars for each in_region value, so we take the first occurrence of each
    if patches and len(unique_in_region) > 0:
        # Get the first bar for each unique in_region value
        handles = []
        labels = []
        for i, in_region_val in enumerate(unique_in_region):
            if i < len(patches):
                handles.append(patches[i])
                labels.append(str(in_region_val))
    
    # Adjust layout to make room for legend at top
    if legend:
        plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave top 4% for legend
    else:
        plt.tight_layout()
    
    if handles and legend:
        # Create legend with user-specified positioning
        legend_kwargs = {
            'handles': handles,
            'labels': labels,
            'loc': legend_loc,
            'ncol': legend_ncol if legend_ncol is not None else len(labels),
            'frameon': False,
        }
        # Set bbox_to_anchor if provided, or use default for 'upper center'
        if legend_bbox_to_anchor is not None:
            legend_kwargs['bbox_to_anchor'] = legend_bbox_to_anchor
        elif legend_loc == 'upper center':
            # Default behavior: center at top
            legend_kwargs['bbox_to_anchor'] = (0.5, 0.98)
        # Add title if provided
        if legend_title is not None:
            legend_kwargs['title'] = legend_title
        
        g.fig.legend(**legend_kwargs)
    
    if save_path is not None:
        g.savefig(save_path, **utils.FIG_SAVE_KWARGS)
    if show:
        plt.show()
    return g