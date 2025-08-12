import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

import src.utils as utils

def _add_legend(g, 
                palette=utils.get_clinsig_palette(),
                 loc='upper center',
                 bbox_to_anchor=(0.5, 1.05),
                 top=0.9,
                 ncol=None):
    
    handles = [plt.Rectangle((0,0),1,1, color=palette[label]) for label in palette]
    labels = list(palette.keys())
    if ncol is None:    
        ncol = len(palette)
    g.figure.legend(handles, labels, 
                 loc=loc,
                 bbox_to_anchor=bbox_to_anchor,
                 ncol=ncol) 
    g.figure.subplots_adjust(top=top)  # Adjust spacing for titles 



def summarise_sites(vep_df,
                       gene_col='GENE',
                       site_col='site',
                       clinsig_col='CLNSIG_simple'):
    
    vep_df['gene_site'] = vep_df[gene_col] + '_' + vep_df[site_col]
    mutant_summary = vep_df.groupby(clinsig_col)['gene_site'].nunique() 
    mutant_summary_str = ', '.join([f"{k}: {v}" for k,v in mutant_summary.items()]) 
    return mutant_summary_str

def summarise_title(vep_df,
                     label_cols = ['GENE', 'site',"sample"]):
    
    labels = {}
    for col in label_cols:
        if col not in vep_df.columns:
            label_cols.remove(col)
            continue
        if vep_df[col].nunique() > 1:
            labels[col] = f"{col}: {vep_df[col].nunique()}"
        else:
            labels[col] = f"{col}: {vep_df[col].tolist()[0]}"
    return ', '.join([labels[col] for col in label_cols if col in labels])


def estimate_runtime(vep_df,
                     total_sites,
                     model_name = "flashzoi",
                     time_col = "time_total",
                     n_gpus = 1):
    

    time_df =  vep_df.loc[vep_df["slot"]==time_col]
    seconds_per_site = time_df.groupby("site")[model_name].sum().mean()
    print(f"{seconds_per_site/60:.2f} minutes per site")
    print(f"Number of days it should take to run all {total_sites} sites genome-wide:\n{seconds_per_site * total_sites / 60 / 60 / 24 / n_gpus:.2f}")


def add_onekg_metadata(vep_df,
                       sample_col = "sample",
                       cols=None,
                       metadata_df = None,
                       how = "left"):
    
    if metadata_df is None:
        import src.onekg as og
        metadata_df = og.get_sample_metadata()
    if cols is None:
        cols = metadata_df.columns

    if all(col in vep_df.columns for col in cols):
        print(f"All cols already in vep_df, skipping")
    else:
        vep_df = vep_df.merge(metadata_df[cols],
                            left_on=sample_col, 
                            right_on="Individual ID", 
                            how=how)
        vep_df.loc[vep_df["sample"]=="REF", "Super Population"] = "REF"
        vep_df.loc[vep_df["sample"]=="REF", "Population"] = "REF"
    return vep_df


def plot_violin(df,
                x="CLNSIG_simple",
                y="flashzoi",
                hue="CLNSIG_simple",
                row=None,#"GENE",
                col="slot",
                palette=utils.get_clinsig_palette(),
                cut=0,
                height=4,
                aspect=1,
                sharey=False,
                sharex=True, 
                **kwargs):
    # Filter for delta metrics and create violin plot with facets

    g = sns.FacetGrid(df, 
                    row=row, 
                    col=col,
                    height=height, 
                    aspect=aspect, 
                    sharey=sharey, 
                    sharex=sharex, 
                    margin_titles=True)
    g.map_dataframe(sns.violinplot, 
                    x=x, 
                    y=y, 
                    hue=hue,
                    palette=palette,
                    cut=cut,
                    **kwargs)
    g.fig.suptitle(summarise_title(df)+"\n"+ summarise_sites(df), y=1.02)
    g.set_titles(row_template="{row_name}", col_template=
                 "{col_name}")
    for ax in g.axes.flat:
        plt.setp(ax.get_xticklabels(), rotation=45)
    plt.tight_layout()
    plt.show()

def plot_kde(df,
             x="flashzoi",
             hue="CLNSIG_simple",
             row="slot",
             agg_vars=None,
             agg_func="mean",
             col=None, 
             height=3,
             aspect=2,
             alpha=.75,
             fill=True,
             legend=True, 
             sharey=False,
             sharex=False,
             **kwargs):
    # Filter for delta metrics and create 2D density plot with facets

    # Aggregate data to make plotting much faster
    if agg_vars is not None:
        df = df.groupby(agg_vars).agg({x: agg_func}).reset_index()

    g = sns.FacetGrid(df, 
                    row=row, 
                    col=col,
                    height=height, 
                    aspect=aspect, 
                    sharey=sharey, 
                    sharex=sharex)
    g.map_dataframe(sns.kdeplot, 
                    x=x, 
                    hue=hue,
                    palette=utils.get_clinsig_palette(),
                    alpha=alpha,
                    fill=fill,  # Fill the KDE plot
                    legend=legend,
                    **kwargs)
    g.set_titles("{row_name}")
    g.fig.suptitle(summarise_title(df)+"\n"+ summarise_sites(df), y=1.02)
    plt.tight_layout()

    # Add legend with explicit legend handles
    handles, labels = g.axes[0,0].get_legend_handles_labels()
    g.fig.legend(handles, labels, title="Clinical Significance", 
                bbox_to_anchor=(1.05, 0.5), loc='center left')
    plt.show()
    

def summary_histograms(vep_df,
                       slot_col="slot", 
                       x="flashzoi",
                       height=2,
                       aspect=2, 
                       col_wrap=2,
                       margin_titles=True,
                       sharex=False,
                       sharey=False, 
                       sample_size=1000):

    # Create a figure with subplots for each slot
    g = sns.FacetGrid(vep_df.groupby(slot_col).sample(sample_size).sort_values(slot_col),
                    col=slot_col,  # Changed from row to col
                    col_wrap=col_wrap,  # Added col_wrap to wrap facets
                    height=height, 
                    aspect=aspect, 
                    margin_titles=margin_titles,
                    sharex=sharex, 
                    sharey=sharey)
    g.map_dataframe(sns.histplot, x=x)
    g.set_titles("{col_name}")
    g.fig.supxlabel("Metric")
    plt.tight_layout()

def get_mondo_within_site_var(vep_df,
                              groupby_cols= ["CLNSIG_simple",
                                             "GENE", 
                                             "CLNHGVS",
                                             "RS",
                                             "site",
                                             "slot"],
                            vep_col = "VEP",
                            variant_col = "site", 
                            haplotype_col = "haplotype",
                            input_col="CLNDISDB",
                            search_terms=["MONDO"]
                            ):
    import src.owlready2 as OWL

    vep_df = vep_df.copy()

    vep_df = _add_haplotype_col(vep_df)

    groupby_cols = [x for x in groupby_cols if x in vep_df.columns]

    group_col = "MONDO"
    split_col = group_col+"_split"
    label_col = group_col+"_label"
    
    if split_col not in vep_df.columns or group_col not in vep_df.columns:
        vep_df = extract_id_cols(vep_df,
                                input_col=input_col,
                                search_terms=search_terms)
    
        vep_df.loc[:,split_col] = vep_df[group_col].apply(lambda x: [m.replace("MONDO:MONDO:", "MONDO:") for m in x] if isinstance(x, list) else [])
        vep_df.loc[:,group_col] = vep_df[split_col].str.join("|")


    if split_col not in vep_df.columns:
        vep_df.loc[:,split_col] = vep_df[group_col].str.split("|")

    groupby_cols = [group_col,split_col] + groupby_cols
    within_site_var = vep_df.explode(split_col).reset_index(drop=True).groupby(groupby_cols).agg({vep_col:"var", 
                                                                                                  haplotype_col:"nunique"}
                                                                                                  ).reset_index().sort_values(by=vep_col, ascending=False)
    
    onto = OWL.get_onto_mondo()
    id_map = OWL.get_id_map(onto)
    within_site_var.loc[:,label_col] = within_site_var[split_col].map(id_map)


    within_site_var_mean = within_site_var.groupby([split_col,label_col]).agg({vep_col:"mean", 
                                                                               variant_col:"nunique", 
                                                                               haplotype_col:"unique"}
                                                                               ).sort_values(by=vep_col, ascending=False).reset_index()
    within_site_var_mean.loc[:,haplotype_col] = within_site_var_mean[haplotype_col].apply(lambda x: x[0] if len(x) > 0 else None)
    
    return within_site_var, within_site_var_mean


def _add_haplotype_col(df):
    if "haplotype" not in df.columns:
        df.loc[:, "haplotype"] = df["sample"] + "_" + df["ploid"]
    return df

def plot_vep_by_superpop(vep_df,
                         within_site_var,
                         i=0,
                         vep_col = "flashzoi",
                         variant_col = "site",
                         clinsig_col = "clinsig",
                         gene_col = "GENE",
                         metric_col = "slot",
                         haps_to_samples=None,
                         unique_haplotypes: bool = True,
                         remove_zeros: bool = False,
                         figsize=(10, 8),
                         hue_top: str = "mutant",
                         hue: str = "mutant",
                         hue_bottom: str = "mutant",
                         legend_loc: str = "upper left",
                         binwidth_scaler: float = 200,
                         add_clinsig_labels: bool = True,
                         palette: str = "Set2"):
    

    if "is_ref" not in vep_df.columns:
        vep_df.loc[:, "is_ref"] = vep_df["sample"]=="REF"

    vep_df = _add_haplotype_col(vep_df)
   
    if i is not None:
        row_selected = within_site_var.drop_duplicates(
            subset=[vep_col]
        ).iloc[i]
        plot_df = vep_df.loc[(vep_df[variant_col]==row_selected[variant_col]) & (vep_df[gene_col]==row_selected[gene_col])]
    else:
        plot_df = vep_df.copy()
        
 
    if remove_zeros:
        plot_df = plot_df.loc[plot_df[vep_col]!=0]
    multi_mutant = plot_df[variant_col].nunique() > 1
    mutant_palette = utils.make_palette(plot_df[variant_col].unique(), 
                                        palette=palette)
    if hue==clinsig_col:
        cmap = utils.get_clinsig_palette()
    elif hue=="Super Population":
        cmap = utils.get_superpop_palette()
    elif hue==variant_col:
        cmap = mutant_palette
    else:
        raise ValueError(f"Invalid hue: {hue}")
    
    # Create a palette for the top subplot
    if hue_top==clinsig_col:
        cmap_top = utils.get_clinsig_palette()
    elif hue_top=="Super Population":
        cmap_top = utils.get_superpop_palette()
    elif hue_top==variant_col:
        cmap_top = mutant_palette
    else:
        raise ValueError(f"Invalid hue_top: {hue_top}")
    
    if hue_bottom==clinsig_col:
        cmap_bottom = utils.get_clinsig_palette()
    elif hue_bottom=="Super Population":
        cmap_bottom = utils.get_superpop_palette()
    elif hue_bottom==variant_col:
        cmap_bottom = mutant_palette
    else:
        raise ValueError(f"Invalid hue_bottom: {hue_bottom}")
    

    def ylabeler(hue):
        if hue==variant_col:
            return "Variant"
        elif hue==clinsig_col:
            return "ClinSig"
        elif hue=="Super Population":
            return "Superpop"
        else:
            return hue

    
    plot_df = add_onekg_metadata(plot_df)

    # Calculate global min and max for consistent x-axis limits
    x_min = plot_df[vep_col].min()
    x_max = plot_df[vep_col].max()
    # Compute optimal binwidth
    # bins=100
    binwidth=(x_max-x_min)/binwidth_scaler

    # Get unique super populations excluding REF
    super_pops = plot_df.loc[plot_df["is_ref"]==False].dropna(subset=["Super Population"])["Super Population"].unique()
    n_pops = len(super_pops)

    # Create figure with subplots - adjust grid size based on number of populations
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(nrows=n_pops + 3, 
                          ncols=1, 
                          height_ratios= [2]+[1]+([1] * (n_pops + 1)), 
                          hspace=0.1,
                          top=0.9)
 
    # Add histogram with all data in first subplot
    ax0 = fig.add_subplot(gs[0])
    
    # Add REF lines and labels first
    ref_rows = plot_df.loc[plot_df["is_ref"]==True].drop_duplicates(subset=[variant_col])
    label_map = {"path":"P", "benign":"B", "likely_path":"LP", "likely_benign":"LB"}
    # Create a new subplot for labels above the histogram
    ax_labels = fig.add_subplot(gs[0])
    ax_labels.set_axis_off()  # Hide the axis
    for _, row in ref_rows.iterrows():
        ref_value = row[vep_col]
        ax0.axvline(x=ref_value, color=mutant_palette[row[variant_col]], linestyle='--', label=row[variant_col])
        label = row[variant_col] + " (" + label_map[row[clinsig_col]] + ")" if add_clinsig_labels else row[variant_col]
        # Position text in the label subplot above the histogram
        ax_labels.text(ref_value - 0.01, ax0.get_ylim()[1], 
                      label, 
                      color=mutant_palette[row[variant_col]], 
                      rotation=90, 
                      va='bottom', 
                      ha='right',
                      transform=ax0.transData)  # Use the data coordinates from ax0
        
    # Now add the main histogram showing each mutant's distribution
    if multi_mutant:
        sns.histplot(data=plot_df, 
                     x=vep_col, 
                     hue=hue_top, 
                     multiple="stack", 
                     legend=False,
                     ax=ax0, 
                     palette=cmap_top,
                     binwidth=binwidth
                     )
    else:
        sns.histplot(plot_df, 
                    x=vep_col, 
                    binwidth=binwidth,
                    color="white",
                    ax=ax0)
    ax0.set_xlabel(None)
    ax0.set_xticklabels([])  # Remove x-tick labels
    # Adjust legend location
    if legend_loc=="upper left":
        ax0.text(0.02, 0.95, "All Populations", transform=ax0.transAxes, ha='left', va='top')
    else:
        ax0.text(0.98, 0.95, "All Populations", transform=ax0.transAxes, ha='right', va='top')
    ax0.set_xlim(x_min, x_max)
    
    if not multi_mutant:
        # Add mean line
        ax0.axvline(x=plot_df[vep_col].mean(), color='grey', linestyle=':', label='Mean', linewidth=1)
        ax0.text(plot_df[vep_col].mean() - 0.01, ax0.get_ylim()[1], f'Mean', color='grey', rotation=90, va='top', ha='right')
        # Add median line
        ax0.axvline(x=plot_df[vep_col].median(), color='grey', linestyle='--', label='Median', linewidth=1)
        ax0.text(plot_df[vep_col].median() - 0.01, ax0.get_ylim()[1], f'Median', color='grey', rotation=90, va='top', ha='right')


     # Add summary histogram with all superpopulations, colored by mutant
    ax1 = fig.add_subplot(gs[1])
    sns.histplot(plot_df.loc[plot_df["Super Population"]!="REF"], 
                    x=vep_col, 
                    # bins=bins,
                    binwidth=binwidth*4,
                    hue=hue_top,
                    palette=cmap_top,
                    legend=False,
                    multiple="fill",
                    ax=ax1)
    # ax1.text(0.98, 0.95, "All Populations", transform=ax1.transAxes, ha='right', va='top')
    ax1.set_xlabel(f"Variant Effect Prediction ({vep_col})")
    ax1.set_ylabel(f"Proportion\nby {ylabeler(hue_top)}")
    ax1.set_xlim(x_min, x_max)


    # Add faceted histograms for each superpopulation
    for idx, pop in enumerate(sorted(super_pops)):
        ax = fig.add_subplot(gs[idx+2])
        sns.histplot(plot_df.loc[plot_df["Super Population"]==pop], 
                    x=vep_col, 
                    # bins=bins,
                    binwidth=binwidth,
                    hue=hue,
                    palette=cmap,
                    legend=True if hue=="Super Population" else False,
                    ax=ax)
        if hue==variant_col:
            ax.text(0.02, 0.95, pop, transform=ax.transAxes, ha='left', va='top')
        else :
            ax.legend(title="Superpop", loc=legend_loc, labels=[pop])
        ax.set_title(None)
        ax.set_xlabel(None)
        ax.set_xticklabels([])  # Remove x-tick labels
        ax.set_xlim(x_min, x_max)

    # Add summary histogram with all superpopulations
    ax1 = fig.add_subplot(gs[-1])
    sns.histplot(plot_df.loc[plot_df["Super Population"]!="REF"], 
                    x=vep_col, 
                    # bins=bins,
                    binwidth=binwidth*4,
                    hue=hue_bottom,
                    palette=cmap_bottom,
                    legend=False,
                    multiple="fill",
                    ax=ax1)
    # ax1.text(0.98, 0.95, "All Populations", transform=ax1.transAxes, ha='right', va='top')
    ax1.set_xlabel(f"Variant Effect Prediction ({vep_col})")
    if hue_bottom==variant_col:
        ax1.set_ylabel("Proportion\nby Variant")
    elif hue_bottom==clinsig_col:
        ax1.set_ylabel("Proportion\nby ClinSig")
    elif hue_bottom=="Super Population":
        ax1.set_ylabel("Proportion\nby Superpop")
    else:
        ax1.set_ylabel(f"Proportion\nby {hue_bottom}")
    ax1.set_xlim(x_min, x_max) 

    if i is not None:
        plt.suptitle(f"Distribution of VEP scores by Super Population\
                    \n• Haplotypes: {plot_df['haplotype'].nunique()}\
                    \n• Variant (Protein): {plot_df['CLNHGVS'].iloc[0]} ({plot_df[gene_col].iloc[0]})\
                    \n• Disease: {row_selected['MONDO_label'].replace('_',' ')}\
                    \n• Review Status: {plot_df['CLNREVSTAT'].iloc[0].replace('_',' ')}", 
                    y=1.03, x=0.125, ha='left')
    else:
        plt.suptitle(f"Distribution of VEP scores by Super Population\
                     \n• Haplotypes: {plot_df['haplotype'].nunique()}\
                     \n• Variants: {plot_df.groupby(clinsig_col)[variant_col].nunique().to_dict()}\
                     \n• Genes: {plot_df[gene_col].iloc[0].split(":")[0] if plot_df[gene_col].nunique() == 1 else plot_df[gene_col].nunique()}\
                     \n• Diseases: {plot_df['CLNDN'].iloc[0] if plot_df['CLNDN'].nunique() == 1 else plot_df['CLNDN'].nunique()}\
                     ", 
                     y=1.03, x=0.125, ha='left')
    plt.tight_layout() 

    return plot_df

def extract_id_cols(df,
                    input_col="CLNDISDB",
                    search_terms=["MONDO","OMIM","Orphanet","MedGen","MeSH"],
                     add_counts=True,
                     sep="|",
                     verbose=True):
    """
    Extract ID columns from the input column using regex.
    Args:
        df: DataFrame
        input_col: Column to extract IDs from
        search_terms: List of search terms to extract
        add_counts: Whether to add count columns
        sep: Separator for the input column
        verbose: Whether to print verbose output
    """
    search_terms = utils.as_list(search_terms)
    
    if verbose:
        print(f"Extracting {len(search_terms)} ID column(s).")
    
    # Create regex pattern once
    pattern = '|'.join(f'({term}:[^,{sep}]+)' for term in search_terms)
    
    # Get unique values and their indices
    unique_values = df[input_col].unique()
    unique_indices = {val: idx for idx, val in enumerate(unique_values)}
    
    # Extract IDs only from unique values
    extracted = pd.Series(unique_values).str.extractall(pattern)
    
    # Process results for each term
    for i, term in enumerate(search_terms):
        if verbose:
            print(f"Extracting {term} IDs.")
        # Get results for unique values
        unique_results = extracted[i].groupby(level=0).agg(list)
        # Map back to original dataframe
        df.loc[:,term] = df[input_col].map(lambda x: unique_results.get(unique_indices[x], []))
        if verbose:
            print(f"Adding {term} count column.")
        if add_counts:
            df.loc[:,f'{term}_n'] = df[term].str.len()
    
    return df

def filter_top_mutants(vep_df, 
                       within_site_var, 
                       search_terms, 
                       search_col=["CLNDN","MONDO_label"],
                       top_n=5,
                       mutants_per_group=None):
    if search_terms is None:
        sub_df = vep_df.copy()
        within_site_var_sub = within_site_var.copy()
    else:
        sub_df = vep_df.loc[vep_df[search_col[0]].str.lower().str.contains("|".join(search_terms))]
        within_site_var_sub = within_site_var.loc[within_site_var[search_col[1]].str.lower().str.contains("|".join(search_terms))]

    if mutants_per_group is not None:
        top_mutants = within_site_var_sub.groupby(["clinsig"]).apply(lambda x: x.head(mutants_per_group))["mutant"]
    else:
        top_mutants = within_site_var_sub["mutant"].unique()[:top_n]
    print(top_mutants.shape[0],"top mutants selected")
    sub_df=  sub_df.loc[sub_df["mutant"].isin(top_mutants)]
    return sub_df

def plot_top_mondo(within_site_var_mean,
                   vep_col="flashzoi",
                   label_col="MONDO_label",
                   id_col="MONDO_split",
                   hue="site",
                   palette="plasma",
                   max_label_length=70,
                   top_n=20,
                   figsize=(7, 8)):
    import seaborn as sns
    import matplotlib.pyplot as plt
    import textwrap

    plot_dat = within_site_var_mean.drop_duplicates(subset=[vep_col]).head(top_n)
    
    plot_dat['disease_label'] = plot_dat.apply(lambda row: textwrap.shorten(row[label_col], width=max_label_length, placeholder="...") + f" ({row[id_col]})", axis=1)
    plt.figure(figsize=figsize) 
    ax = sns.barplot(data=plot_dat,
                y="disease_label", 
                x=vep_col,
                palette=palette,
                hue=hue)
    plt.legend(title="Unique variants", loc="lower right")
    plt.xlabel("Within-variant variance (N unique variants)")
    plt.ylabel("Disease (MONDO ID)")
    
    # Add site count annotations
    for i, row in enumerate(plot_dat.itertuples()):
        ax.text(getattr(row, vep_col) + plot_dat[vep_col].max()*0.01, i, f"({row.site})", 
                va='center', ha='left')
        

# Exponential decay
def exp_func(x, a, b):
    return a * np.exp(b * x)

 # Exponential decay: y = a * exp(-b * x)
def exp_decay_func(x, a, b):
    return a * np.exp(-b * x)

# Logistic function
def logistic_func(x, L, k, x0):
    return L / (1 + np.exp(-k * (x - x0)))

# Quadratic polynomial
def poly2_func(x, a, b, c):
    return a * x**2 + b * x + c

def linear_func(x, a, b):
    return a * x + b


def fit_curve_and_plot(x, y, ax, fit_model="logistic", alpha=0.6):
    """Helper function to fit curve and plot it with confidence intervals"""
    from scipy import stats
    from scipy.optimize import curve_fit
    
    # Fit the selected model
    if fit_model == "logistic":
        popt, pcov = curve_fit(logistic_func, x, y, maxfev=10000)
        y_pred = logistic_func(x, *popt)
    elif fit_model == "exp":
        popt, pcov = curve_fit(exp_func, x, y, maxfev=10000)
        y_pred = exp_func(x, *popt)
    elif fit_model == "exp_decay": 
        popt, pcov = curve_fit(exp_decay_func, x, y, maxfev=10000)
        y_pred = exp_decay_func(x, *popt)
    elif fit_model == "poly2":
        popt, pcov = curve_fit(poly2_func, x, y, maxfev=10000)
        y_pred = poly2_func(x, *popt)
    elif fit_model == "linear":
        popt, pcov = curve_fit(linear_func, x, y, maxfev=10000)
        y_pred = linear_func(x, *popt)
    else:
        raise ValueError(f"Invalid fit model: {fit_model}")

    # Calculate R²
    residuals = y - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - (ss_res / ss_tot)
    
    # Calculate p-value using F-test
    n = len(x)
    p = len(popt)
    f_stat = (ss_tot - ss_res) / p / (ss_res / (n - p))
    p_value = 1 - stats.f.cdf(f_stat, p, n - p)
    
    # Plot curve and confidence intervals
    x_line = np.linspace(x.min(), x.max(), 100)
    if fit_model == "logistic":
        y_line = logistic_func(x_line, *popt)
        perr = np.sqrt(np.diag(pcov))
        y_upper = logistic_func(x_line, *(popt + 1.96 * perr))
        y_lower = logistic_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "exp":
        y_line = exp_func(x_line, *popt)
        perr = np.sqrt(np.diag(pcov))
        y_upper = exp_func(x_line, *(popt + 1.96 * perr))
        y_lower = exp_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "exp_decay":
        y_line = exp_decay_func(x_line, *popt)
        perr = np.sqrt(np.diag(pcov))
        y_upper = exp_decay_func(x_line, *(popt + 1.96 * perr))
        y_lower = exp_decay_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "poly2":
        y_line = poly2_func(x_line, *popt)
        perr = np.sqrt(np.diag(pcov))
        y_upper = poly2_func(x_line, *(popt + 1.96 * perr))
        y_lower = poly2_func(x_line, *(popt - 1.96 * perr))
    elif fit_model == "linear":
        y_line = linear_func(x_line, *popt)
        perr = np.sqrt(np.diag(pcov))
        y_upper = linear_func(x_line, *(popt + 1.96 * perr))
        y_lower = linear_func(x_line, *(popt - 1.96 * perr))
    ax.fill_between(x_line, y_lower, y_upper, color='black', alpha=0.1)
    ax.plot(x_line, y_line, color='black', alpha=alpha)
    
    # Add R² and p-value as text
    ax.text(0.95, 0.95, f'R² = {r2:.3f}\np = {p_value:.2e}', 
           transform=ax.transAxes, 
           verticalalignment='top',
           horizontalalignment='right',
           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    return r2, p_value

def plot_vep_vs_af(df,
                   vep_col="VEP",
                   af_col="AF",
                   col=None,
                   hue="CLNSIG_simple",
                   logy=False,
                   logx=False,
                   sharex=True, 
                   sharey=True,
                   palette="Set2",
                   height=4, 
                   fit_line=True,
                   alpha=0.6,
                   aspect=1,
                   xlim=None,
                   ylim=(0, 1),
                   fit_model="logistic",
                   ax=None,
                   col_wrap=3
                   ): 

    # Create a figure with subplots for each CLNSIG_simple category
    epsilon = 1e-6

    df = df.copy()

    df.dropna(subset=[vep_col, af_col], inplace=True)
    df.loc[:, "VEP_log"] = np.log10(df[vep_col] + epsilon)
    df.loc[:, "AF_log"] = np.log10(df[af_col] + epsilon)

    if col is not None and col == "Super Population":
        col_order = ["REF"]+[x for x in df["Super Population"].unique().tolist() if x!="REF"]
        df.loc[:, "Super Population"] = pd.Categorical(df["Super Population"], categories=col_order, ordered=True)
        df = df.sort_values(by="Super Population")
    else:
        col_order = None
    if hue is not None and hue == "CLNSIG_simple":
        hue_order = list(utils.get_clinsig_palette().keys())
        hue_order.reverse()
        df.loc[:, "CLNSIG_simple"] = pd.Categorical(df["CLNSIG_simple"], categories=hue_order, ordered=True)
        df = df.sort_values(by="CLNSIG_simple")
    else:
        hue_order = None

    if logy:
        y_var = "AF_log"
    else:
        y_var = af_col
    if logx:
        x_var = "VEP_log"
    else:
        x_var = vep_col

    # Get appropriate color palette
    if hue=="Super Population":
        cmap = utils.get_superpop_palette()
    elif hue=="CLNSIG_simple":
        cmap = utils.get_clinsig_palette()
    else:
        cmap = utils.make_palette(df[hue].unique(), palette=palette)

    # Create figure based on whether faceting is requested
    if col is None:
        plt.figure(figsize=(height*aspect, height))
        if ax is None:
            ax = plt.gca()
        
        # Add scatter plot
        sns.scatterplot(data=df,
                       x=x_var,
                       y=y_var, 
                       hue=hue,
                       palette=cmap,
                       alpha=alpha,
                       ax=ax)

        # Add regression line and calculate statistics
        if fit_line:
            fit_curve_and_plot(df[x_var], df[y_var], ax, fit_model, alpha)
            
        if xlim is not None:
            ax.set_xlim(xlim)
        if ylim is not None:
            # Set y-ticks to show only the range we want to display
            ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))
            # Don't set ylim to allow points to be visible outside the tick range
        plt.title(f"Variants: {df['site'].nunique()}")
        
    else:
        g = sns.FacetGrid(df, 
                        col=col,
                        sharex=sharex, 
                        sharey=sharey,
                        col_wrap=col_wrap, 
                        height=height,
                        aspect=aspect,
                        col_order=col_order,
                        margin_titles=True)

        # Add scatter plot
        g.map_dataframe(sns.scatterplot, 
                        x=x_var,
                        y=y_var, 
                        hue=hue,
                        palette=cmap,
                        alpha=alpha)

        # Add curve fit to each facet
        if fit_line:
            def fit_and_plot_facet(data, **kwargs):
                ax = plt.gca()
                fit_curve_and_plot(data[x_var], data[y_var], ax, fit_model, alpha)
                if ylim is not None:
                    # Set y-ticks to show only the range we want to display
                    ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))
            
            g.map_dataframe(fit_and_plot_facet)

        g.add_legend()
        g.fig.suptitle(f"Variants: {df['site'].nunique()}", y=1.02)
        if xlim is not None:
            g.set(xlim=xlim)
        if ylim is not None:
            # Set y-ticks for all facets
            for ax in g.axes.flat:
                ax.set_yticks(np.linspace(ylim[0], ylim[1], 5))
    
    plt.show()


def plot_dr_with_kde_topo(
        dr_df, 
        x_col="dim1",
        y_col="dim2",
        hue_col="superpopulation",
        symbol_col=None, #"Sex",
        point_opacity=0.95,
        point_size=3,

        # KDE background
        add_kde=False, 
        kde_bw_method='scott', 
        kde_n=20, 
        kde_levels=50,  # More levels for a topographic effect
        border_pad_frac=0.15,  # Increase border padding to show full islands

        # New param: how many dotted lines to draw (1=every, 2=every other, etc)
        contour_line_step=4,

        # Shadow effect
        add_shadow=True,
        shadow_offset = 0.0, # adjust for best effect
        shadow_opacity = 0.3,
        shadow_size_increase = 3,  # how much larger than the main marker

        # Plot params
        plot_bgcolor="white",   
        paper_bgcolor="white",  
        height=600,
        width=800,

        # New argument to control gridlines
        show_grid=True,
        hover_data=None,

        # New argument for cluster labeling
        cluster_col=None,
        scatter_kwargs={},
    ): 
    """
    Plots a dimensionality reduction (DR) scatter plot with optional KDE topographic background.

    Parameters
    ----------
    dr_df : pd.DataFrame
        DataFrame containing the DR coordinates and metadata.
    x_col : str, default="dim1"
        Column name for the x-axis (first DR dimension).
    y_col : str, default="dim2"
        Column name for the y-axis (second DR dimension).
    hue_col : str, default="Super Population"
        Column name for coloring points by group.
    symbol_col : str or None, default=None
        Column name for symbolizing points by group.
    add_kde : bool, default=False
        Whether to add a KDE-based topographic background.
    kde_bw_method : str or float, default='scott'
        Bandwidth method for KDE ('scott', 'silverman', or float).
    kde_n : int, default=20
        Number of grid points per axis for KDE evaluation.
    kde_levels : int, default=30
        Number of contour/heatmap levels for the KDE background.
    border_pad_frac : float, default=0.15
        Fractional padding to add to plot borders for KDE background.
    contour_line_step : int, default=1
        Draw every Nth contour line for the topographic effect.
    add_shadow : bool, default=True
        Whether to add a shadow effect behind points.
    shadow_offset : float, default=0.0
        Offset for the shadow effect.
    shadow_opacity : float, default=0.3
        Opacity of the shadow markers.
    shadow_size_increase : float, default=3
        Size increase for shadow markers relative to main markers.
    plot_bgcolor : str, default="white"
        Background color of the plot area.
    paper_bgcolor : str, default="white"
        Background color of the entire figure.
    height : int, default=600
        Height of the figure in pixels.
    width : int, default=800
        Width of the figure in pixels.
    show_grid : bool, default=True
        Whether to show gridlines on the plot.
    hover_data : list of str, default=None
        Columns to include in the hover data.
    cluster_col : str or None, default=None
        Column name for cluster labels to annotate each cluster (one label per cluster).
    scatter_kwargs : dict
        Additional keyword arguments passed to px.scatter.

    Returns
    -------
    fig : plotly.graph_objs.Figure
        The generated Plotly figure.
    """
 
    # Import libraries
    import plotly.express as px
    import plotly.graph_objects as go
    import numpy as np
    from scipy.stats import gaussian_kde
    

    palette = utils.get_superpop_palette()
    fig = go.Figure()

    # Optionally add KDE background
    if add_kde: 
        x = dr_df[x_col].values
        y = dr_df[y_col].values
        if len(x) > 1:
            # Compute KDE
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy, bw_method=kde_bw_method)
            # Compute the full plot range for background fill
            all_x = dr_df[x_col].values
            all_y = dr_df[y_col].values
            # Increase padding to extend the borders and show full islands
            xpad = (all_x.max() - all_x.min()) * border_pad_frac
            ypad = (all_y.max() - all_y.min()) * border_pad_frac
            xgrid = np.linspace(all_x.min() - xpad, all_x.max() + xpad, kde_n)
            ygrid = np.linspace(all_y.min() - ypad, all_y.max() + ypad, kde_n)
            xx, yy = np.meshgrid(xgrid, ygrid)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            # Discretize the KDE into "levels" for a topographic effect
            if kde_levels is not None and kde_levels > 1:
                zmin, zmax = zz.min(), zz.max()
                levels = np.linspace(zmin, zmax, kde_levels + 1)
                zz_digitized = np.digitize(zz, levels, right=True)
                # For contour lines, we want the actual level values
                zz_levels = levels[zz_digitized]
            else:
                zz_levels = zz

            # Add as heatmap (background "elevation")
            fig.add_trace(go.Heatmap(
                x=xgrid,
                y=ygrid,
                z=zz_levels,
                colorscale=utils.topo_colorscale,
                opacity=0.9,
                showscale=False,
                hoverinfo='skip',
                zsmooth='best'
            ))

            # Add contour lines for topographic effect, with control over how many lines to draw
            if kde_levels is not None and kde_levels > 1:
                zmin, zmax = zz.min(), zz.max()
                all_levels = np.linspace(zmin, zmax, kde_levels + 1)
                # Only draw every Nth contour line
                contour_levels = all_levels[::contour_line_step]
                # If the last level is not included, add it to ensure the outermost contour is drawn
                if contour_levels[-1] != all_levels[-1]:
                    contour_levels = np.append(contour_levels, all_levels[-1])
                # Plot each contour line individually for full control
                for i, level in enumerate(contour_levels):
                    # Skip the first level (lowest) if you don't want a line at the very bottom
                    if i == 0:
                        continue
                    fig.add_trace(go.Contour(
                        x=xgrid,
                        y=ygrid,
                        z=zz,
                        contours=dict(
                            start=level,
                            end=level,
                            size=0,
                            coloring='none',
                            showlines=True
                        ),
                        line=dict(
                            color='black',
                            dash='dot',
                            width=1
                        ),
                        showscale=False,
                        hoverinfo='skip',
                        opacity=0.5,
                        showlegend=False
                    ))
            else:
                # Fallback: draw all contours as before
                fig.add_trace(go.Contour(
                    x=xgrid,
                    y=ygrid,
                    z=zz,
                    contours=dict(
                        start=zmin,
                        end=zmax,
                        size=(zmax-zmin)/kde_levels if kde_levels else 1,
                        coloring='none',
                        showlines=True
                    ),
                    line=dict(
                        color='black',
                        dash='dot',
                        width=1
                    ),
                    showscale=False,
                    hoverinfo='skip',
                    opacity=0.5,
                    showlegend=False  # Remove contour (dotted lines) from legend
                ))

    # Add scatter points with shadow effect
    # We'll add a "shadow" marker for each point, slightly offset and with a blurred, semi-transparent black color.
    # Then add the main points on top.

    # Create scatter plot with Plotly Express (for color mapping and legend)
    scatter = px.scatter(
        dr_df,
        x=x_col,
        y=y_col,
        color=hue_col,
        symbol=symbol_col,
        color_discrete_map=palette,
        opacity=point_opacity, 
        hover_data=hover_data,
        width=width,
        height=height,
        **scatter_kwargs
    )
    # Decrease point size by setting marker size in the scatter plot
    scatter.update_traces(marker=dict(size=point_size))

    # Add shadow traces first (one per color group)
    if add_shadow:
        for trace in scatter.data:
            # Get the points for this trace
            x_shadow = [v + shadow_offset for v in trace.x]
            y_shadow = [v - shadow_offset for v in trace.y]
            # Use the same marker size, but a bit larger for the shadow
            marker_size = trace.marker.size if trace.marker.size is not None else 12
            shadow_marker_size = marker_size + shadow_size_increase

            # Add shadow trace (underneath)
            fig.add_trace(
                go.Scatter(
                    x=x_shadow,
                    y=y_shadow,
                    mode="markers",
                    marker=dict(
                        size=shadow_marker_size,
                        color="rgba(0,0,0,{})".format(shadow_opacity),
                        line=dict(width=0),
                    ),
                    showlegend=False,
                    hoverinfo="skip",
                )
            )

    # Add main scatter traces, but make points bigger in legend only
    for trace in scatter.data:
        # Add the trace to the figure
        fig.add_trace(trace)
 
    # Add a black diamond outline around the REF point
    ref_points = dr_df[dr_df["sample"] == "REF"]
    if not ref_points.empty:
        fig.add_scatter(
            x=ref_points[x_col],
            y=ref_points[y_col],
            mode="markers",
            marker=dict(
                symbol="diamond",
                size=18,
                color="rgba(0,0,0,0)",  # transparent fill
                line=dict(
                    color="white",
                    width=3
                )
            ),
            showlegend=False,
            hoverinfo="skip"
        )

     # Add cluster labels if requested
    # Add a parameter to control the cluster label offset
    cluster_label_offset = 0.5  # You can move this to the function signature if you want it user-configurable
 
    if cluster_col is not None and cluster_col in dr_df.columns:
        # For each cluster, pick a representative point (e.g., the centroid)
        cluster_groups = dr_df.groupby(cluster_col)
        cluster_label_traces = []
        for cluster_id, group in cluster_groups:
            if cluster_id in ['-1', -1]:
                continue
            # Use the mean as the label position
            x_label = group[x_col].mean()
            y_label = group[y_col].mean()
            # Offset the label by a fixed amount so it's not right on top of the cluster
            x_offset = cluster_label_offset
            y_offset = cluster_label_offset

            # Add a black shadow text (slightly offset)
            cluster_label_traces.append(
                go.Scatter(
                    x=[x_label + x_offset + 0.01],  # offset for shadow
                    y=[y_label + y_offset - 0.01],
                    mode="text",
                    text=[str(cluster_id)],
                    textposition="middle center",
                    textfont=dict(
                        size=18,
                        color="black",
                        family="Roboto Mono, monospace",
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                )
            )
            # Add the main white label on top, also offset
            cluster_label_traces.append(
                go.Scatter(
                    x=[x_label + x_offset],
                    y=[y_label + y_offset],
                    mode="text",
                    text=[str(cluster_id)],
                    textposition="middle center",
                    textfont=dict(
                        size=18,
                        color="white",
                        family="Roboto Mono, monospace",
                    ),
                    showlegend=False,
                    hoverinfo="skip"
                )
            )
        # Add all label traces at the end so they're above all other layers
        for trace in cluster_label_traces:
            fig.add_trace(trace)

    # Set axis ranges to match the KDE background, with extra padding to show full islands
    if add_kde and 'xgrid' in locals() and 'ygrid' in locals() and len(x) > 1:
        fig.update_xaxes(range=[xgrid[0], xgrid[-1]])
        fig.update_yaxes(range=[ygrid[0], ygrid[-1]])
    else:
        # Even if not using KDE, extend the axis limits to show full islands
        all_x = dr_df[x_col].values
        all_y = dr_df[y_col].values
        xpad = (all_x.max() - all_x.min()) * border_pad_frac
        ypad = (all_y.max() - all_y.min()) * border_pad_frac
        fig.update_xaxes(range=[all_x.min() - xpad, all_x.max() + xpad])
        fig.update_yaxes(range=[all_y.min() - ypad, all_y.max() + ypad])

    # Add a subtle background and grid to mimic a map
    fig.update_layout(
        width=width,
        height=height,
        xaxis_title=None,
        yaxis_title=None,
        plot_bgcolor=plot_bgcolor,
        paper_bgcolor=paper_bgcolor,
        
        xaxis=dict(
            showgrid=show_grid,
            gridcolor="#bdbdbd" if show_grid else None,  # medium gray (grid lines)
            zeroline=False,
            showticklabels=False,
            title=None
        ),
        yaxis=dict(
            showgrid=show_grid,
            gridcolor="#bdbdbd" if show_grid else None,  # medium gray (grid lines)
            zeroline=False,
            showticklabels=False,
            title=None
        ),
        font=dict(
            family="Roboto Mono, monospace",
            size=14,
            color="#222"  # very dark gray (almost black, font)
        ),
        margin=dict(l=40, r=40, t=40, b=40)
    )
    
    fig.show()


def plot_dr_with_kde_topo_static(
        dr_df, 
        x_col="dim1",
        y_col="dim2",
        hue_col="Super Population",
        symbol_col=None,  # "Sex"

        # KDE background
        add_kde=False, 
        kde_bw_method='scott', 
        kde_n=20, 
        kde_levels=50,  # More levels for a topographic effect
        border_pad_frac=0.15,  # Increase border padding to show full islands

        # New param: how many dotted lines to draw (1=every, 2=every other, etc)
        contour_line_step=4,

        # Shadow effect
        add_shadow=True,
        shadow_offset=0.0,  # adjust for best effect
        shadow_opacity=0.3,
        shadow_size_increase=3,  # how much larger than the main marker

        # Plot params
        plot_bgcolor="white",   
        paper_bgcolor="white",  
        height=6,
        width=8,

        # New argument to control gridlines
        show_grid=True): 
    """
    Plots a dimensionality reduction (DR) scatter plot with optional KDE topographic background (static matplotlib version).

    Parameters
    ----------
    dr_df : pd.DataFrame
        DataFrame containing the DR coordinates and metadata.
    x_col : str, default="dim1"
        Column name for the x-axis (first DR dimension).
    y_col : str, default="dim2"
        Column name for the y-axis (second DR dimension).
    hue_col : str, default="Super Population"
        Column name for coloring points by group.
    symbol_col : str or None, default=None
        Column name for symbolizing points by group.
    add_kde : bool, default=False
        Whether to add a KDE-based topographic background.
    kde_bw_method : str or float, default='scott'
        Bandwidth method for KDE ('scott', 'silverman', or float).
    kde_n : int, default=20
        Number of grid points per axis for KDE evaluation.
    kde_levels : int, default=30
        Number of contour/heatmap levels for the KDE background.
    border_pad_frac : float, default=0.15
        Fractional padding to add to plot borders for KDE background.
    contour_line_step : int, default=1
        Draw every Nth contour line for the topographic effect.
    add_shadow : bool, default=True
        Whether to add a shadow effect behind points.
    shadow_offset : float, default=0.0
        Offset for the shadow effect.
    shadow_opacity : float, default=0.3
        Opacity of the shadow markers.
    shadow_size_increase : float, default=3
        Size increase for shadow markers relative to main markers.
    plot_bgcolor : str, default="white"
        Background color of the plot area.
    paper_bgcolor : str, default="white"
        Background color of the entire figure.
    height : float, default=6
        Height of the figure in inches.
    width : float, default=8
        Width of the figure in inches.
    show_grid : bool, default=True
        Whether to show gridlines on the plot.

    Returns
    -------
    fig, ax : matplotlib Figure and Axes
        The generated matplotlib figure and axes.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    from scipy.stats import gaussian_kde

    palette = utils.get_superpop_palette()
    # If palette is a dict, ensure it matches the unique values in hue_col
    unique_hues = dr_df[hue_col].unique()
    color_map = palette if isinstance(palette, dict) else {k: v for k, v in zip(unique_hues, palette)}

    # Set up figure
    fig, ax = plt.subplots(figsize=(width, height))
    fig.patch.set_facecolor(paper_bgcolor)
    ax.set_facecolor(plot_bgcolor)

    # Optionally add KDE background
    if add_kde:
        x = dr_df[x_col].values
        y = dr_df[y_col].values
        if len(x) > 1:
            xy = np.vstack([x, y])
            kde = gaussian_kde(xy, bw_method=kde_bw_method)
            all_x = dr_df[x_col].values
            all_y = dr_df[y_col].values
            xpad = (all_x.max() - all_x.min()) * border_pad_frac
            ypad = (all_y.max() - all_y.min()) * border_pad_frac
            xgrid = np.linspace(all_x.min() - xpad, all_x.max() + xpad, kde_n)
            ygrid = np.linspace(all_y.min() - ypad, all_y.max() + ypad, kde_n)
            xx, yy = np.meshgrid(xgrid, ygrid)
            zz = kde(np.vstack([xx.ravel(), yy.ravel()])).reshape(xx.shape)
            # Discretize the KDE into "levels" for a topographic effect
            if kde_levels is not None and kde_levels > 1:
                zmin, zmax = zz.min(), zz.max()
                levels = np.linspace(zmin, zmax, kde_levels + 1)
                zz_digitized = np.digitize(zz, levels, right=True)
                # The following line is the source of the error if levels is a list of RGB tuples:
                # zz_levels = levels[zz_digitized]
                # Instead, always use a numeric array for imshow, and let cmap handle the color mapping.
                zz_levels = zz
            else:
                zz_levels = zz

            # --- Fix: Ensure cmap is a valid matplotlib colormap, not a list of RGB tuples ---
            cmap = "terrain"
            if hasattr(utils, "topo_colorscale"):
                topo_cmap = getattr(utils, "topo_colorscale")
                # If it's a string, use as colormap name
                if isinstance(topo_cmap, str):
                    cmap = topo_cmap
                # If it's a matplotlib colormap object, use it directly
                elif hasattr(topo_cmap, "name") or callable(getattr(topo_cmap, "__call__", None)):
                    cmap = topo_cmap
                # If it's a list, try to create a ListedColormap, but only if the list is valid
                elif isinstance(topo_cmap, list):
                    try:
                        from matplotlib.colors import ListedColormap, to_rgba
                        # Validate: must be a list of valid color specs (strings or RGB/RGBA tuples of length 3 or 4)
                        valid_colors = []
                        for c in topo_cmap:
                            try:
                                rgba = to_rgba(c)
                                valid_colors.append(c)
                            except Exception:
                                pass
                        if len(valid_colors) == len(topo_cmap) and len(valid_colors) > 0:
                            cmap = ListedColormap(valid_colors)
                        else:
                            cmap = "terrain"
                    except Exception:
                        cmap = "terrain"
            # Plot heatmap background
            ax.imshow(
                np.flipud(zz_levels),
                extent=[xgrid[0], xgrid[-1], ygrid[0], ygrid[-1]],
                aspect='auto',
                cmap=cmap,
                alpha=0.9,
                zorder=0
            )

            # Add contour lines for topographic effect
            if kde_levels is not None and kde_levels > 1:
                zmin, zmax = zz.min(), zz.max()
                all_levels = np.linspace(zmin, zmax, kde_levels + 1)
                contour_levels = all_levels[::contour_line_step]
                ax.contour(
                    xx, yy, zz,
                    levels=contour_levels,
                    colors='k',
                    linewidths=0.5,
                    linestyles='dotted',
                    alpha=0.5,
                    zorder=1
                )

    # Plot points (with optional shadow)
    marker_kwargs = dict(
        s=40,
        linewidth=0.5,
        edgecolor='k',
        alpha=1.0,
        zorder=3
    )
    if symbol_col is not None:
        symbols = dr_df[symbol_col].unique()
        from itertools import cycle
        marker_styles = ['o', 's', '^', 'D', 'P', 'X', '*', 'v', '<', '>']
        symbol_map = {sym: m for sym, m in zip(symbols, cycle(marker_styles))}
    else:
        symbol_map = {}

    for group, group_df in dr_df.groupby(hue_col):
        color = color_map.get(group, "#333")
        if symbol_col is not None:
            for sym, sym_df in group_df.groupby(symbol_col):
                marker = symbol_map.get(sym, 'o')
                # Shadow
                if add_shadow:
                    ax.scatter(
                        sym_df[x_col] + shadow_offset,
                        sym_df[y_col] + shadow_offset,
                        s=marker_kwargs['s'] * shadow_size_increase,
                        color='k',
                        alpha=shadow_opacity,
                        marker=marker,
                        zorder=2,
                        linewidth=0
                    )
                ax.scatter(
                    sym_df[x_col],
                    sym_df[y_col],
                    c=[color],
                    marker=marker,
                    label=f"{group} - {sym}",
                    **marker_kwargs
                )
        else:
            # Shadow
            if add_shadow:
                ax.scatter(
                    group_df[x_col] + shadow_offset,
                    group_df[y_col] + shadow_offset,
                    s=marker_kwargs['s'] * shadow_size_increase,
                    color='k',
                    alpha=shadow_opacity,
                    marker='o',
                    zorder=2,
                    linewidth=0
                )
            ax.scatter(
                group_df[x_col],
                group_df[y_col],
                c=[color],
                marker='o',
                label=group,
                **marker_kwargs
            )

    # Set axis limits to match KDE background, with extra padding
    all_x = dr_df[x_col].values
    all_y = dr_df[y_col].values
    xpad = (all_x.max() - all_x.min()) * border_pad_frac
    ypad = (all_y.max() - all_y.min()) * border_pad_frac
    ax.set_xlim(all_x.min() - xpad, all_x.max() + xpad)
    ax.set_ylim(all_y.min() - ypad, all_y.max() + ypad)

    # Remove axis ticks and labels for a "map" look
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("")
    ax.set_ylabel("")

    # Set grid and background
    ax.grid(show_grid, color="#bdbdbd", linewidth=0.7, linestyle='--', alpha=0.5, zorder=0)
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Set font
    ax.tick_params(labelsize=12)
    if dr_df[hue_col].nunique() < 20:
        ax.legend(loc='best', fontsize=10, frameon=True, framealpha=0.8)
    else:
        ax.legend([], [], frameon=False)

    plt.tight_layout()
    plt.show()
    return fig, ax


def plot_subpopulation_proportions_per_cluster(
        dr_df,
        cluster_col="cluster",
        group_col="Super Population",
        palette=utils.get_superpop_palette(),
        top_height_frac = 0.20,   # <-- adjust this value as desired (e.g., 0.1, 0.2, etc.)
        figsize=(10, 8),
        ):
    """
    Plots the proportions of subpopulations within each cluster.
    """

    # Include all clusters except missing/noise (-1), but keep REF
    plot_df = dr_df[~dr_df[cluster_col].isin(['-1', -1])].copy()
    if plot_df.empty:
        raise ValueError("No clusters to plot")

    # Count individuals per (cluster, superpop)
    count_df = (
        plot_df.groupby([cluster_col, group_col])
        .size()
        .reset_index(name='count')
    )

    # Get total individuals per cluster for sorting
    cluster_totals = count_df.groupby(cluster_col)['count'].sum().sort_values(ascending=False)
    ordered_clusters = cluster_totals.index.tolist()

    # Set cluster as categorical for ordering
    count_df[cluster_col] = pd.Categorical(count_df[cluster_col], 
                                           categories=ordered_clusters, 
                                           ordered=True)
    # Pivot to wide format
    pivot_df = count_df.pivot(index=cluster_col, 
                              columns=group_col, 
                              values='count').fillna(0)
    pivot_df = pivot_df.loc[ordered_clusters]  # ensure cluster order

    # Proportions for main plot
    prop_df = pivot_df.div(pivot_df.sum(axis=1), axis=0)

    # Colors
     
    if isinstance(palette, dict):
        color_list = [palette[sp] for sp in prop_df.columns]
    else:
        color_list = palette

    # Set up figure with two subplots: small one above for absolute numbers, main for proportions
    # Use gridspec with adjustable height ratios
    fig = plt.figure(figsize=figsize)        
    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        2, 1, height_ratios=[top_height_frac, 1 - top_height_frac], hspace=0.05
    )
    ax_top = fig.add_subplot(gs[0, 0])
    ax_main = fig.add_subplot(gs[1, 0], sharex=ax_top)

    # --- Top subplot: absolute numbers ---
    bars = ax_top.bar(
        range(len(ordered_clusters)),
        cluster_totals.values,
        color='gray',
        edgecolor='black',
        width=0.95
    )
    # Annotate the number of individuals above each bar
    for i, (bar, count) in enumerate(zip(bars, cluster_totals.values)):
        height = bar.get_height()
        ax_top.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(cluster_totals) * 0.02,  # small offset above bar
            str(int(count)),
            ha='center',
            va='bottom',
            fontsize=8
        )

    ax_top.set_ylabel('N', fontsize=9)
    ax_top.set_xticks([])  # No x-ticks on top
    ax_top.set_yticks([int(max(cluster_totals))])  # Only show max for compactness
    ax_top.tick_params(axis='y', labelsize=8)
    ax_top.set_title('Individuals per Cluster', fontsize=10, pad=2)
    ax_top.spines['bottom'].set_visible(False)
    ax_top.spines['right'].set_visible(False)
    ax_top.spines['top'].set_visible(False)
    ax_top.spines['left'].set_linewidth(0.5)

    # --- Main subplot: proportions ---
    prop_df.plot(
        kind='bar',
        stacked=True,
        ax=ax_main,
        color=color_list,
        edgecolor='black',
        width=0.95,
        legend=False  # We'll add legend manually
    )

    # Add white vertical lines to separate barplot groups
    for i in range(1, len(ordered_clusters)):
        ax_main.axvline(i - 0.5, color='white', linewidth=2, zorder=20)
        ax_top.axvline(i - 0.5, color='white', linewidth=2, zorder=20)

    # Find which cluster contains REF
    ref_row = dr_df[dr_df['sample'] == 'REF']
    if not ref_row.empty:
        ref_cluster = ref_row['cluster'].values[0]
        if ref_cluster in ordered_clusters:
            ref_x = ordered_clusters.index(ref_cluster)
            # For main plot (proportions)
            bar_height = 1.0
            y_offset = 0.02
            y_annot = 0.04
            ax_main.plot(
                ref_x,
                bar_height + y_offset,
                marker='D',
                markerfacecolor='none',
                markeredgecolor='black',
                markersize=10,
                zorder=30
            )
            ax_main.annotate('REF', (ref_x, bar_height + y_annot), color='black', ha='center', va='bottom', fontsize=10)
            # For top plot (absolute numbers)
            abs_height = cluster_totals.loc[ref_cluster]
            ax_top.plot(
                ref_x,
                abs_height + max(cluster_totals) * 0.05,
                marker='D',
                markerfacecolor='none',
                markeredgecolor='black',
                markersize=8,
                zorder=30
            )
            ax_top.annotate('REF', (ref_x, abs_height + max(cluster_totals) * 0.08), color='black', ha='center', va='bottom', fontsize=8)

    ax_main.set_xlabel('Cluster')
    ax_main.set_ylabel('Proportion of Individuals') 
    ax_main.set_ylim(0, 1.05)
    ax_main.legend(
        title='Super Population',
        bbox_to_anchor=(1, 1),
        loc='upper left'
    )
    ax_main.set_xticks(range(len(ordered_clusters)))
    ax_main.set_xticklabels(ordered_clusters, rotation=0)
    ax_main.tick_params(axis='x', labelsize=10)
    ax_main.tick_params(axis='y', labelsize=10)
    plt.tight_layout()
    plt.show()

def random_forest_feature_importance(
    dr_df,
    X,
    cluster_col="cluster",
    sample_col="sample",
    site_col="site_hap_id",
    top_n=10,
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    one_vs_rest=False,
    as_df=True
):
    """
    Compute feature importances for VEP sites using a Random Forest classifier.

    This function identifies the most important VEP sites (features) for distinguishing between clusters
    in the provided data. It can return either the top features overall or, optionally, the top features
    for each cluster using a one-vs-rest approach.

    Parameters
    ----------
    dr_df : pd.DataFrame
        DataFrame containing sample metadata, including cluster assignments.
    X : pd.DataFrame
        VEP matrix with samples as rows and site_hap_id as columns.
    cluster_col : str, default="cluster"
        Column name in dr_df indicating cluster assignments.
    sample_col : str, default="sample"
        Column name in dr_df indicating sample IDs.
    site_col : str, default="site_hap_id"
        Column name for site/haplotype IDs in X.
    top_n : int, default=10
        Number of top features (sites) to return.
    n_estimators : int, default=100
        Number of trees in the Random Forest.
    random_state : int, default=42
        Random seed for reproducibility.
    n_jobs : int, default=-1
        Number of parallel jobs to run for Random Forest.
    one_vs_rest : bool, default=False
        If True, compute top features for each cluster using a one-vs-rest approach.
        If False, compute top features overall for multiclass classification.
    as_df : bool, default=True
        If True and one_vs_rest is True, return a DataFrame of top features per cluster.
        Otherwise, return a dictionary.

    Returns
    -------
    pd.DataFrame or dict
        If one_vs_rest is False: DataFrame with top_n features and their importances.
        If one_vs_rest is True and as_df is True: DataFrame with top_n features per cluster.
        If one_vs_rest is True and as_df is False: dict mapping cluster labels to Series of top features.

    Notes
    -----
    - The function collapses site_hap_id columns to main site names by removing the last "_#" suffix.
    - Samples with cluster assignment "REF" or cluster < 0 are excluded.
    - Uses scikit-learn's RandomForestClassifier for feature importance.
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.preprocessing import LabelEncoder
    import re

    def _to_df(cluster_top_sites, site_col="site_hap_id"):
        """
        Convert a dictionary of Series (cluster_top_sites) into a stacked DataFrame
        with columns: cluster, site, importance.
        """
        cluster_sites_df = (
            pd.DataFrame(cluster_top_sites)
            .stack()
            .reset_index()
            .rename(columns={site_col: 'site',
                            'site_ploid': 'site',
                             'level_1': "cluster",
                             0: 'importance'})
        ).sort_values(by='importance', ascending=False)
        print(cluster_sites_df.shape)
        return cluster_sites_df

    # Prepare data: X = VEP matrix, y = cluster labels (excluding REF and noise)
    # We'll use only samples with valid cluster assignments (cluster >= 0)
    valid_mask = (dr_df[cluster_col] >= '0') & (dr_df[sample_col] != "REF")
    valid_samples = dr_df.loc[valid_mask, sample_col]

    # Subset VEP matrix and cluster labels
    X_vep = X.loc[valid_samples]
    y = dr_df.set_index(sample_col).loc[valid_samples, cluster_col]

    # Collapse columns to main site name (remove hap_id suffix: last "_#" in col name)
    def strip_hap_id(col):
        """
        Remove the last underscore and number from a column name, e.g. "site_1" -> "site".
        """
        return re.sub(r'_\d+$', '', col)

    main_site_names = X_vep.columns.map(strip_hap_id)
    X_vep_grouped = X_vep.groupby(main_site_names, axis=1).sum()

    if not one_vs_rest:
        # Encode cluster labels if needed
        le = LabelEncoder()
        y_encoded = le.fit_transform(y)

        # Fit a RandomForestClassifier to identify important sites
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs
        )
        rf.fit(X_vep_grouped, y_encoded)

        # Get feature importances
        importances = pd.Series(rf.feature_importances_, index=X_vep_grouped.columns)
        top_sites = importances.sort_values(ascending=False).head(top_n)
        top_sites = top_sites.reset_index().rename(columns={'site_hap_id': 'site', 
                                                            'site_ploid': 'site', 
                                                            0: "importance"})
        return top_sites
    else:
        # Optionally, show per-cluster top sites using one-vs-rest approach
        from tqdm import tqdm
        cluster_top_sites = {}
        for cluster_label in tqdm(np.unique(y), desc="Fitting RandomForestClassifier"):
            # One-vs-rest labels
            y_bin = (y == cluster_label).astype(int)
            rf_bin = RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=random_state,
                n_jobs=n_jobs
            )
            rf_bin.fit(X_vep_grouped, y_bin)
            importances_bin = pd.Series(rf_bin.feature_importances_, index=X_vep_grouped.columns)
            cluster_top_sites[cluster_label] = importances_bin.sort_values(ascending=False).head(top_n)

        if as_df:
            return _to_df(cluster_top_sites, site_col=site_col)
        else:
            return cluster_top_sites

def normalize_vep_df(vep_df,
                     groupby_cols=["GENE","slot"],
                     model_name="flashzoi",
                     method="RobustScaler",
                     suffix="_norm"):
    """
    Normalize the VEP scores for a given model.
    """
    from sklearn.preprocessing import MinMaxScaler, RobustScaler, Normalizer
    if method == "MinMaxScaler":
        scaler = MinMaxScaler()
    elif method == "RobustScaler":
        scaler = RobustScaler()
    elif method == "Normalizer":
        scaler = Normalizer()
    else:
        raise ValueError(f"Invalid method: {method}")

    vep_df[f"{model_name}{suffix}"] = vep_df.groupby(groupby_cols)[model_name].transform(
        lambda x: scaler.fit_transform(x.values.reshape(-1, 1)).ravel() if len(x) > 0 else 0
    )
    return vep_df