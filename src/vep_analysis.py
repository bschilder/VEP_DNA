import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

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
                       metadata_df = None,
                       how = "left"):
    
    if metadata_df is None:
        import src.onekg as og
        metadata_df = og.get_sample_metadata()
    vep_df = vep_df.merge(metadata_df,
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
                height=3,
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
                            vep_col = "flashzoi",
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
                          height_ratios= [1.5]+[1]+([1] * (n_pops + 1)), 
                          hspace=0.1,
                          top=0.9)
 
    # Add histogram with all data in first subplot
    ax0 = fig.add_subplot(gs[0])
    
    # Add REF lines and labels first
    ref_rows = plot_df.loc[plot_df["is_ref"]==True].groupby([metric_col,gene_col, variant_col]).head(1)
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
                    y=1.02, x=0.125, ha='left')
    else:
        plt.suptitle(f"Distribution of VEP scores by Super Population\
                     \n• Haplotypes: {plot_df['haplotype'].nunique()}\
                     \n• Variants: {plot_df.groupby(clinsig_col)[variant_col].nunique().to_dict()}\
                     \n• Genes: {plot_df[gene_col].iloc[0].split(":")[0] if plot_df[gene_col].nunique() == 1 else plot_df[gene_col].nunique()})\
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