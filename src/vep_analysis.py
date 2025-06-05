import matplotlib.pyplot as plt
import seaborn as sns
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