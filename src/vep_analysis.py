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
