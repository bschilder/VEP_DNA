import os
import pooch
import polars as pl


def download_vcf(vcf_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"):
    """
    Download the latest ClinVar VCF file and index file.

    Args:
        vcf_url (str): The URL of the ClinVar VCF file.

    Returns:
        dict: A dictionary containing the path to the VCF file and the path to the index file.
    
    Example:
        download_clinvar()
    """
    vcf_file = pooch.retrieve(vcf_url,
                              fname=os.path.basename(vcf_url),
                              known_hash="19cf6d08cecbd4bae1c09c711a2a31478fc8194a18073c7a86b06583111171b4",
                              progressbar=True)
    idx_file = pooch.retrieve(vcf_url+".tbi",
                              fname=os.path.basename(vcf_url)+".tbi",
                              known_hash="90fd8754c61bc0442c86e295d4d6b7fdbac3ffbb6273d4ead8690e10a2682abf",
                              progressbar=True)
    return {"vcf": vcf_file, "idx": idx_file}


def vcf_to_df(vcf_file=None,
              contig=None):
    from genoray import VCF

    if vcf_file is None:
        vcf_file = download_vcf()["vcf"]

    vcf = VCF(vcf_file, 
            filter=lambda v: "UTR" in v.INFO.get("MC", ""))

    # annotation keys to extract from the INFO field
    info = [
        "AF_ESP", "AF_EXAC", "AF_TGP", "ALLELEID", "CLNDISDB", "CLNDN",
        "CLNHGVS", "CLNREVSTAT", "CLNSIG", "CLNVC", "CLNVCSO", "GENEINFO",
        "MC", "ORIGIN", "RS"
    ]
    vcf_df = vcf.get_record_info(contig=contig,
                                attrs=["CHROM", "POS", "REF", "ALT"], 
                                info=info,
                                progress=True)
    # Get the Molecular Consequence (MC) Sequence Ontology (SO) IDs and term names
    vcf_df = vcf_df.with_columns(
        vcf_df["MC"].str.split("|").list.first().alias("MC_id"),
        vcf_df["MC"].str.split("|").list.last().alias("MC_term")
    )
    # Create a mapping dictionary for review status scores
    review_score_map = {
        "practice_guideline": 4,
        "reviewed_by_expert_panel": 3,
        "criteria_provided,_multiple_submitters,_no_conflicts": 2,
        "criteria_provided,_conflicting_classifications": 1,
        "criteria_provided,_single_submitter": 1,
        "no_assertion_criteria_provided": 0,
        "no_classification_provided": 0,
        "no_classification_for_the_single_variant": 0,
        "no_classifications_from_unflagged_records": 0
    }

    # Create new column using the mapping dictionary
    vcf_df = vcf_df.with_columns(
        pl.col("CLNREVSTAT").replace(review_score_map).alias("CLNREVSTAT_score").cast(pl.Int8)
    )
    
    return vcf_df 

def add_variant_name(df,
                    chrom_col='chrom',
                    start_col='chromStart',
                    end_col='chromEnd',
                    ref_col='REF',
                    alt_col='ALT',
                    alias='name'):
    """Add a variant name column to a DataFrame.
    
    Args:
        df: Polars DataFrame
        chrom_col: Column name for chromosome
        start_col: Column name for start position
        end_col: Column name for end position
        ref_col: Column name for reference allele
        alt_col: Column name for alternate allele
        alias: Name for the output column
        
    Returns:
        DataFrame with added variant name column
    """
    return df.with_columns(pl.concat_str([
        pl.lit('chr'),
        pl.col(chrom_col).str.replace('chr', ''),
        pl.lit(':'),
        pl.col(start_col).cast(pl.Utf8),
        pl.lit('-'),
        pl.col(end_col).cast(pl.Utf8),
        pl.lit('_'),
        pl.col(ref_col),
        pl.col(alt_col)
    ]).alias(alias))

def df_to_bed(vcf_df,  
              save_path=None):
    """Convert VCF DataFrame to BED format.
    
    Args:
        vcf_df: Polars DataFrame in VCF format
        save_path: Optional path to save BED file
        
    Returns:
        DataFrame in BED format
    """
    bed = vcf_df.rename({
        'CHROM': 'chrom',
        'POS': 'chromStart',
        "CLNREVSTAT_score":"score"
    }).with_columns([
        # pl.lit(None).alias('strand'),
        (pl.col('chromStart') + pl.col('REF').str.len_chars()).alias('chromEnd'),
        (pl.col('ALT').list.join(',').alias('ALT')),
    ])
    
    # Add variant name using the function instead of method
    bed = add_variant_name(bed)
    
    bed = bed.select([
        # required columns
        'chrom',
        'chromStart',
        'chromEnd',
        # optional columns
        "name",
        "score",
        # "strand",
        # extra columns
        "REF",
        "ALT",
        'MC_id',
        'MC_term',
        "AF_ESP", "AF_EXAC", "AF_TGP", "ALLELEID", "CLNDISDB", "CLNDN",
        "CLNHGVS", "CLNREVSTAT", "CLNSIG", "CLNVC", "CLNVCSO", "GENEINFO",
        # "MC", "ORIGIN", "RS"
    ]).filter(pl.col('chrom').str.contains('^[0-9]+$|^X$|^Y$'))

    bed = bed.with_columns(pl.col('ALT').fill_null("")).drop_nulls(subset=['ALT'])
    bed = bed.with_columns(pl.col('REF').fill_null("")).drop_nulls(subset=['REF'])

    if save_path:
        bed.to_pandas().to_csv(save_path, sep='\t', index=False)
    return bed


def df_to_sites(vcf_df):

    sites = vcf_df.with_columns([
        # pl.lit(None).alias('strand'),
        (pl.col('POS') + pl.col('REF').str.len_chars()).alias('POS_END'),
        (pl.col('ALT').list.join(',').alias('ALT')),
    ])

    # Add variant name using the function instead of method
    sites = add_variant_name(sites,
                             chrom_col='CHROM',
                             start_col='POS',
                             end_col='POS_END')
    
    sites = sites.select([
        # required columns
        'CHROM',
        'POS',
        'POS_END',
        # optional columns
        "name",
        # extra columns
        "REF",
        "ALT",
        'MC_id',
        'MC_term',
        "AF_ESP", "AF_EXAC", "AF_TGP", "ALLELEID", "CLNDISDB", "CLNDN",
        "CLNHGVS", "CLNREVSTAT", "CLNSIG", "CLNVC", "CLNVCSO", "GENEINFO",
        "CLNREVSTAT_score"
        # "MC", "ORIGIN", "RS"
    ]).filter(pl.col('CHROM').str.contains('^[0-9]+$|^X$|^Y$'))

    sites = sites.with_columns(pl.col('ALT').fill_null("")).drop_nulls(subset=['ALT'])
    sites = sites.with_columns(pl.col('REF').fill_null("")).drop_nulls(subset=['REF'])

    return sites


def bed_to_sites(bed):
    """Convert BED DataFrame to sites format.
    
    Args:
        bed_df: Polars DataFrame in BED format
        save_path: Optional path to save sites file

    Returns:
        DataFrame in sites format
    """
    sites =  bed.rename({
        'chrom': 'CHROM',
        'chromStart': 'POS',
        'chromEnd': 'POS_END',
        "score": "CLNREVSTAT_score"
    })

    sites = sites.select([
        'CHROM',
        'POS',
        'REF',
        'ALT',
        *[col for col in sites.columns if col not in ['CHROM', 'POS', 'REF', 'ALT']]
    ])
    return sites


def read_bed(path, 
             schema_overrides=None,
             separator='\t',
             **kwargs):
    """
    Read a BED file created by the clinvar submodule for usew with GenVarloader.
    """

    if schema_overrides is None:
        schema_overrides = {
            'chrom': pl.Utf8,
            'chromStart': pl.Int64,
            'chromEnd': pl.Int64,
            'score': pl.Float64
        }
    # Import bed file
    bed = pl.read_csv(
        path,
        schema_overrides=schema_overrides,
        separator=separator,
        **kwargs
    ).drop_nulls(subset=['ALT'])
    return bed

def filter_df(vcf_df,
              filters = {},
              verbose=True):
    """
    Filter the VCF DataFrame based on the provided filters.
    
    Args:
        vcf_df (pl.DataFrame): The input VCF DataFrame.
        filters (dict): A dictionary of filters to apply to the DataFrame.
        verbose (bool): Whether to print the number of genes in the filtered DataFrame.
    Returns:
        pl.DataFrame: The filtered VCF DataFrame.

    Example:
        >>> vcf_df = vcf_to_df()
        >>> filters = {
                "MC_term": ["5_prime_UTR_variant","3_prime_UTR_variant"],
                "CLNREVSTAT_score": 2,
                "CLNVC": "single_nucleotide_variant",
                "CLNSIG": ["benign","pathogenic","Benign","Pathogenic"]
            }
        >>> filter_df(vcf_df, filters)
    """

    # Build filter conditions dynamically based on provided filters
    filter_conditions = []
    for key, value in filters.items():
        if key not in vcf_df.columns:
            if verbose:
                print(f"Column {key} not found in DataFrame. Skipping filter.")
            continue
        if isinstance(value, list):
            filter_conditions.append(pl.col(key).str.contains("|".join(value)))
        else:
            filter_conditions.append(pl.col(key) >= value if key == "CLNREVSTAT_score" else pl.col(key) == value)

    # Combine all conditions with &
    combined_condition = filter_conditions[0]
    for condition in filter_conditions[1:]:
        combined_condition = combined_condition & condition

    cv_df = (vcf_df
        .filter(combined_condition)
        # Convert POS column to integer
        .with_columns(pl.col("POS").cast(pl.Int64))
        .drop_nulls(subset=['CLNDN'])
    )

    if verbose:
        print(f"Filtered DataFrame shape: {cv_df.shape}")
        print(f"Variant count: {cv_df.shape[0]}")
        print(f"Gene count: {cv_df['GENEINFO'].unique().len()}")

    return cv_df