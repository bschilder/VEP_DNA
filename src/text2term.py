import text2term  
from text2term import Mapper
import os
import glob

def get_top_ont_mappings_i(
    df, 
    id_map,
    sort_by=["Mapping Score"],
    ascending=[False],
    left_on="name",
    right_on="Source Term",
    how="left",
    prefix="ONT",
    top_n=1,
    filter_obsolete=True,
    within_ont_ids_only=True
):
    """
    Merge the top ontology mappings from `id_map` into the DataFrame `df`.

    Args:
        df (pd.DataFrame): The source DataFrame containing terms to be mapped.
        id_map (pd.DataFrame): DataFrame containing ontology mapping results.
        sort_by (list, optional): Columns to sort by in `id_map` before selecting top mappings. Defaults to ["Mapping Score"].
        ascending (list, optional): Sort order for each column in `sort_by`. Defaults to [False].
        left_on (str, optional): Column in `df` to merge on. Defaults to "name".
        right_on (str, optional): Column in `id_map` to merge on. Defaults to "Source Term".
        how (str, optional): Type of merge to perform. Defaults to "left".
        prefix (str, optional): Prefix for new columns added from `id_map`. Defaults to "ONT".
        top_n (int, optional): Number of top mappings to keep per term. Defaults to 1.
        filter_obsolete (bool, optional): Whether to filter out obsolete terms from `id_map`. Defaults to True.
        within_ont_ids_only (bool, optional): Whether to only keep the top mapping for each term within the same ontology 
        (e.g. IDs in the EFO must start with "EFO:..."). Defaults to True.

    Returns:
        pd.DataFrame: The merged DataFrame with top ontology mappings added.
    """
    id_map = id_map.copy()
    if filter_obsolete:
        id_map = id_map.loc[~id_map["Mapped Term Label"].str.contains("obsolete")]

    col_map = {
        "Mapped Term CURIE": prefix + "_ID",
        "Mapped Term Label": prefix + "_LABEL",
        "Mapping Score": prefix + "_SCORE"
    }
    for col in col_map.values():
        if col in df.columns:
            df = df.drop(columns=[col])

    if right_on in df.columns:
        df = df.drop(columns=[right_on])
    
    if within_ont_ids_only:
        id_map = id_map.loc[id_map["Mapped Term CURIE"].str.startswith(prefix + ":")]

    id_map_121 = id_map.sort_values(by=sort_by, ascending=ascending).groupby([right_on]).head(top_n)
    return df.merge(
        id_map_121[[right_on] + list(col_map.keys())].rename(columns=col_map),
        left_on=left_on,
        right_on=right_on,
        how=how
    )

def get_top_ont_mappings(
    df, 
    term_col="name",
    ont_dict=None,
    within_ont_ids_only=True,
    top_n=1,
    cache_folder=None,
    verbose=True
):
    """
    Map terms in a DataFrame to multiple ontologies and merge the top mappings.

    Args:
        df (pd.DataFrame): The source DataFrame containing terms to be mapped.
        term_col (str, optional): Column in `df` containing the terms to map. Defaults to "name".
        ont_dict (dict, optional): Dictionary mapping ontology acronyms to ontology URLs.
            Defaults to a set of common biomedical ontologies.
        top_n (int, optional): Number of top mappings to keep per term for each ontology. Defaults to 1.
        cache_folder (str, optional): Path to folder for caching ontology data. Defaults to "~/.cache/text2term".
        within_ont_ids_only (bool, optional): Whether to only keep the top mapping for each term within the same ontology 
        (e.g. IDs in the EFO must start with "EFO:..."). Defaults to True.
        verbose (bool, optional): Whether to print progress messages. Defaults to True.

    Returns:
        pd.DataFrame: The DataFrame with top ontology mappings from each ontology merged in.
    """
    if ont_dict is None:
        ont_dict = {
            "EFO": "http://www.ebi.ac.uk/efo/efo.owl",
            "UBERON": "http://purl.obolibrary.org/obo/uberon.owl",
            "CL": "http://purl.obolibrary.org/obo/cl.owl",
            "CLO": "http://purl.obolibrary.org/obo/clo.owl",
            "MONDO": "http://purl.obolibrary.org/obo/mondo.owl",
        }
    if cache_folder is None:
        cache_folder = os.path.expanduser("~/.cache/text2term")

    os.makedirs(cache_folder, exist_ok=True)

    source_terms = df[term_col].unique().tolist()
    if verbose:
        print("Searching for {} terms".format(len(source_terms)))

    # Iterate over ontologies
    id_maps = {}
    for ont in ont_dict.keys():
        print("\n", ont)
        cache_path = glob.glob(os.path.join(cache_folder, ont, f"{ont}*.pickle"))
        if len(cache_path) == 0:
            # Cache ontology if not already cached
            _ = text2term.cache_ontology(
                ontology_url=ont_dict[ont],
                ontology_acronym=ont,
                cache_folder=cache_folder
            )

        id_map = text2term.map_terms(
            source_terms=source_terms,
            target_ontology=ont,
            # mapper=Mapper.FUZZY,
            excl_deprecated=True,
            use_cache=True,
            cache_folder=cache_folder
        )
        id_map["ONT"] = id_map["Mapped Term CURIE"].str.split(":").str[0]
        id_maps[ont] = id_map
        df = get_top_ont_mappings_i(df, id_map, 
                                    prefix=ont, 
                                    top_n=top_n, 
                                    within_ont_ids_only=within_ont_ids_only,
                                    left_on=term_col)
    
    df = add_top_ont_mappings(df)
    
    return df


def add_top_ont_mappings(df, prefix="ONT"):
    df = df.copy()
    score_cols = [col for col in df.columns if col.endswith("_SCORE") and not col.startswith(prefix)]
    df[prefix+"_TOP"] = df[score_cols].idxmax(axis=1).str.split("_").str[0]
    # add the corresponding top _LABEL col
    def get_top_label(row):
        top_ont = row[prefix+"_TOP"]
        label_col = f"{top_ont}_LABEL"
        return row[label_col] if label_col in row else None
    df[prefix+"_LABEL"] = df.apply(get_top_label, axis=1)
    df[prefix+"_SCORE"] = df[score_cols].max(axis=1)
    return df
