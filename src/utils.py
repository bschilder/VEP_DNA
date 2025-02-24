from typing import Optional, Union


def intersect(x: Union[list, set], 
              y: Union[list, set], 
              as_list: bool = True) -> Union[list, set]:
    """
    Intersect two lists or sets.
    
    Args:
        x: First list or set to intersect
        y: Second list or set to intersect 
        as_list: If True, returns result as list. If False, returns as set. Default True.
    
    Returns:
        list or set: Intersection of x and y, returned as list if as_list=True, otherwise as set
    """
    if as_list:
        return list(set(x) & set(y))
    else:
        return set(x) & set(y)

def _make_palette(values: list, 
                  palette: str) -> dict:
    """
    Create a color palette dictionary mapping values to colors.
    
    Args:
        values: List of values to map to colors
        palette: Name of seaborn color palette to use (e.g. 'bwr_r' for blue-white-red reversed)
    
    Returns:
        dict: Dictionary mapping each value to a hex color code
    """
    # sample 4 colors from a palette that goes from hot to cold
    import seaborn as sns
    return dict(zip(values, sns.color_palette(palette, len(values)).as_hex()))

def get_clinsig_palette(values=['path', 'likely_path', 'likely_benign', 'benign'],
                         palette='bwr_r'):
    """
    Create a color palette dictionary mapping clinical significance values to colors.
    
    Args:
        values: List of clinical significance values to map to colors. Default is
               ['path', 'likely_path', 'likely_benign', 'benign']
        palette: Name of seaborn color palette to use. Default is 'bwr_r' for 
                blue-white-red reversed palette.
    
    Returns:
        dict: Dictionary mapping each clinical significance value to a hex color code
    """
    return _make_palette(values, palette) 
