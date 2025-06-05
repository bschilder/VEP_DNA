from Bio.Seq import Seq

def dna_to_rna(seq):
    """
    Convert a DNA sequence to an RNA sequence.

    Args:
        seq: str, the DNA sequence to convert.

    Returns:
        str, the RNA sequence.

    Example:
        >>> dna_to_rna('ATCG')
        'AUCG'
    """
    return str(Seq(seq).transcribe()).lower()