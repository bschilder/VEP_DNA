import os
import pooch



def download_clinvar(vcf_url = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/vcf_GRCh38/clinvar.vcf.gz"):
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