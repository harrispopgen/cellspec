"""Annotate variants with additional information."""

import anndata as ad
import numpy as np
from pyfaidx import Fasta
from tqdm import tqdm

from cellspec.utils.context import parse_variant_id, strand_standardize_trinuc


def annotate_contexts(
    adata: ad.AnnData,
    fasta_path: str,
    chrom_prefix: str | None = None,
    show_progress: bool = True,
    inplace: bool = True,
) -> ad.AnnData | None:
    """
    Annotate variants with trinucleotide contexts.

    Adds three columns to adata.var:
    - 'anc': Ancestral (reference) trinucleotide
    - 'der': Derived (alternate) trinucleotide
    - 'trinuc_type': Standardized COSMIC 96-type label (e.g., 'ACA>AAA')

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with variants
    fasta_path : str
        Path to reference genome FASTA file
    chrom_prefix : str, optional
        Chromosome prefix in variant IDs. If None (default), accepts any
        chromosome naming. Use 'chr' for human/mouse if needed.
    show_progress : bool, default True
        Show progress bar
    inplace : bool, default True
        Modify adata in place or return copy

    Returns
    -------
    ad.AnnData or None
        Modified AnnData (if inplace=False), otherwise None

    Examples
    --------
    >>> import cellspec as spc
    >>> # Works with any organism
    >>> adata = spc.pp.load_vcf("variants.vcf.gz")
    >>> spc.pp.annotate_contexts(adata, fasta_path="reference.fa")
    >>> print(adata.var["trinuc_type"].head())
    >>>
    >>> # Human data with 'chr' prefix
    >>> spc.pp.annotate_contexts(adata, fasta_path="hg38.fa", chrom_prefix="chr")

    Notes
    -----
    Trinucleotide contexts are strand-standardized following COSMIC convention:
    - Always report pyrimidine (C or T) as the reference base
    - If reference is purine (A or G), reverse complement the trinucleotide
    - This ensures consistent 96-channel representation

    Works with any organism's chromosome naming:
    - Human/mouse: 'chr1', 'chr2', etc.
    - C. elegans: 'I', 'II', 'III', 'IV', 'V', 'X'
    - Drosophila: '2L', '2R', '3L', '3R', 'X'
    """
    if not inplace:
        adata = adata.copy()

    # Load reference genome
    ref = Fasta(fasta_path, sequence_always_upper=True)

    # Initialize lists for annotations
    ancs = []
    ders = []
    types = []

    # Annotate each variant
    iterator = (
        tqdm(adata.var_names, desc="Annotating trinuc contexts", unit=" sites") if show_progress else adata.var_names
    )

    for variant_id in iterator:
        parsed = parse_variant_id(variant_id, chrom_prefix)

        if parsed is None:
            ancs.append(np.nan)
            ders.append(np.nan)
            types.append(np.nan)
            continue

        chrom, pos, ref_base, alt_base = parsed

        # Check if it's a SNP (single nucleotide)
        if len(ref_base) != 1 or len(alt_base) != 1:
            # Not a SNP (indel or MNP)
            ancs.append(np.nan)
            ders.append(np.nan)
            types.append(np.nan)
            continue

        try:
            # Extract 3-mer centered on variant
            # pos is 1-based VCF position, need to fetch [pos-1, pos, pos+1] in 1-based
            # which is [pos-2, pos-1, pos] in 0-based indexing
            anc_trinuc = ref[chrom][pos - 2 : pos + 1].seq

            if len(anc_trinuc) != 3:
                # Edge case: variant too close to chromosome end
                ancs.append(np.nan)
                ders.append(np.nan)
                types.append(np.nan)
                continue

            # Create derived trinucleotide
            der_trinuc = anc_trinuc[0] + alt_base + anc_trinuc[2]

            # Strand standardize
            trinuc_type = strand_standardize_trinuc(anc_trinuc, alt_base)

            ancs.append(anc_trinuc)
            ders.append(der_trinuc)
            types.append(trinuc_type)

        except (KeyError, IndexError):
            # Chromosome not found or index out of range
            ancs.append(np.nan)
            ders.append(np.nan)
            types.append(np.nan)

    # Add to adata.var
    adata.var["anc"] = ancs
    adata.var["der"] = ders
    adata.var["trinuc_type"] = types

    if not inplace:
        return adata
