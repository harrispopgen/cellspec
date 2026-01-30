"""Annotate variants with additional information."""

import numpy as np
import anndata as ad
from pyfaidx import Fasta
from tqdm import tqdm
from scipy.sparse import issparse
from typing import Optional

from ..utils.context import parse_variant_id, strand_standardize_trinuc, compute_vaf


def annotate_contexts(
    adata: ad.AnnData,
    fasta_path: str,
    chrom_prefix: Optional[str] = None,
    show_progress: bool = True,
    inplace: bool = True
) -> Optional[ad.AnnData]:
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
    >>> adata = spc.pp.load_vcf('variants.vcf.gz')
    >>> spc.pp.annotate_contexts(adata, fasta_path='reference.fa')
    >>> print(adata.var['trinuc_type'].head())
    >>>
    >>> # Human data with 'chr' prefix
    >>> spc.pp.annotate_contexts(adata, fasta_path='hg38.fa', chrom_prefix='chr')

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
    iterator = tqdm(adata.var_names, desc="Annotating trinuc contexts", unit=" sites") if show_progress else adata.var_names

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


def add_vaf_layer(
    adata: ad.AnnData,
    target_dp: int = 10,
    show_progress: bool = False,
    inplace: bool = True
) -> Optional[ad.AnnData]:
    """
    Add VAF (Variant Allele Frequency) layer to AnnData object.

    Computes VAF with hypergeometric downsampling for coverage normalization.
    Adds 'VAF' layer to adata.layers.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with 'AD' and 'DP' layers
    target_dp : int, default 10
        Target depth for downsampling
    show_progress : bool, default False
        Show progress bar (can be slow for large datasets)
    inplace : bool, default True
        Modify adata in place or return copy

    Returns
    -------
    ad.AnnData or None
        Modified AnnData (if inplace=False), otherwise None

    Examples
    --------
    >>> import cellspec as spc
    >>> adata = spc.pp.load_vcf('variants.vcf.gz')
    >>> spc.pp.add_vaf_layer(adata, target_dp=10)
    >>> print(adata.layers['VAF'])

    Notes
    -----
    VAF is computed per cell/sample independently. High-coverage sites are
    downsampled to target_dp using hypergeometric sampling to normalize
    for coverage differences.

    Requires 'AD' and 'DP' layers in adata.layers.
    """
    if not inplace:
        adata = adata.copy()

    if 'AD' not in adata.layers or 'DP' not in adata.layers:
        raise ValueError("AnnData must have 'AD' and 'DP' layers. Run spc.pp.load_vcf() first.")

    # Get AD and DP
    ad_data = adata.layers['AD']
    dp_data = adata.layers['DP']

    # Handle sparse matrices
    if issparse(ad_data):
        ad_data = ad_data.toarray()
    if issparse(dp_data):
        dp_data = dp_data.toarray()

    # Compute VAF for each cell (column)
    n_cells = adata.n_obs
    n_vars = adata.n_vars

    vaf_matrix = np.zeros((n_cells, n_vars), dtype=np.float32)

    iterator = tqdm(range(n_cells), desc="Computing VAF", unit="cell") if show_progress else range(n_cells)

    for cell_idx in iterator:
        ad_col = ad_data[cell_idx, :]
        dp_col = dp_data[cell_idx, :]
        vaf_matrix[cell_idx, :] = compute_vaf(ad_col, dp_col, target_dp=target_dp)

    # Add to layers (keep as dense for now, can convert to sparse if needed)
    adata.layers['VAF'] = vaf_matrix

    if not inplace:
        return adata
