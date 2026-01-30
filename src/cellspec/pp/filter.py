"""Filter variants and cells in AnnData objects."""

import numpy as np
import anndata as ad
from scipy.sparse import issparse
from typing import Optional

from ..utils.context import parse_variant_id


def filter_variants(
    adata: ad.AnnData,
    min_cells: Optional[int] = None,
    min_fraction: Optional[float] = None,
    inplace: bool = True
) -> Optional[ad.AnnData]:
    """
    Filter variants based on presence across cells/samples.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with variants
    min_cells : int, optional
        Minimum number of cells/samples a variant must be present in
    min_fraction : float, optional
        Minimum fraction of cells/samples a variant must be present in (0-1)
    inplace : bool, default True
        Modify adata in place or return copy

    Returns
    -------
    ad.AnnData or None
        Filtered AnnData (if inplace=False), otherwise None

    Examples
    --------
    >>> import cellspec as spc
    >>> # Keep variants present in at least 3 cells
    >>> spc.pp.filter_variants(adata, min_cells=3)

    >>> # Keep variants present in at least 1% of cells
    >>> spc.pp.filter_variants(adata, min_fraction=0.01)

    Notes
    -----
    Presence is determined by .X (binary presence/absence matrix).
    """
    if not inplace:
        adata = adata.copy()

    # Get presence matrix
    X = adata.X
    if issparse(X):
        presence_counts = np.array(X.sum(axis=0)).flatten()
    else:
        presence_counts = X.sum(axis=0)

    # Build mask
    mask = np.ones(adata.n_vars, dtype=bool)

    if min_cells is not None:
        mask &= presence_counts >= min_cells

    if min_fraction is not None:
        min_cells_from_fraction = int(np.ceil(adata.n_obs * min_fraction))
        mask &= presence_counts >= min_cells_from_fraction

    # Filter
    adata._inplace_subset_var(mask)

    if not inplace:
        return adata


def filter_cells(
    adata: ad.AnnData,
    min_variants: Optional[int] = None,
    max_variants: Optional[int] = None,
    inplace: bool = True
) -> Optional[ad.AnnData]:
    """
    Filter cells/samples based on number of variants detected.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with variants
    min_variants : int, optional
        Minimum number of variants a cell must have
    max_variants : int, optional
        Maximum number of variants a cell must have
    inplace : bool, default True
        Modify adata in place or return copy

    Returns
    -------
    ad.AnnData or None
        Filtered AnnData (if inplace=False), otherwise None

    Examples
    --------
    >>> import cellspec as spc
    >>> # Keep cells with at least 10 variants
    >>> spc.pp.filter_cells(adata, min_variants=10)

    >>> # Remove cells with too many variants (potential doublets)
    >>> spc.pp.filter_cells(adata, min_variants=10, max_variants=1000)

    Notes
    -----
    Variant counts are from .X (binary presence/absence matrix).
    """
    if not inplace:
        adata = adata.copy()

    # Get variant counts per cell
    X = adata.X
    if issparse(X):
        variant_counts = np.array(X.sum(axis=1)).flatten()
    else:
        variant_counts = X.sum(axis=1)

    # Build mask
    mask = np.ones(adata.n_obs, dtype=bool)

    if min_variants is not None:
        mask &= variant_counts >= min_variants

    if max_variants is not None:
        mask &= variant_counts <= max_variants

    # Filter
    adata._inplace_subset_obs(mask)

    if not inplace:
        return adata


def filter_by_coverage(
    adata: ad.AnnData,
    min_depth: int,
    min_fraction: float = 1.0,
    inplace: bool = True
) -> Optional[ad.AnnData]:
    """
    Filter variants based on sequencing depth (coverage) across cells/samples.

    Keeps only variants where a minimum fraction of cells/samples meet the
    depth threshold. By default, requires ALL cells to have sufficient coverage.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with variants (must have 'DP' layer)
    min_depth : int
        Minimum depth (DP) threshold per cell
    min_fraction : float, default 1.0
        Minimum fraction of cells that must meet the depth threshold (0-1).
        - 1.0 (default): ALL cells must have DP >= min_depth
        - 0.8: At least 80% of cells must have DP >= min_depth
        - 0.5: At least 50% of cells must have DP >= min_depth
    inplace : bool, default True
        Modify adata in place or return copy

    Returns
    -------
    ad.AnnData or None
        Filtered AnnData (if inplace=False), otherwise None

    Examples
    --------
    >>> import cellspec as spc
    >>> # Keep only variants with DP >= 10 in ALL cells (default)
    >>> spc.pp.filter_by_coverage(adata, min_depth=10)
    >>>
    >>> # Keep variants with DP >= 10 in at least 80% of cells
    >>> spc.pp.filter_by_coverage(adata, min_depth=10, min_fraction=0.8)
    >>>
    >>> # Keep variants with DP >= 20 in at least 50% of cells
    >>> spc.pp.filter_by_coverage(adata, min_depth=20, min_fraction=0.5)

    Notes
    -----
    This function is useful for ensuring high-quality variant calls by removing
    sites with insufficient coverage. The default (min_fraction=1.0) is stringent
    and ensures every cell has adequate depth at retained variants.

    Requires the 'DP' layer in adata.layers, which is automatically created by
    spc.pp.load_vcf() when loading VCF files.
    """
    if not inplace:
        adata = adata.copy()

    # Check for DP layer
    if 'DP' not in adata.layers:
        raise ValueError(
            "'DP' layer not found in adata.layers. "
            "This layer is required for coverage filtering and is created by spc.pp.load_vcf()."
        )

    # Get DP matrix
    DP = adata.layers['DP']
    if issparse(DP):
        DP = DP.toarray()

    # Count how many cells meet the depth threshold for each variant
    meets_threshold = DP >= min_depth
    n_cells_passing = meets_threshold.sum(axis=0)

    # Calculate minimum number of cells required
    min_cells_required = int(np.ceil(adata.n_obs * min_fraction))

    # Create mask for variants that pass
    mask = n_cells_passing >= min_cells_required

    # Filter
    adata._inplace_subset_var(mask)

    if not inplace:
        return adata


def filter_to_snps(
    adata: ad.AnnData,
    chrom_prefix: Optional[str] = None,
    inplace: bool = True
) -> Optional[ad.AnnData]:
    """
    Filter to only single nucleotide variants (SNPs).

    Removes indels and multi-allelic sites, keeping only sites where
    both reference and alternate alleles are single nucleotides.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with variants
    chrom_prefix : str, optional
        Chromosome prefix for parsing variant IDs. If None (default), accepts
        any chromosome naming (e.g., 'chr1', 'I', '1'). Use 'chr' for
        human/mouse data if you want to be strict.
    inplace : bool, default True
        Modify adata in place or return copy

    Returns
    -------
    ad.AnnData or None
        Filtered AnnData (if inplace=False), otherwise None

    Examples
    --------
    >>> import cellspec as spc
    >>> # Default: works with any chromosome naming
    >>> adata = spc.pp.load_vcf('variants.vcf.gz')
    >>> spc.pp.filter_to_snps(adata)
    >>> print(f"Retained {adata.n_vars} SNPs")
    >>>
    >>> # For human/mouse data, can specify prefix
    >>> spc.pp.filter_to_snps(adata, chrom_prefix='chr')

    Notes
    -----
    This is typically run before trinucleotide context annotation,
    as trinuc contexts are only defined for SNPs.

    Works with any organism's chromosome naming:
    - Human/mouse: 'chr1-12345-A>T'
    - C. elegans: 'I-12345-A>T'
    - Drosophila: '2L-12345-A>T'
    """
    if not inplace:
        adata = adata.copy()

    # Check each variant
    is_snp = []
    for variant_id in adata.var_names:
        parsed = parse_variant_id(variant_id, chrom_prefix)
        if parsed is None:
            is_snp.append(False)
        else:
            chrom, pos, ref_base, alt_base = parsed
            # SNP: both ref and alt are single nucleotides (A, C, G, or T)
            # Exclude variants with missing ALT (represented as ".")
            is_valid_base = lambda b: len(b) == 1 and b in "ACGT"
            is_snp.append(is_valid_base(ref_base) and is_valid_base(alt_base))

    is_snp = np.array(is_snp)

    # Filter
    adata._inplace_subset_var(is_snp)

    if not inplace:
        return adata
