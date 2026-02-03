"""Variant allele frequency calculations."""

import anndata as ad
import numpy as np
from scipy.sparse import issparse
from tqdm import tqdm

from cellspec.utils.context import compute_vaf


def add_vaf_layer(
    adata: ad.AnnData,
    target_dp: int = 10,
    show_progress: bool = False,
    inplace: bool = True,
) -> ad.AnnData | None:
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
    >>> adata = spc.pp.load_vcf("variants.vcf.gz")
    >>> spc.tl.add_vaf_layer(adata, target_dp=10)
    >>> print(adata.layers["VAF"])

    Notes
    -----
    VAF is computed per cell/sample independently. High-coverage sites are
    downsampled to target_dp using hypergeometric sampling to normalize
    for coverage differences.

    Requires 'AD' and 'DP' layers in adata.layers.
    """
    if not inplace:
        adata = adata.copy()

    if "AD" not in adata.layers or "DP" not in adata.layers:
        raise ValueError("AnnData must have 'AD' and 'DP' layers. Run spc.pp.load_vcf() first.")

    # Get AD and DP
    ad_data = adata.layers["AD"]
    dp_data = adata.layers["DP"]

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
    adata.layers["VAF"] = vaf_matrix

    if not inplace:
        return adata


def compute_bulk_vaf(
    adata: ad.AnnData,
    target_dp: int | None = None,
    inplace: bool = True,
) -> ad.AnnData | None:
    """
    Compute pseudobulk VAF for each variant across all samples.

    Sums AD and DP across all samples to create pseudobulk counts, then computes
    VAF with hypergeometric downsampling for coverage normalization. Stores result
    in adata.var['bulk_vaf'].

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with 'AD' and 'DP' layers
    target_dp : int, optional
        Target depth for downsampling. If None (default), uses the minimum
        total depth across all sites.
    inplace : bool, default True
        Modify adata in place or return copy

    Returns
    -------
    ad.AnnData or None
        Modified AnnData (if inplace=False), otherwise None

    Examples
    --------
    >>> import cellspec as spc
    >>> adata = spc.pp.load_vcf("variants.vcf.gz")
    >>> # Use default target_dp (minimum depth across sites)
    >>> spc.tl.compute_bulk_vaf(adata)
    >>> print(adata.var["bulk_vaf"])
    >>>
    >>> # Use custom target_dp
    >>> spc.tl.compute_bulk_vaf(adata, target_dp=50)

    Notes
    -----
    Pseudobulk VAF is computed by summing AD and DP across all samples for each
    variant, then applying hypergeometric downsampling to normalize for coverage
    differences. This is useful for identifying high-confidence variants that
    are supported across multiple samples.

    Requires 'AD' and 'DP' layers in adata.layers.
    """
    if not inplace:
        adata = adata.copy()

    if "AD" not in adata.layers or "DP" not in adata.layers:
        raise ValueError("AnnData must have 'AD' and 'DP' layers. Run spc.pp.load_vcf() first.")

    # Get AD and DP
    ad_data = adata.layers["AD"]
    dp_data = adata.layers["DP"]

    # Handle sparse matrices - sum along samples (axis=0)
    if issparse(ad_data):
        bulk_ad = np.ravel(ad_data.sum(axis=0))
    else:
        bulk_ad = np.sum(ad_data, axis=0)

    if issparse(dp_data):
        bulk_dp = np.ravel(dp_data.sum(axis=0))
    else:
        bulk_dp = np.sum(dp_data, axis=0)

    # If target_dp not specified, use minimum depth across all sites
    if target_dp is None:
        target_dp = int(np.min(bulk_dp[bulk_dp > 0]))
        if target_dp == 0:
            raise ValueError("All sites have zero depth. Cannot compute bulk VAF.")

    # Compute VAF with hypergeometric downsampling
    bulk_vaf = compute_vaf(bulk_ad, bulk_dp, target_dp=target_dp)

    # Store in adata.var
    adata.var["bulk_vaf"] = bulk_vaf

    if not inplace:
        return adata
