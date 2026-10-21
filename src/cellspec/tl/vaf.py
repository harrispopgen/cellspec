"""Variant allele frequency calculations."""

import warnings

import anndata as ad
import numpy as np
from scipy.sparse import issparse
from tqdm import tqdm

from cellspec.utils.context import compute_vaf


def _default_min_dp_from_entries(dp_values: np.ndarray) -> int:
    """25th percentile of nonzero DP entries, floored at 1."""
    nz = dp_values[dp_values > 0]
    if nz.size == 0:
        raise ValueError("Cannot derive default min_dp: all DP entries are zero.")
    return max(1, int(np.percentile(nz, 25)))


def add_vaf_layer(
    adata: ad.AnnData,
    min_dp: int | None = None,
    normalize: bool = False,
    nan_warn_threshold: float = 0.5,
    show_progress: bool = False,
    inplace: bool = True,
) -> ad.AnnData | None:
    """
    Add a per-cell VAF layer to ``adata.layers['VAF']``.

    Default behavior is **raw** VAF ``= AD / DP`` for (cell, site) entries
    meeting ``min_dp``, and ``NaN`` for entries below the threshold. Set
    ``normalize=True`` to apply hypergeometric downsampling at ``min_dp`` for
    coverage-uniform estimates.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with ``AD`` and ``DP`` layers (from :func:`cellspec.pp.load_vcf`).
    min_dp : int, optional
        Minimum DP floor applied per (cell, site) entry. If ``None``, defaults
        to the 25th percentile of nonzero ``DP`` entries (floored at 1).
    normalize : bool, default False
        If True, hypergeometrically downsample alt-depth to ``min_dp`` before
        dividing. If False, use raw ``AD / DP``.
    nan_warn_threshold : float, default 0.5
        If more than this fraction of nonzero-DP entries fail the ``min_dp``
        threshold (and thus become ``NaN``), a ``UserWarning`` is emitted.
    show_progress : bool, default False
        Show a per-cell progress bar (can be slow for very large datasets).
    inplace : bool, default True
        Modify ``adata`` in place or return a copy.

    Returns
    -------
    ad.AnnData or None

    Examples
    --------
    >>> import cellspec as spc
    >>> adata = spc.pp.load_vcf("variants.vcf.gz")
    >>> # Raw per-cell VAF with min_dp picked from the 25th percentile
    >>> spc.tl.add_vaf_layer(adata)
    >>>
    >>> # Explicit min_dp with hypergeometric normalization
    >>> spc.tl.add_vaf_layer(adata, min_dp=20, normalize=True)
    """
    if not inplace:
        adata = adata.copy()

    if "AD" not in adata.layers or "DP" not in adata.layers:
        raise ValueError("AnnData must have 'AD' and 'DP' layers. Run spc.pp.load_vcf() first.")

    ad_layer = adata.layers["AD"]
    dp_layer = adata.layers["DP"]
    ad_arr = ad_layer.toarray() if issparse(ad_layer) else ad_layer
    dp_arr = dp_layer.toarray() if issparse(dp_layer) else dp_layer

    if min_dp is None:
        min_dp = _default_min_dp_from_entries(dp_arr.ravel())

    n_cells = adata.n_obs
    vaf_matrix = np.full((n_cells, adata.n_vars), np.nan, dtype=np.float32)

    iterator = tqdm(range(n_cells), desc="Computing VAF", unit="cell") if show_progress else range(n_cells)
    for cell_idx in iterator:
        vaf_matrix[cell_idx, :] = compute_vaf(
            ad_arr[cell_idx, :], dp_arr[cell_idx, :], min_dp=min_dp, normalize=normalize
        )

    adata.layers["VAF"] = vaf_matrix
    adata.uns["vaf_metadata"] = {"min_dp": int(min_dp), "normalize": bool(normalize)}

    # Warn if many entries failed the min_dp floor. We measure the fail-rate
    # over "callable attempts" (DP > 0) so that layer sparsity doesn't
    # dominate the signal.
    callable_entries = dp_arr > 0
    n_callable = int(callable_entries.sum())
    if n_callable > 0:
        n_below = int(((dp_arr > 0) & (dp_arr < min_dp)).sum())
        failed_fraction = n_below / n_callable
        if failed_fraction > nan_warn_threshold:
            warnings.warn(
                f"{failed_fraction:.1%} of (cell, site) entries with DP > 0 fell below "
                f"min_dp={min_dp} and were set to NaN. Consider lowering min_dp or "
                f"pre-filtering sites with spc.tl.filter_sites(min_bulk_dp=...).",
                UserWarning,
                stacklevel=2,
            )

    if not inplace:
        return adata


def compute_bulk_vaf(
    adata: ad.AnnData,
    min_dp: int | None = None,
    normalize: bool = False,
    key: str = "bulk_vaf",
    inplace: bool = True,
) -> ad.AnnData | None:
    """
    Compute pseudobulk VAF per site, stored in ``adata.var[key]``.

    AD and DP are summed across cells; the resulting per-site totals go through
    :func:`cellspec.utils.compute_vaf` with the same ``min_dp`` / ``normalize``
    semantics. Sites with bulk DP below ``min_dp`` are stored as ``NaN``.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with ``AD`` and ``DP`` layers.
    min_dp : int, optional
        Minimum bulk DP floor. If ``None``, defaults to the 25th percentile of
        nonzero per-site bulk DP (floored at 1). If a prior
        :func:`cellspec.tl.filter_sites` call applied ``min_bulk_dp``, a value
        of ``min_dp`` below that threshold emits a warning (the filter has
        already removed the affected sites, so the lower floor is a no-op).
    normalize : bool, default False
        Apply hypergeometric downsampling at ``min_dp``.
    key : str, default 'bulk_vaf'
        Destination column in ``adata.var``. Use distinct keys (e.g.
        ``'bulk_vaf_raw'`` and ``'bulk_vaf_norm'``) to store raw and
        normalized VAFs side-by-side.
    inplace : bool, default True

    Returns
    -------
    ad.AnnData or None

    Examples
    --------
    >>> import cellspec as spc
    >>> spc.tl.compute_bulk_vaf(adata, key="bulk_vaf_raw")
    >>> spc.tl.compute_bulk_vaf(adata, normalize=True, key="bulk_vaf_norm")
    """
    if not inplace:
        adata = adata.copy()

    if "AD" not in adata.layers or "DP" not in adata.layers:
        raise ValueError("AnnData must have 'AD' and 'DP' layers. Run spc.pp.load_vcf() first.")

    ad_layer = adata.layers["AD"]
    dp_layer = adata.layers["DP"]

    if issparse(ad_layer):
        bulk_ad = np.ravel(ad_layer.sum(axis=0))
    else:
        bulk_ad = ad_layer.sum(axis=0)
    if issparse(dp_layer):
        bulk_dp = np.ravel(dp_layer.sum(axis=0))
    else:
        bulk_dp = dp_layer.sum(axis=0)

    if min_dp is None:
        min_dp = _default_min_dp_from_entries(bulk_dp)

    # Warn if min_dp is below a previously-applied filter_sites DP threshold.
    for applied in adata.uns.get("filter_sites_history", []):
        prior_floor = applied.get("min_bulk_dp")
        if prior_floor is not None and min_dp < prior_floor:
            warnings.warn(
                f"min_dp={min_dp} is below a previously applied filter_sites "
                f"threshold min_bulk_dp={prior_floor}. Sites below that floor "
                f"are already gone, so the lower min_dp has no additional effect.",
                UserWarning,
                stacklevel=2,
            )
            break

    bulk_vaf = compute_vaf(bulk_ad, bulk_dp, min_dp=min_dp, normalize=normalize)

    adata.var[key] = bulk_vaf
    adata.uns[f"{key}_metadata"] = {"min_dp": int(min_dp), "normalize": bool(normalize)}

    if not inplace:
        return adata
