"""Per-site and per-cell summary statistics for QC and downstream filtering."""

import anndata as ad
import numpy as np
from scipy.sparse import issparse

SITE_STATS_ALL = ("total", "mean", "median", "min", "max", "carrier_freq")


def compute_site_stats(
    adata: ad.AnnData,
    stats: list[str] | tuple[str, ...] = SITE_STATS_ALL,
    layer: str = "DP",
    carrier_source: str = "AD",
    min_alt_depth: int = 1,
) -> None:
    """
    Compute per-site depth summaries and carrier frequency, written to ``adata.var``.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with the layer named by ``layer`` (typically ``'DP'``).
    stats : list or tuple of str, default all
        Subset of ``{'total', 'mean', 'median', 'min', 'max', 'carrier_freq'}``.
    layer : str, default 'DP'
        Which depth layer to summarize over cells (axis=0).
    carrier_source : str, default 'AD'
        How to decide whether a cell carries the alt allele at a site when
        computing ``carrier_freq``:

        - ``'AD'`` — a cell is a carrier iff ``adata.layers['AD'] >= min_alt_depth``.
        - ``'X'`` — a cell is a carrier iff its genotype in ``adata.X`` is HET
          or HOM_ALT (``1`` or ``3``).
    min_alt_depth : int, default 1
        AD threshold used when ``carrier_source='AD'``.

    Returns
    -------
    None
        Writes columns on ``adata.var``:

        - ``dp_total``, ``dp_mean``, ``dp_median``, ``dp_min``, ``dp_max`` for the
          corresponding depth stats requested in ``stats``.
        - ``n_carriers`` and ``carrier_freq`` when ``'carrier_freq'`` is requested.

    Notes
    -----
    These columns feed :func:`cellspec.tl.filter_sites`. For sparse data,
    missing (cell, site) entries count as ``0`` for ``dp_min``/``dp_max`` — if
    you need "min across cells that have a call" first mask to ``DP > 0``.
    """
    unknown = set(stats) - set(SITE_STATS_ALL)
    if unknown:
        raise ValueError(f"Unknown site stats: {sorted(unknown)}. Valid: {list(SITE_STATS_ALL)}")

    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers. Available: {list(adata.layers)}")

    DP = adata.layers[layer]
    needs_dense = any(s in stats for s in ("median",))
    DP_arr = DP.toarray() if (issparse(DP) and needs_dense) else (DP if not issparse(DP) else None)

    def _axis0(fn_sparse, fn_dense):
        if DP_arr is not None:
            return fn_dense(DP_arr)
        return fn_sparse(DP)

    if "total" in stats:
        adata.var["dp_total"] = _axis0(
            lambda sp: np.ravel(sp.sum(axis=0)),
            lambda arr: arr.sum(axis=0),
        )
    if "mean" in stats:
        adata.var["dp_mean"] = _axis0(
            lambda sp: np.ravel(sp.sum(axis=0)) / sp.shape[0],
            lambda arr: arr.mean(axis=0),
        )
    if "median" in stats:
        # Median has no sparse implementation; _axis0 already materialized a dense view.
        adata.var["dp_median"] = np.median(DP_arr, axis=0)
    if "min" in stats:
        adata.var["dp_min"] = _axis0(
            lambda sp: np.ravel(sp.min(axis=0).toarray()),
            lambda arr: arr.min(axis=0),
        )
    if "max" in stats:
        adata.var["dp_max"] = _axis0(
            lambda sp: np.ravel(sp.max(axis=0).toarray()),
            lambda arr: arr.max(axis=0),
        )

    if "carrier_freq" in stats:
        if carrier_source == "AD":
            if "AD" not in adata.layers:
                raise ValueError("'AD' layer required when carrier_source='AD'.")
            AD = adata.layers["AD"]
            if issparse(AD):
                if min_alt_depth <= 1:
                    counts = np.ravel((AD > 0).sum(axis=0))
                else:
                    counts = (AD.toarray() >= min_alt_depth).sum(axis=0)
            else:
                counts = (AD >= min_alt_depth).sum(axis=0)
        elif carrier_source == "X":
            X = adata.X
            if issparse(X):
                # X stores only HET (1) and HOM_ALT (3); any nonzero is a carrier.
                counts = np.ravel((X > 0).sum(axis=0))
            else:
                counts = ((X == 1) | (X == 3)).sum(axis=0)
        else:
            raise ValueError(f"carrier_source must be 'AD' or 'X', got '{carrier_source}'")

        adata.var["n_carriers"] = counts.astype(np.int64)
        adata.var["carrier_freq"] = counts / adata.n_obs


def compute_cell_stats(
    adata: ad.AnnData,
    min_depth: int | None = None,
    max_depth: int | None = None,
    layer: str = "DP",
) -> None:
    """
    Compute per-cell depth summaries, written to ``adata.obs``.

    Parameters
    ----------
    adata : ad.AnnData
    min_depth : int, optional
        If given, also writes ``adata.obs['callable_sites']`` = count of sites
        with ``DP >= min_depth`` for each cell. This replaces the removed
        ``tl.compute_callable_sites``.
    max_depth : int, optional
        Upper bound paired with ``min_depth`` when counting callable sites.
    layer : str, default 'DP'
        Depth layer to summarize over sites (axis=1).

    Returns
    -------
    None
        Writes:

        - ``adata.obs['total_dp']`` — summed depth across all sites.
        - ``adata.obs['mean_dp_per_site']`` — ``total_dp / n_vars``.
        - ``adata.obs['callable_sites']`` — only if ``min_depth`` is given.
        - ``adata.uns['callable_sites_metadata']`` — thresholds used for the
          callable-site count.
    """
    if layer not in adata.layers:
        raise ValueError(f"Layer '{layer}' not found in adata.layers. Available: {list(adata.layers)}")

    DP = adata.layers[layer]
    if issparse(DP):
        total = np.ravel(DP.sum(axis=1))
        DP_arr = DP.toarray() if min_depth is not None else None
    else:
        total = DP.sum(axis=1)
        DP_arr = DP

    adata.obs["total_dp"] = total
    adata.obs["mean_dp_per_site"] = total / adata.n_vars

    if min_depth is not None:
        mask = DP_arr >= min_depth
        if max_depth is not None:
            mask &= DP_arr <= max_depth
        adata.obs["callable_sites"] = mask.sum(axis=1).astype(np.int64)
        adata.uns["callable_sites_metadata"] = {"min_depth": int(min_depth), "max_depth": max_depth}
