"""Site- and cell-level filtering from precomputed stats columns."""

import anndata as ad
import numpy as np

# Threshold-arg -> (adata.var column, comparison op, remediation hint).
# min_X means the data value must be >= threshold; max_X means <= threshold.
_SITE_FILTER_SPEC: dict[str, tuple[str, str, str]] = {
    "min_carrier_freq": ("carrier_freq", ">=", "compute_site_stats(stats=['carrier_freq'])"),
    "max_carrier_freq": ("carrier_freq", "<=", "compute_site_stats(stats=['carrier_freq'])"),
    "min_carrier_count": ("carrier_count", ">=", "compute_site_stats(stats=['carrier_freq'])"),
    "max_carrier_count": ("carrier_count", "<=", "compute_site_stats(stats=['carrier_freq'])"),
    "min_bulk_dp": ("dp_total", ">=", "compute_site_stats(stats=['total'])"),
    "max_bulk_dp": ("dp_total", "<=", "compute_site_stats(stats=['total'])"),
    "min_mean_dp": ("dp_mean", ">=", "compute_site_stats(stats=['mean'])"),
    "max_mean_dp": ("dp_mean", "<=", "compute_site_stats(stats=['mean'])"),
    "min_median_dp": ("dp_median", ">=", "compute_site_stats(stats=['median'])"),
    "max_median_dp": ("dp_median", "<=", "compute_site_stats(stats=['median'])"),
}

_CELL_FILTER_SPEC: dict[str, tuple[str, str]] = {
    "min_total_dp": ("total_dp", ">="),
    "max_total_dp": ("total_dp", "<="),
    "min_mean_dp_per_site": ("mean_dp_per_site", ">="),
    "max_mean_dp_per_site": ("mean_dp_per_site", "<="),
    "min_callable_sites": ("callable_sites", ">="),
    "max_callable_sites": ("callable_sites", "<="),
}


def filter_sites(
    adata: ad.AnnData,
    *,
    min_carrier_freq: float | None = None,
    max_carrier_freq: float | None = None,
    min_carrier_count: int | None = None,
    max_carrier_count: int | None = None,
    min_bulk_dp: float | None = None,
    max_bulk_dp: float | None = None,
    min_mean_dp: float | None = None,
    max_mean_dp: float | None = None,
    min_median_dp: float | None = None,
    max_median_dp: float | None = None,
    inplace: bool = True,
) -> ad.AnnData | None:
    """
    Filter variants using thresholds on precomputed ``adata.var`` columns.

    All thresholds are inclusive (``min_*`` requires ``>=``, ``max_*`` requires
    ``<=``). The relevant ``adata.var`` columns must already have been written
    by :func:`cellspec.tl.compute_site_stats`; otherwise a ``ValueError`` with
    the exact remediation call is raised.

    Each call appends the applied thresholds to
    ``adata.uns['filter_sites_history']`` as a dict, so downstream functions
    (e.g. :func:`cellspec.tl.compute_bulk_vaf`) can warn about inconsistent
    settings.

    Parameters
    ----------
    adata : ad.AnnData
    min_carrier_freq, max_carrier_freq : float, optional
        Bounds on ``adata.var['carrier_freq']`` (fraction of cells carrying alt).
    min_carrier_count, max_carrier_count : int, optional
        Bounds on ``adata.var['carrier_count']`` (absolute count).
    min_bulk_dp, max_bulk_dp : float, optional
        Bounds on ``adata.var['dp_total']`` (summed depth across cells).
    min_mean_dp, max_mean_dp : float, optional
        Bounds on ``adata.var['dp_mean']``.
    min_median_dp, max_median_dp : float, optional
        Bounds on ``adata.var['dp_median']``.
    inplace : bool, default True
        Modify ``adata`` in place or return a copy.

    Returns
    -------
    ad.AnnData or None
    """
    thresholds = {
        "min_carrier_freq": min_carrier_freq,
        "max_carrier_freq": max_carrier_freq,
        "min_carrier_count": min_carrier_count,
        "max_carrier_count": max_carrier_count,
        "min_bulk_dp": min_bulk_dp,
        "max_bulk_dp": max_bulk_dp,
        "min_mean_dp": min_mean_dp,
        "max_mean_dp": max_mean_dp,
        "min_median_dp": min_median_dp,
        "max_median_dp": max_median_dp,
    }

    if not inplace:
        adata = adata.copy()

    mask = np.ones(adata.n_vars, dtype=bool)
    applied: dict[str, float] = {}
    for arg, value in thresholds.items():
        if value is None:
            continue
        col, op, hint = _SITE_FILTER_SPEC[arg]
        if col not in adata.var.columns:
            raise ValueError(f"Threshold '{arg}' needs adata.var['{col}'], which is missing. Run spc.tl.{hint} first.")
        series = adata.var[col].to_numpy()
        if op == ">=":
            mask &= series >= value
        else:
            mask &= series <= value
        applied[arg] = float(value)

    adata._inplace_subset_var(mask)

    history = list(adata.uns.get("filter_sites_history", []))
    history.append(applied)
    adata.uns["filter_sites_history"] = history

    if not inplace:
        return adata


def filter_cells(
    adata: ad.AnnData,
    *,
    min_total_dp: float | None = None,
    max_total_dp: float | None = None,
    min_mean_dp_per_site: float | None = None,
    max_mean_dp_per_site: float | None = None,
    min_callable_sites: int | None = None,
    max_callable_sites: int | None = None,
    inplace: bool = True,
) -> ad.AnnData | None:
    """
    Filter cells using thresholds on precomputed ``adata.obs`` columns.

    All thresholds are inclusive. The relevant ``adata.obs`` columns must
    already have been written by :func:`cellspec.tl.compute_cell_stats`;
    otherwise a ``ValueError`` with the exact remediation call is raised.

    Each call appends the applied thresholds to
    ``adata.uns['filter_cells_history']``.

    Parameters
    ----------
    adata : ad.AnnData
    min_total_dp, max_total_dp : float, optional
        Bounds on ``adata.obs['total_dp']``.
    min_mean_dp_per_site, max_mean_dp_per_site : float, optional
        Bounds on ``adata.obs['mean_dp_per_site']``.
    min_callable_sites, max_callable_sites : int, optional
        Bounds on ``adata.obs['callable_sites']``.
    inplace : bool, default True

    Returns
    -------
    ad.AnnData or None
    """
    thresholds = {
        "min_total_dp": min_total_dp,
        "max_total_dp": max_total_dp,
        "min_mean_dp_per_site": min_mean_dp_per_site,
        "max_mean_dp_per_site": max_mean_dp_per_site,
        "min_callable_sites": min_callable_sites,
        "max_callable_sites": max_callable_sites,
    }

    if not inplace:
        adata = adata.copy()

    mask = np.ones(adata.n_obs, dtype=bool)
    applied: dict[str, float] = {}
    for arg, value in thresholds.items():
        if value is None:
            continue
        col, op = _CELL_FILTER_SPEC[arg]
        if col not in adata.obs.columns:
            raise ValueError(
                f"Threshold '{arg}' needs adata.obs['{col}'], which is missing. "
                f"Run spc.tl.compute_cell_stats(min_depth=...) first "
                f"(min_depth required for 'callable_sites')."
                if col == "callable_sites"
                else f"Threshold '{arg}' needs adata.obs['{col}'], which is missing. "
                f"Run spc.tl.compute_cell_stats() first."
            )
        series = adata.obs[col].to_numpy()
        if op == ">=":
            mask &= series >= value
        else:
            mask &= series <= value
        applied[arg] = float(value)

    adata._inplace_subset_obs(mask)

    history = list(adata.uns.get("filter_cells_history", []))
    history.append(applied)
    adata.uns["filter_cells_history"] = history

    if not inplace:
        return adata
