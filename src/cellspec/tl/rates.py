"""Compute mutation rates and callable sites."""

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from tqdm import tqdm


def compute_callable_sites(
    adata: ad.AnnData, min_depth: int, max_depth: int | None = None, key: str = "callable", show_progress: bool = False
) -> pd.Series:
    """
    Compute the number of callable sites per cell/sample.

    A site is "callable" if it has adequate sequencing depth. This is used
    for rate normalization: mutations per callable site.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with variants
    min_depth : int
        Minimum depth for a site to be callable
    max_depth : int, optional
        Maximum depth (exclude PCR artifacts)
    key : str, default 'callable'
        Key for storing in adata.obs
    show_progress : bool, default False
        Show progress bar

    Returns
    -------
    pd.Series
        Number of callable sites per cell/sample
        Also stored in adata.obs[f'{key}_sites']

    Examples
    --------
    >>> import cellspec as spc
    >>> callable = spc.tl.compute_callable_sites(adata, min_depth=10, max_depth=200)
    >>> print(callable.head())

    Notes
    -----
    This counts how many variants have adequate depth in each cell/sample,
    which serves as the denominator for mutation rate calculations.
    """
    if "DP" not in adata.layers:
        raise ValueError("adata.layers must contain 'DP' (total depth)")

    # Get depth matrix
    DP = adata.layers["DP"]
    if issparse(DP):
        DP = DP.toarray()

    n_cells = adata.n_obs
    callable_counts = np.zeros(n_cells, dtype=int)

    # Compute callable sites per cell
    iterator = tqdm(range(n_cells), desc="Computing callable sites", unit="cell") if show_progress else range(n_cells)

    for cell_idx in iterator:
        mask = DP[cell_idx, :] >= min_depth
        if max_depth is not None:
            mask &= DP[cell_idx, :] <= max_depth
        callable_counts[cell_idx] = mask.sum()

    # Create Series
    callable_series = pd.Series(callable_counts, index=adata.obs_names, name=f"{key}_sites")

    # Store in adata.obs
    adata.obs[f"{key}_sites"] = callable_series

    # Store metadata
    adata.uns[f"{key}_metadata"] = {"min_depth": min_depth, "max_depth": max_depth}

    return callable_series


def compute_rates(
    adata: ad.AnnData, spectrum_key: str, callable_key: str = "callable", rate_key: str | None = None
) -> pd.DataFrame:
    """
    Compute mutation rates (mutations per callable site) from spectrum.

    This is a convenience function equivalent to::

        normalize_spectrum(
            adata, key=spectrum_key, method="obs_column", obs_column=f"{callable_key}_sites", output_key=rate_key
        )

    For more flexible normalization (by coverage, read count, custom vectors),
    use spc.tl.normalize_spectrum() instead.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with computed spectrum and callable sites
    spectrum_key : str
        Key for spectrum in adata.obsm (e.g., 'spectrum_somatic')
    callable_key : str, default 'callable'
        Key for callable sites in adata.obs (e.g., 'callable_sites')
    rate_key : str, optional
        Key for storing rates. If None, uses f'{spectrum_key}_rate'

    Returns
    -------
    pd.DataFrame
        Mutation rates (cells × 96 contexts)
        Also stored in adata.obsm[rate_key]

    Examples
    --------
    >>> import cellspec as spc
    >>> # Compute spectrum and callable sites
    >>> spc.tl.compute_spectrum(adata, min_depth=10, key="somatic")
    >>> spc.tl.compute_callable_sites(adata, min_depth=10, key="callable")
    >>> # Compute rates
    >>> rates = spc.tl.compute_rates(adata, spectrum_key="somatic")

    Notes
    -----
    Rate = mutations / callable_sites for each cell and each trinuc context.
    Cells with 0 callable sites get rate = 0.

    See Also
    --------
    normalize_spectrum : More general normalization function
    """
    if rate_key is None:
        rate_key = f"{spectrum_key}_rate"

    # Get spectrum
    spectrum_obsm_key = f"spectrum_{spectrum_key}"
    if spectrum_obsm_key not in adata.obsm:
        raise ValueError(
            f"'{spectrum_obsm_key}' not found in adata.obsm. "
            f"Run spc.tl.compute_spectrum(..., key='{spectrum_key}') first."
        )

    spectrum_df = adata.obsm[spectrum_obsm_key]

    # Get callable sites
    callable_col = f"{callable_key}_sites"
    if callable_col not in adata.obs.columns:
        raise ValueError(
            f"'{callable_col}' not found in adata.obs. "
            f"Run spc.tl.compute_callable_sites(..., key='{callable_key}') first."
        )

    callable_sites = adata.obs[callable_col].values

    # Compute rates
    rates_df = spectrum_df.copy()
    for i, n_callable in enumerate(callable_sites):
        if n_callable > 0:
            rates_df.iloc[i, :] = spectrum_df.iloc[i, :] / n_callable
        else:
            rates_df.iloc[i, :] = 0

    # Store in adata
    adata.obsm[rate_key] = rates_df

    # Store metadata
    adata.uns[f"{rate_key}_metadata"] = {"spectrum_key": spectrum_key, "callable_key": callable_key, "is_rate": True}

    return rates_df
