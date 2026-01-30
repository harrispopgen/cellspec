"""Normalize mutation spectra by various strategies."""

import numpy as np
import pandas as pd
import anndata as ad
from typing import Optional, Union


def normalize_spectrum(
    adata: ad.AnnData,
    key: str,
    method: str = 'proportion',
    obs_column: Optional[str] = None,
    normalization_vector: Optional[Union[np.ndarray, pd.Series]] = None,
    output_key: Optional[str] = None,
    inplace: bool = False
) -> pd.DataFrame:
    """
    Normalize a mutation spectrum by various strategies.

    This function normalizes spectra to enable fair comparison across samples with
    different sequencing depths, mutation burdens, or other covariates.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with computed spectrum
    key : str
        Key for spectrum in adata.obsm (e.g., 'somatic' for 'spectrum_somatic')
    method : str, default 'proportion'
        Normalization method:
        - 'proportion': Divide by row sum (each sample sums to 1)
        - 'obs_column': Divide by a column in adata.obs
        - 'vector': Divide by a custom normalization vector
    obs_column : str, optional
        Column name in adata.obs to use for normalization when method='obs_column'.
        Examples: 'callable_sites', 'total_reads', 'coverage'
    normalization_vector : np.ndarray or pd.Series, optional
        Custom vector to divide by when method='vector'.
        Must have length equal to number of cells/samples.
    output_key : str, optional
        Key for storing normalized spectrum. If None, uses f'{key}_normalized'
        The normalized spectrum is stored in adata.obsm[f'spectrum_{output_key}']
    inplace : bool, default False
        If True, replaces the original spectrum with normalized values.
        If False, stores normalized spectrum under output_key.

    Returns
    -------
    pd.DataFrame
        Normalized spectrum DataFrame (cells × 96 contexts)

    Examples
    --------
    >>> import cellspec as spc
    >>>
    >>> # Normalize to proportions (each sample sums to 1)
    >>> spc.tl.normalize_spectrum(
    ...     adata,
    ...     key='somatic',
    ...     method='proportion',
    ...     output_key='somatic_prop'
    ... )
    >>>
    >>> # Normalize by callable sites (mutations per callable site)
    >>> spc.tl.compute_callable_sites(adata, min_depth=10, max_depth=200)
    >>> spc.tl.normalize_spectrum(
    ...     adata,
    ...     key='somatic',
    ...     method='obs_column',
    ...     obs_column='callable_sites',
    ...     output_key='somatic_rate'
    ... )
    >>>
    >>> # Normalize by total read count
    >>> spc.tl.normalize_spectrum(
    ...     adata,
    ...     key='somatic',
    ...     method='obs_column',
    ...     obs_column='total_reads',
    ...     output_key='somatic_per_read'
    ... )
    >>>
    >>> # Normalize by custom vector
    >>> coverage_vector = adata.obs['mean_coverage'].values
    >>> spc.tl.normalize_spectrum(
    ...     adata,
    ...     key='somatic',
    ...     method='vector',
    ...     normalization_vector=coverage_vector,
    ...     output_key='somatic_per_coverage'
    ... )
    >>>
    >>> # Normalize in place (replaces original)
    >>> spc.tl.normalize_spectrum(
    ...     adata,
    ...     key='somatic',
    ...     method='proportion',
    ...     inplace=True
    ... )

    Notes
    -----
    - Division by zero is handled by setting those rows to NaN
    - Normalization metadata is stored in adata.uns
    - The original spectrum is preserved unless inplace=True

    See Also
    --------
    compute_rates : Specialized function for rate normalization by callable sites
    """
    # Validate inputs
    spectrum_key = f'spectrum_{key}'
    if spectrum_key not in adata.obsm:
        raise ValueError(
            f"'{spectrum_key}' not found in adata.obsm. "
            f"Run spc.tl.compute_spectrum(..., key='{key}') first."
        )

    valid_methods = ['proportion', 'obs_column', 'vector']
    if method not in valid_methods:
        raise ValueError(
            f"Invalid method '{method}'. "
            f"Must be one of: {', '.join(valid_methods)}"
        )

    if method == 'obs_column':
        if obs_column is None:
            raise ValueError("method='obs_column' requires obs_column parameter")
        if obs_column not in adata.obs.columns:
            raise ValueError(
                f"Column '{obs_column}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

    if method == 'vector':
        if normalization_vector is None:
            raise ValueError("method='vector' requires normalization_vector parameter")
        if len(normalization_vector) != adata.n_obs:
            raise ValueError(
                f"normalization_vector length ({len(normalization_vector)}) "
                f"does not match number of samples ({adata.n_obs})"
            )

    # Get spectrum
    spectrum_df = adata.obsm[spectrum_key].copy()

    # Perform normalization
    if method == 'proportion':
        # Normalize to proportions (each row sums to 1)
        row_sums = spectrum_df.sum(axis=1)
        # Avoid division by zero
        normalized_df = spectrum_df.div(row_sums, axis=0)
        # Set rows with zero sum to NaN or 0
        normalized_df = normalized_df.fillna(0)
        normalization_info = {'method': 'proportion'}

    elif method == 'obs_column':
        # Normalize by obs column
        normalization_values = adata.obs[obs_column].values
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_df = spectrum_df.div(normalization_values, axis=0)
        normalized_df = normalized_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        normalization_info = {
            'method': 'obs_column',
            'obs_column': obs_column
        }

    elif method == 'vector':
        # Normalize by custom vector
        if isinstance(normalization_vector, pd.Series):
            normalization_vector = normalization_vector.values
        # Avoid division by zero
        with np.errstate(divide='ignore', invalid='ignore'):
            normalized_df = spectrum_df.div(normalization_vector, axis=0)
        normalized_df = normalized_df.replace([np.inf, -np.inf], np.nan).fillna(0)
        normalization_info = {
            'method': 'vector',
            'vector_description': 'custom normalization vector'
        }

    # Determine output key
    if inplace:
        final_key = key
    else:
        if output_key is None:
            final_key = f'{key}_normalized'
        else:
            final_key = output_key

    # Store normalized spectrum
    adata.obsm[f'spectrum_{final_key}'] = normalized_df

    # Store normalization metadata
    normalization_info.update({
        'source_key': key,
        'normalized': True
    })
    adata.uns[f'spectrum_{final_key}_normalization'] = normalization_info

    # Copy original metadata if available
    if f'spectrum_{key}_metadata' in adata.uns:
        original_metadata = adata.uns[f'spectrum_{key}_metadata'].copy()
        original_metadata['normalization'] = normalization_info
        adata.uns[f'spectrum_{final_key}_metadata'] = original_metadata

    return normalized_df
