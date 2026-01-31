"""Compute mutation spectra from AnnData objects."""

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from tqdm import tqdm

from cellspec.utils.constants import get_canonical_96_order


def compute_spectrum(
    adata: ad.AnnData,
    count_strategy: str = "presence",
    genotypes: list[int] | None = None,
    min_alt_depth: int | None = None,
    min_depth: int | None = None,
    max_depth: int | None = None,
    variant_mask: pd.DataFrame | np.ndarray | None = None,
    private_key: str | None = None,
    key: str = "spectrum",
    show_progress: bool = True,
    description: str | None = None,
) -> pd.DataFrame:
    """
    Compute 96-channel mutation spectrum for each cell/sample or per private mutation group.

    By default, computes spectrum per cell (stored in adata.obsm[f'spectrum_{key}']).
    If private_key is provided, computes spectrum per private mutation group
    (stored in adata.uns[f'spectrum_{key}']).

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with annotated variants
    count_strategy : str, default 'presence'
        Strategy for counting mutations:
        - 'presence': Count any variant present in .X (binary)
        - 'genotype': Count based on genotype (requires genotype layer)
        - 'alt_depth': Count sites with AD >= threshold
    genotypes : list of int, optional
        Genotypes to count when count_strategy='genotype'
        Not applicable for binary presence data (uses .X which is already filtered)
    min_alt_depth : int, optional
        Minimum AD when count_strategy='alt_depth'
    min_depth : int, optional
        Minimum DP for counting per cell (filters during counting)
    max_depth : int, optional
        Maximum DP for counting per cell (filters during counting)
    variant_mask : pd.DataFrame or np.ndarray, optional
        Boolean mask specifying which variants to count for each cell.

        - If DataFrame: rows are variants (matching adata.var.index),
          columns are cells (matching adata.obs_names)
        - If np.ndarray: shape (n_cells, n_variants) or (n_variants,)
        - If (n_variants,): same mask applied to all cells
        - True = count this variant for this cell, False = exclude

        Note: Cannot be used together with private_key.
    private_key : str, optional
        Key for private mutations DataFrame in adata.uns (e.g., 'private').
        If provided, computes spectrum per private mutation group instead of per cell.
        Looks up adata.uns[f'{private_key}_mutations'] and adata.uns[f'{private_key}_metadata'].
        For each group, counts trinucleotide contexts for variants private to that group
        within cells belonging to that group.
        Returns (96 contexts × groups) DataFrame stored in adata.uns[f'spectrum_{key}'].

        Note: Cannot be used together with variant_mask.
    key : str, default 'spectrum'
        Key for storing spectrum (in adata.obsm if per-cell, adata.uns if per-group)
    show_progress : bool, default True
        Show progress bar
    description : str, optional
        Description for metadata

    Returns
    -------
    pd.DataFrame
        Spectrum DataFrame. Shape depends on mode:
        - Per-cell mode: (cells × 96 contexts), stored in adata.obsm[f'spectrum_{key}']
        - Per-group mode: (96 contexts × groups), stored in adata.uns[f'spectrum_{key}']

    Examples
    --------
    >>> import cellspec as spc
    >>> adata = spc.pp.load_vcf("variants.vcf.gz")
    >>> spc.pp.annotate_contexts(adata, "reference.fa")
    >>> spc.pp.filter_to_snps(adata)
    >>>
    >>> # Count all variants present (default) - per cell
    >>> spectrum = spc.tl.compute_spectrum(adata, key="all")
    >>>
    >>> # Count only sites with DP >= 10 - per cell
    >>> spectrum = spc.tl.compute_spectrum(adata, min_depth=10, max_depth=200, key="dp10")
    >>>
    >>> # Compute spectrum per private mutation group
    >>> spc.tl.private_mutations(adata, groupby="lineage", genotypes=[3], store_key="private")
    >>> spectrum = spc.tl.compute_spectrum(
    ...     adata, count_strategy="genotype", genotypes=[3], private_key="private", key="private"
    ... )
    >>> # Returns (96 contexts × lineages) DataFrame in adata.uns['spectrum_private']

    Notes
    -----
    **Per-cell mode (default):**
    The function filters counting per-cell based on depth thresholds.
    A variant is counted for cell_i if:
    1. It passes the count_strategy filter
    2. AND cell_i has adequate depth at that site (if min/max_depth specified)
    3. AND the variant_mask is True for this cell (if variant_mask provided)

    Results stored in:
    - adata.obsm[f'spectrum_{key}']: DataFrame (cells × 96 contexts)
    - adata.uns[f'spectrum_{key}_metadata']: Dict with computation details

    **Per-group mode (private_key provided):**
    For each private mutation group:
    1. Identifies variants marked as private to that group
    2. Identifies cells belonging to that group
    3. Counts trinucleotide contexts for private variants within those cells only
    4. Applies count_strategy and depth filters per-cell as usual

    Results stored in:
    - adata.uns[f'spectrum_{key}']: DataFrame (96 contexts × groups)
    - adata.uns[f'spectrum_{key}_metadata']: Dict with computation details
    """
    # Validate inputs
    if "trinuc_type" not in adata.var.columns:
        raise ValueError("'trinuc_type' not found in adata.var. Run spc.pp.annotate_contexts() first.")

    valid_strategies = ["presence", "genotype", "alt_depth"]
    if count_strategy not in valid_strategies:
        raise ValueError(f"Invalid count_strategy '{count_strategy}'. Must be one of: {', '.join(valid_strategies)}")

    if count_strategy == "alt_depth" and min_alt_depth is None:
        raise ValueError("count_strategy='alt_depth' requires min_alt_depth parameter")

    if count_strategy == "genotype" and genotypes is None:
        raise ValueError("count_strategy='genotype' requires genotypes parameter")

    if variant_mask is not None and private_key is not None:
        raise ValueError("variant_mask and private_key cannot both be provided")

    # Branch based on mode
    if private_key is not None:
        return _compute_spectrum_per_group(
            adata=adata,
            private_key=private_key,
            count_strategy=count_strategy,
            genotypes=genotypes,
            min_alt_depth=min_alt_depth,
            min_depth=min_depth,
            max_depth=max_depth,
            key=key,
            show_progress=show_progress,
            description=description,
        )

    # Process variant_mask
    variant_mask_array = None
    if variant_mask is not None:
        if isinstance(variant_mask, pd.DataFrame):
            # Convert DataFrame to array (variants × cells), then transpose to (cells × variants)
            variant_mask_array = variant_mask.T.values
        elif isinstance(variant_mask, np.ndarray):
            if variant_mask.ndim == 1:
                # 1D array: apply same mask to all cells
                variant_mask_array = np.tile(variant_mask, (adata.n_obs, 1))
            elif variant_mask.shape == (adata.n_obs, adata.n_vars):
                variant_mask_array = variant_mask
            elif variant_mask.shape == (adata.n_vars, adata.n_obs):
                # Transpose if needed
                variant_mask_array = variant_mask.T
            else:
                raise ValueError(
                    f"variant_mask array shape {variant_mask.shape} doesn't match "
                    f"expected (n_cells={adata.n_obs}, n_vars={adata.n_vars})"
                )
        else:
            raise TypeError("variant_mask must be pd.DataFrame or np.ndarray")

    # Get canonical 96 order
    canonical_contexts = get_canonical_96_order()

    # Initialize spectrum matrix (cells × contexts)
    n_cells = adata.n_obs
    n_contexts = len(canonical_contexts)
    spectrum_matrix = np.zeros((n_cells, n_contexts), dtype=int)

    # Get trinuc types for each variant
    trinuc_types = adata.var["trinuc_type"].values

    # Create mapping from trinuc context to index
    context_to_idx = {context: idx for idx, context in enumerate(canonical_contexts)}

    # Get data matrices
    X = adata.X
    if issparse(X):
        X = X.toarray()

    # Get layers if needed
    if min_depth is not None or max_depth is not None:
        if "DP" not in adata.layers:
            raise ValueError("min_depth/max_depth requires 'DP' layer in adata.layers")
        DP = adata.layers["DP"]
        if issparse(DP):
            DP = DP.toarray()
    else:
        DP = None

    if count_strategy == "alt_depth":
        if "AD" not in adata.layers:
            raise ValueError("count_strategy='alt_depth' requires 'AD' layer")
        AD = adata.layers["AD"]
        if issparse(AD):
            AD = AD.toarray()
    else:
        AD = None

    # Compute spectrum for each cell
    iterator = (
        tqdm(range(n_cells), desc=f"Computing spectrum (key='{key}')", unit="cell") if show_progress else range(n_cells)
    )

    for cell_idx in iterator:
        # Start with presence from .X
        if count_strategy == "presence":
            mask = X[cell_idx, :] > 0  # Binary presence

        elif count_strategy == "genotype":
            # Count specific genotypes
            mask = np.isin(X[cell_idx, :], genotypes)

        elif count_strategy == "alt_depth":
            # Count sites with AD >= threshold
            mask = AD[cell_idx, :] >= min_alt_depth

        # Apply per-cell depth filtering
        if min_depth is not None:
            mask &= DP[cell_idx, :] >= min_depth
        if max_depth is not None:
            mask &= DP[cell_idx, :] <= max_depth

        # Apply variant mask if provided
        if variant_mask_array is not None:
            mask &= variant_mask_array[cell_idx, :]

        # Get trinuc types for variants passing filters
        passing_trinuc_types = trinuc_types[mask]

        # Count mutations by trinuc type
        for trinuc in passing_trinuc_types:
            if pd.notna(trinuc) and trinuc in context_to_idx:
                spectrum_matrix[cell_idx, context_to_idx[trinuc]] += 1

    # Create DataFrame
    spectrum_df = pd.DataFrame(spectrum_matrix, index=adata.obs_names, columns=canonical_contexts)

    # Store in adata
    adata.obsm[f"spectrum_{key}"] = spectrum_df

    # Store metadata
    metadata = {
        "count_strategy": count_strategy,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "description": description,
        "n_cells": n_cells,
        "n_variants": adata.n_vars,
        "variant_mask_applied": variant_mask is not None,
    }
    if count_strategy == "genotype":
        metadata["genotypes"] = genotypes
    if count_strategy == "alt_depth":
        metadata["min_alt_depth"] = min_alt_depth

    adata.uns[f"spectrum_{key}_metadata"] = metadata

    return spectrum_df


def _compute_spectrum_per_group(
    adata: ad.AnnData,
    private_key: str,
    count_strategy: str,
    genotypes: list[int] | None,
    min_alt_depth: int | None,
    min_depth: int | None,
    max_depth: int | None,
    key: str,
    show_progress: bool,
    description: str | None,
) -> pd.DataFrame:
    """
    Internal function to compute spectrum per private mutation group.

    For each group, counts trinucleotide contexts for variants private to that group
    within cells belonging to that group only.

    Returns (96 contexts × groups) DataFrame.
    """
    # Load private mutations data from adata.uns
    private_mutations_key = f"{private_key}_mutations"
    private_metadata_key = f"{private_key}_metadata"

    if private_mutations_key not in adata.uns:
        raise ValueError(
            f"'{private_mutations_key}' not found in adata.uns. "
            f"Run spc.tl.private_mutations() with store_key='{private_key}' first."
        )

    if private_metadata_key not in adata.uns:
        raise ValueError(
            f"'{private_metadata_key}' not found in adata.uns. "
            f"Run spc.tl.private_mutations() with store_key='{private_key}' first."
        )

    private_df = adata.uns[private_mutations_key]
    private_metadata = adata.uns[private_metadata_key]
    groupby = private_metadata.get("groupby", None)

    # Get canonical 96 order
    canonical_contexts = get_canonical_96_order()
    n_contexts = len(canonical_contexts)
    context_to_idx = {context: idx for idx, context in enumerate(canonical_contexts)}

    # Get trinuc types for each variant
    trinuc_types = adata.var["trinuc_type"].values

    # Get data matrices
    X = adata.X
    if issparse(X):
        X = X.toarray()

    # Get layers if needed
    if min_depth is not None or max_depth is not None:
        if "DP" not in adata.layers:
            raise ValueError("min_depth/max_depth requires 'DP' layer in adata.layers")
        DP = adata.layers["DP"]
        if issparse(DP):
            DP = DP.toarray()
    else:
        DP = None

    if count_strategy == "alt_depth":
        if "AD" not in adata.layers:
            raise ValueError("count_strategy='alt_depth' requires 'AD' layer")
        AD = adata.layers["AD"]
        if issparse(AD):
            AD = AD.toarray()
    else:
        AD = None

    # Initialize spectrum matrix (contexts × groups)
    groups = list(private_df.columns)
    n_groups = len(groups)
    spectrum_matrix = np.zeros((n_contexts, n_groups), dtype=int)

    # Iterate over groups
    iterator = (
        tqdm(enumerate(groups), total=n_groups, desc=f"Computing spectrum (key='{key}')", unit="group")
        if show_progress
        else enumerate(groups)
    )

    for group_idx, group in iterator:
        # Get boolean mask of variants private to this group
        private_variant_mask = private_df[group].values

        # Get indices of cells in this group
        if groupby is None:
            # Each "group" is a single cell
            if group not in adata.obs_names:
                raise ValueError(
                    f"Group '{group}' not found in adata.obs_names. "
                    f"Private mutations were computed per-cell but cell '{group}' is missing."
                )
            cell_indices = [adata.obs_names.get_loc(group)]
        else:
            # Group is defined by groupby column
            if groupby not in adata.obs.columns:
                raise ValueError(
                    f"groupby column '{groupby}' from private mutations metadata "
                    f"not found in adata.obs. Available columns: {list(adata.obs.columns)}"
                )
            cell_mask = adata.obs[groupby] == group
            cell_indices = np.where(cell_mask)[0]

        # For each cell in this group, count private variants
        for cell_idx in cell_indices:
            # Apply count_strategy
            if count_strategy == "presence":
                mask = X[cell_idx, :] > 0
            elif count_strategy == "genotype":
                mask = np.isin(X[cell_idx, :], genotypes)
            elif count_strategy == "alt_depth":
                mask = AD[cell_idx, :] >= min_alt_depth

            # Apply per-cell depth filtering
            if min_depth is not None:
                mask &= DP[cell_idx, :] >= min_depth
            if max_depth is not None:
                mask &= DP[cell_idx, :] <= max_depth

            # Apply private variant mask - only count variants private to this group
            mask &= private_variant_mask

            # Get trinuc types for variants passing filters
            passing_trinuc_types = trinuc_types[mask]

            # Count mutations by trinuc type
            for trinuc in passing_trinuc_types:
                if pd.notna(trinuc) and trinuc in context_to_idx:
                    spectrum_matrix[context_to_idx[trinuc], group_idx] += 1

    # Create DataFrame (96 contexts × groups)
    spectrum_df = pd.DataFrame(spectrum_matrix, index=canonical_contexts, columns=groups)

    # Store in adata.uns (not adata.obsm since it's not per-cell)
    adata.uns[f"spectrum_{key}"] = spectrum_df

    # Store metadata
    metadata = {
        "mode": "per_group",
        "private_key": private_key,
        "groupby": groupby,
        "count_strategy": count_strategy,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "description": description,
        "n_groups": n_groups,
        "groups": groups,
        "n_variants": adata.n_vars,
    }
    if count_strategy == "genotype":
        metadata["genotypes"] = genotypes
    if count_strategy == "alt_depth":
        metadata["min_alt_depth"] = min_alt_depth

    adata.uns[f"spectrum_{key}_metadata"] = metadata

    return spectrum_df
