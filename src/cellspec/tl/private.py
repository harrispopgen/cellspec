"""Identify private mutations (unique to individual cells or groups)."""

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse


def private_mutations(
    adata: ad.AnnData,
    groupby: str | None = None,
    count_strategy: str = "presence",
    genotypes: list[int] | None = None,
    min_alt_depth: int | None = None,
    min_depth: int | None = None,
    max_depth: int | None = None,
    store_key: str = "private",
    inplace: bool = True,
) -> pd.DataFrame:
    """
    Identify mutations that are private (unique) to each cell/sample or group.

    **Per-cell mode (groupby=None):**
    A mutation is considered private to a cell if it is present in that cell
    and absent from all other cells.

    **Per-group mode (groupby specified):**
    A mutation is considered private to a group if it is:
    1. Present in ALL cells/samples within that group (shared)
    2. AND absent from ALL cells/samples in all other groups (unique to group)

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with mutation data
    groupby : str, optional
        Column name in adata.obs to group by (e.g., 'cell_type', 'sample_id').
        If None, treats each cell/sample individually.
    count_strategy : str, default 'presence'
        Strategy for counting mutations:
        - 'presence': Count any variant present in .X (any non-zero value)
        - 'genotype': Count only specific genotypes (requires genotypes parameter)
        - 'alt_depth': Count sites with AD >= threshold (requires min_alt_depth)
    genotypes : list of int, optional
        Genotypes to count when count_strategy='genotype'.
        E.g., [1, 3] for HET and HOM_ALT, or [3] for HOM_ALT only.
    min_alt_depth : int, optional
        Minimum AD when count_strategy='alt_depth'
    min_depth : int, optional
        Minimum DP for counting per cell (filters per-cell)
    max_depth : int, optional
        Maximum DP for counting per cell (filters per-cell)
    store_key : str, default 'private'
        Key for storing results in adata.uns
    inplace : bool, default True
        If True, stores results in adata.uns[f'{store_key}_mutations'].
        If False, only returns the DataFrame.

    Returns
    -------
    pd.DataFrame
        Boolean DataFrame with variants as rows and cells/groups as columns.
        True indicates the mutation is private to that cell/group.
        Also stores counts in adata.obs[f'{store_key}_count'] (per cell/group).

    Examples
    --------
    >>> import cellspec as spc
    >>> # Find private mutations for each sample (any non-zero genotype)
    >>> private_df = spc.tl.private_mutations(adata)
    >>> # Count per sample stored in adata.obs['private_count']
    >>> print(adata.obs["private_count"])

    >>> # Find private mutations per cell type
    >>> private_df = spc.tl.private_mutations(adata, groupby="cell_type")

    >>> # Only count HOM_ALT (genotype=3) as private
    >>> private_df = spc.tl.private_mutations(adata, count_strategy="genotype", genotypes=[3])

    >>> # Use alternate depth with depth filtering
    >>> private_df = spc.tl.private_mutations(
    ...     adata, count_strategy="alt_depth", min_alt_depth=3, min_depth=10, max_depth=200
    ... )

    Notes
    -----
    **Per-cell mode (groupby=None):**
    For each cell, a mutation is private if:
    1. It is present (based on count_strategy) in that cell
    2. It passes depth filters (if specified)
    3. It is absent from ALL other cells

    **Per-group mode (groupby specified):**
    For each group, a mutation is private if:
    1. It is present (based on count_strategy) in ALL cells within that group
    2. It passes depth filters (if specified) for all cells in the group
    3. It is absent from ALL cells in all other groups

    This makes private mutations "group-specific shared mutations" - mutations
    that are shared among all members of a group but found in no other groups.

    Summing along axis=1 (across cells/groups) should give values of 0 or 1,
    since a mutation can only be private to one cell/group.

    Results are stored in:

    - adata.uns[f'{store_key}_mutations']: Boolean DataFrame (variants × cells/groups)
    - adata.obs[f'{store_key}_count'] or adata.uns[f'{store_key}_counts']:
      Count of private mutations per cell/group
    - adata.uns[f'{store_key}_metadata']: Parameters used
    """
    # Validate inputs
    valid_strategies = ["presence", "genotype", "alt_depth"]
    if count_strategy not in valid_strategies:
        raise ValueError(f"Invalid count_strategy '{count_strategy}'. Must be one of: {', '.join(valid_strategies)}")

    if count_strategy == "genotype" and genotypes is None:
        raise ValueError("count_strategy='genotype' requires genotypes parameter")

    if count_strategy == "alt_depth" and min_alt_depth is None:
        raise ValueError("count_strategy='alt_depth' requires min_alt_depth parameter")

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

    if groupby is None:
        # Work with individual cells/samples
        cell_names = list(adata.obs_names)
        n_groups = len(cell_names)

        # Initialize results
        private_dict = {}

        for i, name in enumerate(cell_names):
            # Determine if mutation is "present" based on count_strategy
            if count_strategy == "presence":
                mask_current = X[i, :] > 0
            elif count_strategy == "genotype":
                mask_current = np.isin(X[i, :], genotypes)
            elif count_strategy == "alt_depth":
                mask_current = AD[i, :] >= min_alt_depth

            # Apply per-cell depth filtering
            if min_depth is not None:
                mask_current = mask_current & (DP[i, :] >= min_depth)
            if max_depth is not None:
                mask_current = mask_current & (DP[i, :] <= max_depth)

            in_current = np.ravel(mask_current)

            # Check if mutation is absent from all other cells
            other_indices = [j for j in range(n_groups) if j != i]

            # For each other cell, check if mutation is present (using same strategy)
            present_in_others = np.zeros(adata.n_vars, dtype=bool)
            for j in other_indices:
                if count_strategy == "presence":
                    mask_other = X[j, :] > 0
                elif count_strategy == "genotype":
                    mask_other = np.isin(X[j, :], genotypes)
                elif count_strategy == "alt_depth":
                    mask_other = AD[j, :] >= min_alt_depth

                # Apply per-cell depth filtering
                if min_depth is not None:
                    mask_other = mask_other & (DP[j, :] >= min_depth)
                if max_depth is not None:
                    mask_other = mask_other & (DP[j, :] <= max_depth)

                present_in_others = present_in_others | np.ravel(mask_other)

            not_in_others = ~present_in_others

            # Private = present in current AND absent in all others
            private_dict[name] = in_current & not_in_others

        # Create DataFrame
        private_df = pd.DataFrame.from_dict(private_dict, orient="columns")
        private_df.index = adata.var.index

        # Store counts in .obs
        if inplace:
            counts = private_df.sum(axis=0)
            adata.obs[f"{store_key}_count"] = counts
            adata.uns[f"{store_key}_mutations"] = private_df

            metadata = {
                "groupby": None,
                "count_strategy": count_strategy,
                "min_depth": min_depth,
                "max_depth": max_depth,
                "n_cells": n_groups,
            }
            if count_strategy == "genotype":
                metadata["genotypes"] = genotypes
            if count_strategy == "alt_depth":
                metadata["min_alt_depth"] = min_alt_depth

            adata.uns[f"{store_key}_metadata"] = metadata

    else:
        # Group by column in .obs
        if groupby not in adata.obs.columns:
            raise ValueError(
                f"groupby column '{groupby}' not found in adata.obs. Available columns: {list(adata.obs.columns)}"
            )

        # Get unique groups
        groups = adata.obs[groupby].unique()
        n_groups = len(groups)

        # Initialize results
        private_dict = {}

        for group in groups:
            # Get cells in current group
            current_mask = adata.obs[groupby] == group
            current_indices = np.where(current_mask)[0]

            # Get cells in other groups
            other_indices = np.where(~current_mask)[0]

            # Check if mutation is present in ALL cells in current group
            # Initialize to True, then use AND to require presence in every cell
            shared_in_current_group = np.ones(adata.n_vars, dtype=bool)
            for i in current_indices:
                if count_strategy == "presence":
                    mask = X[i, :] > 0
                elif count_strategy == "genotype":
                    mask = np.isin(X[i, :], genotypes)
                elif count_strategy == "alt_depth":
                    mask = AD[i, :] >= min_alt_depth

                # Apply per-cell depth filtering
                if min_depth is not None:
                    mask = mask & (DP[i, :] >= min_depth)
                if max_depth is not None:
                    mask = mask & (DP[i, :] <= max_depth)

                # Use AND: only keep True if present in this cell AND all previous cells
                shared_in_current_group = shared_in_current_group & np.ravel(mask)

            # For each cell in other groups, check if mutation is present
            present_in_other_groups = np.zeros(adata.n_vars, dtype=bool)
            for i in other_indices:
                if count_strategy == "presence":
                    mask = X[i, :] > 0
                elif count_strategy == "genotype":
                    mask = np.isin(X[i, :], genotypes)
                elif count_strategy == "alt_depth":
                    mask = AD[i, :] >= min_alt_depth

                # Apply per-cell depth filtering
                if min_depth is not None:
                    mask = mask & (DP[i, :] >= min_depth)
                if max_depth is not None:
                    mask = mask & (DP[i, :] <= max_depth)

                present_in_other_groups = present_in_other_groups | np.ravel(mask)

            not_in_others = ~present_in_other_groups

            # Private = shared in ALL cells of current group AND absent in all other groups
            private_dict[str(group)] = shared_in_current_group & not_in_others

        # Create DataFrame
        private_df = pd.DataFrame.from_dict(private_dict, orient="columns")
        private_df.index = adata.var.index

        # Store counts
        if inplace:
            counts = private_df.sum(axis=0)
            adata.uns[f"{store_key}_counts"] = counts.to_dict()  # Convert Series to dict for h5ad compatibility
            adata.uns[f"{store_key}_mutations"] = private_df

            metadata = {
                "groupby": groupby,
                "count_strategy": count_strategy,
                "min_depth": min_depth,
                "max_depth": max_depth,
                "n_groups": n_groups,
                "groups": list(groups),
            }
            if count_strategy == "genotype":
                metadata["genotypes"] = genotypes
            if count_strategy == "alt_depth":
                metadata["min_alt_depth"] = min_alt_depth

            adata.uns[f"{store_key}_metadata"] = metadata

    # Validation check
    max_private_per_mutation = private_df.sum(axis=1).max()
    if max_private_per_mutation > 1:
        import warnings

        warnings.warn(
            f"Found mutations marked as private to multiple cells/groups (max={max_private_per_mutation}). "
            "This suggests a bug - please check your data or contact the developers.",
            stacklevel=2,
        )

    return private_df
