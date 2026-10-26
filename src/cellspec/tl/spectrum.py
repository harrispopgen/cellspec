"""Compute mutation spectra from AnnData objects."""

import os

import anndata as ad
import numpy as np
import pandas as pd
from scipy.sparse import issparse
from tqdm import tqdm

from cellspec.utils.constants import get_canonical_96_order

AGGREGATE_MODES = ("sum", "private_spectrum", "randomize")


def _default_key(aggregate: str | None, groupby: str | None, private_key: str | None) -> str:
    """Build a default `key` from the aggregation parameters."""
    if aggregate is None:
        return "per_cell"
    if aggregate == "private_spectrum":
        return f"private_spectrum_from_{private_key}"
    if groupby is None:
        return aggregate
    return f"{aggregate}_by_{groupby}"


def _build_carrier_matrix(
    adata: ad.AnnData,
    count_strategy: str,
    genotypes: list[int] | None,
    min_alt_depth: int | None,
    min_depth: int | None,
    max_depth: int | None,
    variant_mask: np.ndarray | pd.DataFrame | None,
) -> np.ndarray:
    """
    Build a (n_cells × n_vars) boolean matrix marking cells that carry each variant.

    Applies ``count_strategy`` (presence / genotype / alt_depth) and the per-cell
    depth filters. ``variant_mask`` can be a 1D ``(n_vars,)`` boolean (same mask
    for every cell), a 2D ``(n_cells, n_vars)`` boolean, or a DataFrame
    ``(n_vars × n_cells)`` matching ``adata.var.index`` and ``adata.obs_names``.
    """
    X = adata.X
    if issparse(X):
        X = X.toarray()

    if count_strategy == "presence":
        carrier = X > 0
    elif count_strategy == "genotype":
        if genotypes is None:
            raise ValueError("count_strategy='genotype' requires genotypes parameter")
        carrier = np.isin(X, genotypes)
    elif count_strategy == "alt_depth":
        if min_alt_depth is None:
            raise ValueError("count_strategy='alt_depth' requires min_alt_depth parameter")
        if "AD" not in adata.layers:
            raise ValueError("count_strategy='alt_depth' requires 'AD' layer in adata.layers")
        AD = adata.layers["AD"]
        if issparse(AD):
            AD = AD.toarray()
        carrier = AD >= min_alt_depth
    else:
        raise ValueError(f"Invalid count_strategy '{count_strategy}'. Must be one of: presence, genotype, alt_depth")

    if min_depth is not None or max_depth is not None:
        if "DP" not in adata.layers:
            raise ValueError("min_depth/max_depth requires 'DP' layer in adata.layers")
        DP = adata.layers["DP"]
        if issparse(DP):
            DP = DP.toarray()
        if min_depth is not None:
            carrier &= DP >= min_depth
        if max_depth is not None:
            carrier &= DP <= max_depth

    if variant_mask is not None:
        if isinstance(variant_mask, pd.DataFrame):
            # (variants × cells) → (cells × variants)
            mask_arr = variant_mask.T.values
        elif isinstance(variant_mask, np.ndarray):
            if variant_mask.ndim == 1:
                mask_arr = np.broadcast_to(variant_mask, (adata.n_obs, adata.n_vars))
            elif variant_mask.shape == (adata.n_obs, adata.n_vars):
                mask_arr = variant_mask
            elif variant_mask.shape == (adata.n_vars, adata.n_obs):
                mask_arr = variant_mask.T
            else:
                raise ValueError(
                    f"variant_mask array shape {variant_mask.shape} doesn't match "
                    f"expected (n_cells={adata.n_obs}, n_vars={adata.n_vars})"
                )
        else:
            raise TypeError("variant_mask must be pd.DataFrame or np.ndarray")
        carrier = carrier & mask_arr

    return carrier


def _context_idx_lookup() -> tuple[list[str], dict[str, int]]:
    """Return canonical 96 context list and context→index mapping."""
    canonical = get_canonical_96_order()
    return canonical, {c: i for i, c in enumerate(canonical)}


def _per_cell_counts(carrier: np.ndarray, trinuc_types: np.ndarray) -> np.ndarray:
    """Collapse (n_cells × n_vars) carrier bool to (n_cells × 96) context counts."""
    canonical, context_to_idx = _context_idx_lookup()
    n_cells = carrier.shape[0]
    n_contexts = len(canonical)
    out = np.zeros((n_cells, n_contexts), dtype=np.int64)
    # Precompute each variant's context slot (-1 for missing / non-SNP).
    ctx_slot = np.full(len(trinuc_types), -1, dtype=np.int64)
    for j, ctx in enumerate(trinuc_types):
        if isinstance(ctx, str) and ctx in context_to_idx:
            ctx_slot[j] = context_to_idx[ctx]
    valid_vars = np.where(ctx_slot >= 0)[0]
    valid_slots = ctx_slot[valid_vars]
    sub_carrier = carrier[:, valid_vars]
    # np.add.at for scatter-add indexed by context slot.
    for col_idx, slot in enumerate(valid_slots):
        np.add.at(out, (np.arange(n_cells)[sub_carrier[:, col_idx]], slot), 1)
    return out


def _groupby_counts(
    carrier: np.ndarray,
    trinuc_types: np.ndarray,
    group_ids: np.ndarray,
) -> tuple[np.ndarray, list[str]]:
    """
    Sum carrier matrix to (96 × n_groups) via group indicator reduction.

    group_ids maps each cell → group label; returns the group labels in order.
    """
    canonical, context_to_idx = _context_idx_lookup()
    unique_groups, group_idx = np.unique(group_ids, return_inverse=True)
    n_groups = len(unique_groups)
    out = np.zeros((len(canonical), n_groups), dtype=np.int64)
    ctx_slot = np.full(len(trinuc_types), -1, dtype=np.int64)
    for j, ctx in enumerate(trinuc_types):
        if isinstance(ctx, str) and ctx in context_to_idx:
            ctx_slot[j] = context_to_idx[ctx]
    valid_vars = np.where(ctx_slot >= 0)[0]
    for j in valid_vars:
        if not carrier[:, j].any():
            continue
        counts = np.bincount(group_idx[carrier[:, j]], minlength=n_groups)
        out[ctx_slot[j], :] += counts
    return out, [str(g) for g in unique_groups]


def _randomize_counts(
    carrier: np.ndarray,
    trinuc_types: np.ndarray,
    group_ids: np.ndarray | None,
    random_state: int | np.random.Generator | None,
) -> tuple[np.ndarray, list[str] | None]:
    """
    Assign each variant to one random carrier (then to that carrier's group).

    Weights are uniform across *cells* that carry the variant — a group with
    more carrier cells is proportionally more likely to be chosen. Variants
    with no carriers contribute nothing.
    """
    canonical, context_to_idx = _context_idx_lookup()
    rng = np.random.default_rng(random_state)

    ctx_slot = np.full(len(trinuc_types), -1, dtype=np.int64)
    for j, ctx in enumerate(trinuc_types):
        if isinstance(ctx, str) and ctx in context_to_idx:
            ctx_slot[j] = context_to_idx[ctx]
    valid_vars = np.where(ctx_slot >= 0)[0]

    has_carrier = carrier[:, valid_vars].any(axis=0)
    final_vars = valid_vars[has_carrier]
    if len(final_vars) == 0:
        if group_ids is None:
            return np.zeros(len(canonical), dtype=np.int64), None
        unique = sorted(set(map(str, group_ids)))
        return np.zeros((len(canonical), len(unique)), dtype=np.int64), unique

    # Draw a uniform weight per (cell, valid variant) and zero out non-carriers.
    weights = rng.random((carrier.shape[0], len(final_vars))).astype(np.float32)
    weights *= carrier[:, final_vars]
    chosen_cell = weights.argmax(axis=0)  # (len(final_vars),)

    if group_ids is None:
        out = np.zeros(len(canonical), dtype=np.int64)
        np.add.at(out, ctx_slot[final_vars], 1)
        return out, None

    unique_groups, group_idx = np.unique(group_ids, return_inverse=True)
    chosen_group = group_idx[chosen_cell]
    out = np.zeros((len(canonical), len(unique_groups)), dtype=np.int64)
    np.add.at(out, (ctx_slot[final_vars], chosen_group), 1)
    return out, [str(g) for g in unique_groups]


def _private_spectrum_counts(
    adata: ad.AnnData,
    private_key: str,
    variant_mask: np.ndarray | pd.DataFrame | None,
) -> tuple[np.ndarray, list[str]]:
    """
    Build (96 × owners) spectrum by counting each private variant once per owner.

    An "owner" is a cell (if ``private_mutations`` was run with ``groupby=None``)
    or a group (if run with ``groupby='…'``). Each private variant contributes
    exactly one count to its owner, regardless of how many cells carry it —
    this avoids re-counting a mutation that is by definition shared across every
    cell in the owner group.
    """
    canonical, context_to_idx = _context_idx_lookup()
    private_mutations_key = f"{private_key}_mutations"
    if private_mutations_key not in adata.uns:
        raise ValueError(
            f"'{private_mutations_key}' not found in adata.uns. "
            f"Run spc.tl.private_mutations(..., store_key='{private_key}') first."
        )
    private_df = adata.uns[private_mutations_key]
    # private_df is (variants × owners) bool, variant index matches adata.var.index.
    private_arr = private_df.values.astype(bool)

    if variant_mask is not None:
        if isinstance(variant_mask, pd.DataFrame):
            raise TypeError("variant_mask must be a 1D array when aggregate='private_spectrum'")
        variant_mask = np.asarray(variant_mask)
        if variant_mask.ndim != 1 or len(variant_mask) != adata.n_vars:
            raise ValueError(
                f"variant_mask for private_spectrum must be 1D of length n_vars={adata.n_vars}, "
                f"got shape {variant_mask.shape}"
            )
        private_arr = private_arr & variant_mask[:, None]

    trinuc_types = adata.var["trinuc_type"].values
    ctx_slot = np.full(len(trinuc_types), -1, dtype=np.int64)
    for j, ctx in enumerate(trinuc_types):
        if isinstance(ctx, str) and ctx in context_to_idx:
            ctx_slot[j] = context_to_idx[ctx]

    owners = list(private_df.columns)
    out = np.zeros((len(canonical), len(owners)), dtype=np.int64)
    for owner_idx in range(len(owners)):
        private_vars = private_arr[:, owner_idx]
        valid = private_vars & (ctx_slot >= 0)
        if not valid.any():
            continue
        slots = ctx_slot[valid]
        np.add.at(out[:, owner_idx], slots, 1)
    return out, [str(o) for o in owners]


def compute_spectrum(
    adata: ad.AnnData,
    *,
    count_strategy: str = "presence",
    genotypes: list[int] | None = None,
    min_alt_depth: int | None = None,
    min_depth: int | None = None,
    max_depth: int | None = None,
    variant_mask: pd.DataFrame | np.ndarray | None = None,
    aggregate: str | None = None,
    groupby: str | None = None,
    private_key: str | None = None,
    random_state: int | np.random.Generator | None = None,
    key: str | None = None,
    csv_path: str | None = None,
    show_progress: bool = True,
    description: str | None = None,
) -> pd.DataFrame | pd.Series:
    """
    Compute 96-channel trinucleotide mutation spectra from an AnnData.

    The ``aggregate`` argument selects what gets stored and where:

    - ``aggregate=None`` (default) — per-cell matrix, stored in
      ``adata.obsm[f'spectrum_{key}']`` as ``(n_cells, 96)``.
    - ``aggregate='sum'`` — naive sum of carrier counts. With ``groupby=None``
      this is a single 96-vector (all cells pooled). With ``groupby='col'`` it
      is ``(96, n_groups)``, where each cell contributes its own carrier count
      to the group it belongs to. A variant shared by *k* cells of a group
      therefore contributes *k* to that group's count.
    - ``aggregate='private_spectrum'`` — count each private variant exactly
      once per owner (cell or group). Requires that
      :func:`cellspec.tl.private_mutations` has already been run and the
      resulting mask is at ``adata.uns[f'{private_key}_mutations']``. Shape is
      ``(96, n_owners)`` where ``n_owners`` is the number of cells or groups
      used in the ``private_mutations`` call.
    - ``aggregate='randomize'`` — Beichman-style single-carrier assignment.
      For each variant with at least one carrier, draw one carrier cell
      uniformly and credit the mutation to that cell (or, with ``groupby``, to
      that cell's group). Each variant contributes 1 total count. With
      ``groupby=None`` the result is deterministic (every covered variant is
      counted once); with ``groupby='col'`` the per-group distribution depends
      on the draw, controlled by ``random_state``.

    Non-``None`` aggregated spectra are stored in ``adata.uns[f'spectrum_{key}']``.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData with ``'trinuc_type'`` on ``adata.var`` (from
        :func:`cellspec.pp.annotate_contexts`).
    count_strategy : {'presence', 'genotype', 'alt_depth'}, default 'presence'
        How to decide whether a cell carries a variant at a site.
    genotypes : list of int, optional
        Required for ``count_strategy='genotype'``.
    min_alt_depth : int, optional
        Required for ``count_strategy='alt_depth'``.
    min_depth, max_depth : int, optional
        Per-cell depth bounds filtering counting.
    variant_mask : np.ndarray or pd.DataFrame, optional
        Extra variant-level filter. Shapes accepted: ``(n_vars,)``,
        ``(n_cells, n_vars)``, ``(n_vars, n_cells)``, or a DataFrame
        ``(variants × cells)``. For ``aggregate='private_spectrum'`` only the
        ``(n_vars,)`` form is accepted (further restricts the private set).
    aggregate : {None, 'sum', 'private_spectrum', 'randomize'}, default None
        See description above.
    groupby : str, optional
        Column in ``adata.obs`` to aggregate within. Used by
        ``aggregate='sum'`` and ``aggregate='randomize'``. For
        ``aggregate='private_spectrum'`` the grouping is inherited from the
        ``private_mutations`` call and this argument is ignored.
    private_key : str, optional
        ``store_key`` previously passed to :func:`cellspec.tl.private_mutations`.
        Required when ``aggregate='private_spectrum'``.
    random_state : int or np.random.Generator, optional
        Seed/generator for ``aggregate='randomize'`` when used with
        ``groupby`` (ignored otherwise).
    key : str, optional
        Storage key. If ``None``, defaults to:

        - ``'per_cell'`` when ``aggregate=None``
        - ``aggregate`` name when ``groupby=None`` (e.g. ``'sum'``,
          ``'randomize'``)
        - ``f'{aggregate}_by_{groupby}'`` when ``groupby`` is given
        - ``f'private_spectrum_from_{private_key}'`` for private spectra
    csv_path : str, optional
        If given, writes the computed spectrum DataFrame/Series to this path
        as CSV. Parent directories are created automatically.
    show_progress : bool, default True
        Draw a tqdm progress bar for the per-cell pass.
    description : str, optional
        Free-form string stored on metadata.

    Returns
    -------
    pd.DataFrame or pd.Series
        - ``aggregate=None``: DataFrame ``(n_cells, 96)``
        - ``aggregate='sum' | 'randomize'`` + ``groupby=None``: Series indexed by
          the 96 contexts
        - All other cases: DataFrame ``(96, n_owners_or_groups)`` with the 96
          contexts as the row index

    Examples
    --------
    >>> import cellspec as spc
    >>> # Per-cell matrix (default)
    >>> spc.tl.compute_spectrum(adata)
    >>> adata.obsm["spectrum_per_cell"]  # (n_cells, 96)

    >>> # Pooled sum across all cells
    >>> spc.tl.compute_spectrum(adata, aggregate="sum")
    >>> adata.uns["spectrum_sum"]  # Series of length 96

    >>> # Sum within lineage groups, write a CSV copy
    >>> spc.tl.compute_spectrum(adata, aggregate="sum", groupby="lineage", csv_path="spectra/sum_by_lineage.csv")

    >>> # Private-mutation spectrum per lineage
    >>> spc.tl.private_mutations(adata, groupby="lineage", store_key="private")
    >>> spc.tl.compute_spectrum(adata, aggregate="private_spectrum", private_key="private")

    >>> # Randomized assignment per lineage (reproducible)
    >>> spc.tl.compute_spectrum(adata, aggregate="randomize", groupby="lineage", random_state=0)

    Notes
    -----
    **Randomize algorithm.** For each variant *j* with at least one carrier,
    draw a uniform ``U[0, 1)`` weight for every cell that carries it (and 0 for
    every cell that doesn't). Pick the cell with the highest weight — ties with
    the ``0`` entries are impossible since draws are continuous. Credit the
    variant to that cell (``groupby=None``) or to that cell's group
    (``groupby='…'``). Because the draw weights are per-cell, groups with more
    carrier cells are proportionally more likely to be chosen.
    """
    if "trinuc_type" not in adata.var.columns:
        raise ValueError("'trinuc_type' not found in adata.var. Run spc.pp.annotate_contexts() first.")

    if aggregate is not None and aggregate not in AGGREGATE_MODES:
        raise ValueError(f"aggregate must be one of {AGGREGATE_MODES} or None, got '{aggregate}'")

    if aggregate == "private_spectrum" and private_key is None:
        raise ValueError("aggregate='private_spectrum' requires private_key")
    if aggregate != "private_spectrum" and private_key is not None:
        raise ValueError("private_key only applies when aggregate='private_spectrum'")

    if groupby is not None and groupby not in adata.obs.columns:
        raise ValueError(f"groupby column '{groupby}' not found in adata.obs")

    if key is None:
        key = _default_key(aggregate, groupby, private_key)

    canonical = get_canonical_96_order()
    trinuc_types = adata.var["trinuc_type"].values

    # Private spectrum takes a dedicated path — no carrier matrix needed since
    # ownership is already encoded in the private mask.
    if aggregate == "private_spectrum":
        matrix, owners = _private_spectrum_counts(adata, private_key, variant_mask)
        spectrum_df = pd.DataFrame(matrix, index=canonical, columns=owners)
        result: pd.DataFrame | pd.Series = spectrum_df
        private_metadata = adata.uns.get(f"{private_key}_metadata", {})
        storage_metadata = {
            "aggregate": "private_spectrum",
            "private_key": private_key,
            "private_groupby": private_metadata.get("groupby"),
            "n_owners": len(owners),
            "owners": owners,
            "n_variants": adata.n_vars,
            "description": description,
        }
        _store_aggregated(adata, key, result, storage_metadata)
        _maybe_write_csv(result, csv_path)
        return result

    # Everything else goes through a cell × var carrier matrix.
    iterator_desc = f"Computing spectrum (key='{key}')"
    if show_progress:
        # Just a visual nod — the heavy work is vectorized below.
        with tqdm(total=1, desc=iterator_desc, bar_format="{desc}: {elapsed}") as _:
            carrier = _build_carrier_matrix(
                adata, count_strategy, genotypes, min_alt_depth, min_depth, max_depth, variant_mask
            )
    else:
        carrier = _build_carrier_matrix(
            adata, count_strategy, genotypes, min_alt_depth, min_depth, max_depth, variant_mask
        )

    storage_metadata = {
        "aggregate": aggregate,
        "count_strategy": count_strategy,
        "min_depth": min_depth,
        "max_depth": max_depth,
        "groupby": groupby,
        "description": description,
        "n_cells": adata.n_obs,
        "n_variants": adata.n_vars,
        "variant_mask_applied": variant_mask is not None,
    }
    if count_strategy == "genotype":
        storage_metadata["genotypes"] = genotypes
    if count_strategy == "alt_depth":
        storage_metadata["min_alt_depth"] = min_alt_depth

    if aggregate is None:
        per_cell = _per_cell_counts(carrier, trinuc_types)
        spectrum_df = pd.DataFrame(per_cell, index=adata.obs_names, columns=canonical)
        adata.obsm[f"spectrum_{key}"] = spectrum_df
        adata.uns[f"spectrum_{key}_metadata"] = storage_metadata
        _maybe_write_csv(spectrum_df, csv_path)
        return spectrum_df

    if aggregate == "sum":
        if groupby is None:
            per_cell = _per_cell_counts(carrier, trinuc_types)
            total = per_cell.sum(axis=0).astype(np.int64)
            result = pd.Series(total, index=canonical, name=key)
        else:
            group_ids = adata.obs[groupby].to_numpy()
            matrix, groups = _groupby_counts(carrier, trinuc_types, group_ids)
            result = pd.DataFrame(matrix, index=canonical, columns=groups)
            storage_metadata["n_groups"] = len(groups)
            storage_metadata["groups"] = groups
    elif aggregate == "randomize":
        if groupby is None:
            matrix, _ = _randomize_counts(carrier, trinuc_types, None, random_state)
            result = pd.Series(matrix, index=canonical, name=key)
        else:
            group_ids = adata.obs[groupby].to_numpy()
            matrix, groups = _randomize_counts(carrier, trinuc_types, group_ids, random_state)
            result = pd.DataFrame(matrix, index=canonical, columns=groups)
            storage_metadata["n_groups"] = len(groups)
            storage_metadata["groups"] = groups
            storage_metadata["random_state"] = random_state if isinstance(random_state, int) else None

    _store_aggregated(adata, key, result, storage_metadata)
    _maybe_write_csv(result, csv_path)
    return result


def _store_aggregated(adata: ad.AnnData, key: str, result: pd.DataFrame | pd.Series, metadata: dict) -> None:
    """Write an aggregated spectrum result + metadata into ``adata.uns``."""
    adata.uns[f"spectrum_{key}"] = result
    adata.uns[f"spectrum_{key}_metadata"] = metadata


def _maybe_write_csv(result: pd.DataFrame | pd.Series, csv_path: str | None) -> None:
    """Write ``result`` to ``csv_path`` if given, creating parent dirs."""
    if csv_path is None:
        return
    parent = os.path.dirname(csv_path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    if isinstance(result, pd.Series):
        result.to_frame(name=result.name or "count").to_csv(csv_path)
    else:
        result.to_csv(csv_path)
