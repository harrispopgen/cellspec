"""Plot mutation spectra."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import anndata as ad
from typing import Optional, Dict, List, Tuple, Union
import os

from ..utils.constants import MUTATION_TYPES, MUTATION_COLORS, get_canonical_96_order


def spectrum(
    adata: ad.AnnData,
    key: str = "spectrum",
    cells: Optional[List[str]] = None,
    groupby: Optional[str] = None,
    aggregate: str = 'sum',
    normalize: bool = False,
    outdir: Optional[str] = None,
    figsize: Tuple[int, int] = (24, 6),
    dpi: int = 600,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot mutation spectra for cells/samples or per-group private mutations.

    Automatically detects whether spectrum is per-cell (adata.obsm) or
    per-group (adata.uns) and plots accordingly.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with computed spectrum
    key : str, default 'spectrum'
        Key for spectrum in adata.obsm (per-cell) or adata.uns (per-group)
    cells : list of str, optional
        Specific cells to plot. If None and groupby is None and aggregate is None,
        plots all cells as separate files.
        Note: Cannot be used with per-group spectra.
    groupby : str, optional
        Column name from adata.obs to group by (e.g., 'cell_type', 'sample_id').
        If provided, plots one spectrum per group, aggregated by the method specified in 'aggregate'.
        If None and aggregate is provided, aggregates across all cells.
        If None and aggregate is None, plots individual cells.
        Note: Cannot be used with per-group spectra (already grouped).
    aggregate : str, default 'sum'
        How to aggregate spectra: 'sum', 'mean', or None.
        - If groupby is provided: aggregates within each group
        - If groupby is None: aggregates across all cells
        - If None: plots individual cells (ignores groupby)
        Note: For per-group spectra, must be 'sum' or None.
    normalize : bool, default False
        Normalize to proportions (sum to 1).
        For advanced normalization (by callable sites, coverage, etc.),
        use spc.tl.normalize_spectrum() first and plot the normalized key.
    outdir : str, optional
        Output directory for saving plots
    figsize : tuple, default (24, 6)
        Figure size (per spectrum row)
    dpi : int, default 600
        DPI for saved figures
    title : str, optional
        Plot title (used for aggregate plots or individual cell prefix)
    ylabel : str, optional
        Y-axis label. If None, uses "Mutation count" or "Proportion" based on normalize parameter
    ylim : tuple of (float, float), optional
        Y-axis limits as (ymin, ymax). If None, automatically calculated from data.

    Examples
    --------
    >>> import cellspec as spc
    >>> # Per-cell spectra examples:
    >>> # Plot all cells as separate files
    >>> spc.pl.spectrum(adata, key='somatic', outdir='figures/somatic/')

    >>> # Plot aggregate spectrum (sum across all cells)
    >>> spc.pl.spectrum(adata, key='somatic', aggregate='sum')

    >>> # Plot aggregate spectrum (mean across all cells)
    >>> spc.pl.spectrum(adata, key='somatic', aggregate='mean')

    >>> # Plot spectra grouped by cell type (sum within each group)
    >>> spc.pl.spectrum(adata, key='somatic', groupby='cell_type')

    >>> # Plot spectra grouped by sample (mean within each group)
    >>> spc.pl.spectrum(adata, key='somatic', groupby='sample_id', aggregate='mean')

    >>> # Plot individual cells
    >>> spc.pl.spectrum(adata, key='somatic', aggregate=None)

    >>> # Plot specific cells
    >>> spc.pl.spectrum(adata, key='somatic', cells=['cell_1', 'cell_2', 'cell_3'], aggregate=None)

    >>> # Pre-normalize by callable sites then plot
    >>> spc.tl.normalize_spectrum(adata, key='somatic', method='obs_column',
    ...                           obs_column='callable_sites', output_key='somatic_rate')
    >>> spc.pl.spectrum(adata, key='somatic_rate', aggregate='sum')

    >>> # Per-group spectra examples:
    >>> # Compute and plot private mutation spectra per lineage
    >>> spc.tl.private_mutations(adata, groupby='lineage', genotypes=[3], store_key='private')
    >>> spc.tl.compute_spectrum(adata, genotypes=[3], private_key='private', key='private_spectra')
    >>> spc.pl.spectrum(adata, key='private_spectra')  # Automatically detects per-group mode

    Notes
    -----
    **Per-cell mode:**
    - Spectrum stored in adata.obsm[f'spectrum_{key}']
    - Shape: (cells × 96 contexts)
    - Supports groupby, aggregate, cells parameters

    **Per-group mode:**
    - Spectrum stored in adata.uns[f'spectrum_{key}']
    - Shape: (96 contexts × groups)
    - Already grouped/aggregated by private_mutations groups
    - Does not support groupby, cells parameters (raises error)
    - Automatically plots one row per group

    For large datasets (>10 cells), using aggregate='sum' or groupby='cell_type'
    or saving individual files to outdir is recommended over plotting all cells in one figure.

    For advanced normalization strategies (e.g., by callable sites, coverage,
    or custom vectors), use spc.tl.normalize_spectrum() before plotting.
    """
    # Get spectrum - check both obsm (per-cell) and uns (per-group)
    spectrum_key = f'spectrum_{key}'
    metadata_key = f'spectrum_{key}_metadata'

    # Determine mode by checking metadata
    is_per_group = False
    if metadata_key in adata.uns:
        metadata = adata.uns[metadata_key]
        is_per_group = metadata.get('mode') == 'per_group'

    if is_per_group:
        # Per-group mode: data in adata.uns, shape (96 contexts × groups)
        if spectrum_key not in adata.uns:
            raise ValueError(
                f"'{spectrum_key}' not found in adata.uns. "
                f"Run spc.tl.compute_spectrum(..., key='{key}', private_key='...') first."
            )
        spectrum_df = adata.uns[spectrum_key]

        # Validate incompatible parameters
        if cells is not None:
            raise ValueError("'cells' parameter cannot be used with per-group spectra")
        if groupby is not None:
            raise ValueError("'groupby' parameter cannot be used with per-group spectra (already grouped)")
        if aggregate is not None and aggregate != 'sum':
            raise ValueError("'aggregate' parameter must be 'sum' or None for per-group spectra")

        # Convert to dict for plotting (spectrum_df is 96 contexts × groups)
        spectra_dict = {col: spectrum_df[col] for col in spectrum_df.columns}

        # Plot multiple spectra
        _plot_multiple_spectra(
            spectra_dict,
            normalize=normalize,
            title=title or f"Private Mutation Spectra",
            outpath=os.path.join(outdir, f"{key}_private.png") if outdir else None,
            figsize=figsize,
            dpi=dpi,
            ylabel=ylabel,
            ylim=ylim
        )
        return

    # Per-cell mode: data in adata.obsm, shape (cells × 96 contexts)
    if spectrum_key not in adata.obsm:
        raise ValueError(
            f"'{spectrum_key}' not found in adata.obsm. "
            f"Run spc.tl.compute_spectrum(..., key='{key}') first."
        )

    spectrum_df = adata.obsm[spectrum_key]

    # Select cells
    if cells is not None:
        spectrum_df = spectrum_df.loc[cells, :]

    # Determine plotting mode
    if groupby is not None:
        # Group by column in .obs
        if groupby not in adata.obs.columns:
            raise ValueError(
                f"groupby column '{groupby}' not found in adata.obs. "
                f"Available columns: {list(adata.obs.columns)}"
            )

        if aggregate not in ['sum', 'mean']:
            raise ValueError(f"aggregate must be 'sum' or 'mean' when using groupby, got '{aggregate}'")

        spectra_dict = {}
        for group_name, group_indices in adata.obs.groupby(groupby).groups.items():
            # Get cells in this group that are in spectrum_df
            group_cells = [cell for cell in group_indices if cell in spectrum_df.index]
            if len(group_cells) > 0:
                if aggregate == 'sum':
                    spectra_dict[str(group_name)] = spectrum_df.loc[group_cells, :].sum(axis=0)
                else:  # mean
                    spectra_dict[str(group_name)] = spectrum_df.loc[group_cells, :].mean(axis=0)

        # Plot multiple spectra
        _plot_multiple_spectra(
            spectra_dict,
            normalize=normalize,
            title=title or f"Spectra by {groupby}",
            outpath=os.path.join(outdir, f"{key}_by_{groupby}.png") if outdir else None,
            figsize=figsize,
            dpi=dpi,
            ylabel=ylabel,
            ylim=ylim
        )

    elif aggregate is not None:
        # Aggregate across all cells
        if aggregate == 'sum':
            aggregated = spectrum_df.sum(axis=0)
        elif aggregate == 'mean':
            aggregated = spectrum_df.mean(axis=0)
        else:
            raise ValueError(f"aggregate must be 'sum', 'mean', or None, got '{aggregate}'")

        # Plot single spectrum
        _plot_single_spectrum(
            aggregated,
            normalize=normalize,
            title=title or f"Aggregated Spectrum ({aggregate})",
            outpath=os.path.join(outdir, f"{key}_aggregated.png") if outdir else None,
            figsize=figsize,
            dpi=dpi,
            ylabel=ylabel,
            ylim=ylim
        )

    else:
        # Plot individual cells
        if outdir:
            os.makedirs(outdir, exist_ok=True)

        for cell_name in spectrum_df.index:
            cell_spectrum = spectrum_df.loc[cell_name, :]
            cell_title = f"{title} - {cell_name}" if title else cell_name
            outpath = os.path.join(outdir, f"{cell_name}_spectrum.png") if outdir else None

            _plot_single_spectrum(
                cell_spectrum,
                normalize=normalize,
                title=cell_title,
                outpath=outpath,
                figsize=figsize,
                dpi=dpi,
                ylabel=ylabel,
                ylim=ylim
            )

            if outdir:
                print(f"Saved: {outpath}")


def spectrum_from_df(
    spectrum_df: pd.DataFrame,
    normalize: bool = False,
    title: Optional[str] = None,
    outpath: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 600,
    ylabel: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Plot mutation spectrum directly from a DataFrame.

    Automatically detects format based on shape:
    - If shape is (n_samples, 96): treats as per-cell format (cells × contexts)
    - If shape is (96, n_groups): treats as per-group format (contexts × groups)
    - If shape is (96,): treats as single spectrum

    Parameters
    ----------
    spectrum_df : pd.DataFrame or pd.Series
        Mutation spectrum DataFrame or Series:
        - Series with 96 trinuc contexts: plots single spectrum
        - DataFrame (cells × 96 contexts): plots multiple spectra, one row per cell
        - DataFrame (96 contexts × groups): plots multiple spectra, one row per group
    normalize : bool, default False
        Normalize to proportions (sum to 1)
    title : str, optional
        Plot title
    outpath : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size. If None, auto-calculated based on number of spectra
    dpi : int, default 600
        DPI for saved figure
    ylabel : str, optional
        Y-axis label. If None, uses "Mutation count" or "Proportion" based on normalize
    ylim : tuple of (float, float), optional
        Y-axis limits as (ymin, ymax). If None, automatically calculated from data.

    Examples
    --------
    >>> import cellspec as spc
    >>> import pandas as pd
    >>>
    >>> # Plot a single spectrum (Series)
    >>> spectrum = adata.obsm['spectrum_somatic'].sum(axis=0)
    >>> spc.pl.spectrum_from_df(spectrum, title='Total Somatic Mutations')
    >>>
    >>> # Plot from custom DataFrame (96 contexts × groups)
    >>> private_spectra = adata.uns['spectrum_private']
    >>> spc.pl.spectrum_from_df(private_spectra, title='Private Mutations', normalize=True)
    >>>
    >>> # Plot from subset of cells (cells × 96 contexts)
    >>> cell_subset = adata.obsm['spectrum_somatic'].loc[['cell1', 'cell2', 'cell3']]
    >>> spc.pl.spectrum_from_df(cell_subset, title='Selected Cells')

    Notes
    -----
    This function automatically detects the DataFrame format and plots accordingly.
    For more control over grouping and aggregation from AnnData objects, use spc.pl.spectrum().
    """
    # Handle Series (single spectrum)
    if isinstance(spectrum_df, pd.Series):
        if figsize is None:
            figsize = (24, 6)
        _plot_single_spectrum(
            spectrum_df,
            normalize=normalize,
            title=title,
            outpath=outpath,
            figsize=figsize,
            dpi=dpi,
            ylabel=ylabel,
            ylim=ylim
        )
        return

    # Handle DataFrame
    if not isinstance(spectrum_df, pd.DataFrame):
        raise TypeError(f"spectrum_df must be pd.DataFrame or pd.Series, got {type(spectrum_df)}")

    # Detect format based on shape
    n_rows, n_cols = spectrum_df.shape

    # Check if it's 96-channel spectrum
    canonical_contexts = get_canonical_96_order()

    # Determine orientation
    if n_cols == 96:
        # Per-cell format: (cells × 96 contexts)
        # Plot as multiple spectra, one per row
        spectra_dict = {idx: spectrum_df.loc[idx, :] for idx in spectrum_df.index}
    elif n_rows == 96:
        # Per-group format: (96 contexts × groups)
        # Plot as multiple spectra, one per column
        spectra_dict = {col: spectrum_df[col] for col in spectrum_df.columns}
    else:
        raise ValueError(
            f"Unexpected spectrum shape {spectrum_df.shape}. "
            f"Expected either (n_samples, 96) or (96, n_groups) for 96-channel spectra."
        )

    # Auto-calculate figsize if not provided
    if figsize is None:
        n_spectra = len(spectra_dict)
        figsize = (24, 6 * n_spectra)

    # Plot multiple spectra
    _plot_multiple_spectra(
        spectra_dict,
        normalize=normalize,
        title=title,
        outpath=outpath,
        figsize=figsize,
        dpi=dpi,
        ylabel=ylabel,
        ylim=ylim
    )


def compare_spectra(
    adata: ad.AnnData,
    keys: List[str],
    aggregate: str = 'sum',
    normalize: bool = False,
    outpath: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 600,
    title: Optional[str] = None,
    ylabel: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    """
    Compare multiple spectra (e.g., different filtering strategies or conditions).

    Supports both per-cell and per-group spectra. Per-group spectra are already
    aggregated, so each group will appear as a separate row in the plot.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with computed spectra
    keys : list of str
        Keys for spectra (in adata.obsm for per-cell or adata.uns for per-group)
    aggregate : str, default 'sum'
        How to aggregate each per-cell spectrum: 'sum' or 'mean'.
        For per-group spectra, must be 'sum' (already aggregated).
    normalize : bool, default False
        Normalize to proportions.
        For advanced normalization (by callable sites, coverage, etc.),
        use spc.tl.normalize_spectrum() first and pass normalized keys.
    outpath : str, optional
        Path to save figure
    figsize : tuple, optional
        Figure size. If None, auto-calculated based on number of spectra
    dpi : int, default 600
        DPI for saved figure
    title : str, optional
        Overall plot title (displayed at the top)
    ylabel : str, optional
        Y-axis label. If None, uses "Mutation count" or "Proportion" based on normalize parameter
    ylim : tuple of (float, float), optional
        Y-axis limits as (ymin, ymax). If None, automatically calculated from data.

    Examples
    --------
    >>> import cellspec as spc
    >>> # Compare germline vs somatic (per-cell spectra)
    >>> spc.pl.compare_spectra(
    ...     adata,
    ...     keys=['germline', 'somatic'],
    ...     aggregate='sum',
    ...     normalize=True,
    ...     title='Germline vs Somatic',
    ...     outpath='figures/comparison.png'
    ... )
    >>>
    >>> # Compare with rate normalization
    >>> spc.tl.normalize_spectrum(adata, key='germline', method='obs_column',
    ...                           obs_column='callable_sites', output_key='germline_rate')
    >>> spc.tl.normalize_spectrum(adata, key='somatic', method='obs_column',
    ...                           obs_column='callable_sites', output_key='somatic_rate')
    >>> spc.pl.compare_spectra(
    ...     adata,
    ...     keys=['germline_rate', 'somatic_rate'],
    ...     aggregate='sum',
    ...     title='Germline vs Somatic (per callable site)'
    ... )
    >>>
    >>> # Compare per-group private mutation spectra
    >>> # If 'private_spectra' has 3 groups, this will show all 3 in one plot
    >>> spc.pl.compare_spectra(
    ...     adata,
    ...     keys=['private_spectra'],
    ...     title='Private Mutations by Lineage'
    ... )
    """
    # Collect spectra
    spectra_dict = {}
    for key in keys:
        spectrum_key = f'spectrum_{key}'
        metadata_key = f'spectrum_{key}_metadata'

        # Check if per-group mode
        is_per_group = False
        if metadata_key in adata.uns:
            metadata = adata.uns[metadata_key]
            is_per_group = metadata.get('mode') == 'per_group'

        if is_per_group:
            # Per-group mode: data in adata.uns
            if spectrum_key not in adata.uns:
                raise ValueError(f"'{spectrum_key}' not found in adata.uns")

            if aggregate != 'sum':
                raise ValueError(
                    f"For per-group spectrum '{key}', aggregate must be 'sum' (data is already aggregated)"
                )

            # Per-group spectra are already aggregated
            # Add each group as a separate entry with key_groupname
            spectrum_df = adata.uns[spectrum_key]
            for group in spectrum_df.columns:
                spectra_dict[f"{key}_{group}"] = spectrum_df[group]

        else:
            # Per-cell mode: data in adata.obsm
            if spectrum_key not in adata.obsm:
                raise ValueError(f"'{spectrum_key}' not found in adata.obsm")

            spectrum_df = adata.obsm[spectrum_key]

            # Aggregate
            if aggregate == 'sum':
                spectra_dict[key] = spectrum_df.sum(axis=0)
            elif aggregate == 'mean':
                spectra_dict[key] = spectrum_df.mean(axis=0)
            else:
                raise ValueError(f"Invalid aggregate '{aggregate}'. Must be 'sum' or 'mean'.")

    # Plot
    _plot_multiple_spectra(
        spectra_dict,
        normalize=normalize,
        title=title or "Spectrum Comparison",
        outpath=outpath,
        figsize=figsize,
        dpi=dpi,
        ylabel=ylabel,
        ylim=ylim
    )


def _plot_single_spectrum(
    spectrum: pd.Series,
    normalize: bool = False,
    title: Optional[str] = None,
    outpath: Optional[str] = None,
    figsize: Tuple[int, int] = (24, 6),
    dpi: int = 600,
    ylabel: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    """Plot a single mutation spectrum."""
    if normalize and spectrum.sum() > 0:
        spectrum = spectrum / spectrum.sum()

    # Create figure
    fig, axes = plt.subplots(1, 6, figsize=figsize, sharey=True)
    if title:
        fig.suptitle(title, fontsize=20, fontweight='bold')

    # Get colors
    colors = [MUTATION_COLORS[t] for t in MUTATION_TYPES]

    # Calculate y-axis limits
    if ylim is not None:
        y_min, y_max = ylim
    else:
        y_min = 0
        y_max = spectrum.max() * 1.15

    # Determine y-axis label
    if ylabel is None:
        ylabel = "Proportion" if normalize else "Mutation count"

    # Plot each mutation type
    for col, mut_type in enumerate(MUTATION_TYPES):
        ax = axes[col]

        # Get contexts for this mutation type
        contexts = [c for c in spectrum.index if f"{c[1]}>{c[5]}" == mut_type]

        # Get counts
        counts = [spectrum[c] for c in contexts]

        # Plot bars
        x_pos = np.arange(len(contexts))
        ax.bar(x_pos, counts, color=colors[col], width=0.8, edgecolor='none')

        # Add colored background patch
        rect = patches.Rectangle(
            (0, y_max * 0.98), len(contexts), y_max * 0.02,
            linewidth=0, edgecolor='none', facecolor=colors[col]
        )
        ax.add_patch(rect)

        # Style
        ax.set_xlim(-0.5, len(contexts) - 0.5)
        ax.set_ylim(y_min, y_max)
        ax.set_title(mut_type, fontweight='bold', fontsize=16)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([c[0:3] for c in contexts], rotation=90, fontsize=8)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

        # Y-label only on first subplot
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=14)

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def _plot_multiple_spectra(
    spectra_dict: Dict[str, pd.Series],
    normalize: bool = False,
    title: Optional[str] = None,
    outpath: Optional[str] = None,
    figsize: Optional[Tuple[int, int]] = None,
    dpi: int = 600,
    ylabel: Optional[str] = None,
    ylim: Optional[Tuple[float, float]] = None
) -> None:
    """Plot multiple spectra for comparison."""
    n_spectra = len(spectra_dict)

    if figsize is None:
        figsize = (24, 6 * n_spectra)

    fig, axes = plt.subplots(n_spectra, 6, figsize=figsize, sharey='row')
    if title:
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.995)

    # Handle single spectrum case
    if n_spectra == 1:
        axes = axes.reshape(1, -1)

    # Get colors
    colors = [MUTATION_COLORS[t] for t in MUTATION_TYPES]

    # Calculate global y-limits
    if ylim is not None:
        y_min, y_max = ylim
    else:
        all_values = [s.values for s in spectra_dict.values()]
        if normalize:
            all_values = [v / v.sum() if v.sum() > 0 else v for v in all_values]
        max_value = max(v.max() for v in all_values)
        y_min = 0
        y_max = max_value * 1.15

    # Determine y-axis label
    if ylabel is None:
        ylabel = "Proportion" if normalize else "Mutation count"

    # Plot each spectrum
    for row, (spectrum_name, spectrum) in enumerate(spectra_dict.items()):
        if normalize and spectrum.sum() > 0:
            spectrum = spectrum / spectrum.sum()

        # Plot each mutation type
        for col, mut_type in enumerate(MUTATION_TYPES):
            ax = axes[row, col]

            # Get contexts for this mutation type
            contexts = [c for c in spectrum.index if f"{c[1]}>{c[5]}" == mut_type]

            # Get counts
            counts = [spectrum[c] for c in contexts]

            # Plot bars
            x_pos = np.arange(len(contexts))
            ax.bar(x_pos, counts, color=colors[col], width=0.8, edgecolor='none')

            # Add colored background patch
            rect = patches.Rectangle(
                (0, y_max * 0.98), len(contexts), y_max * 0.02,
                linewidth=0, edgecolor='none', facecolor=colors[col]
            )
            ax.add_patch(rect)

            # Style
            ax.set_xlim(-0.5, len(contexts) - 0.5)
            ax.set_ylim(y_min, y_max)

            # Add spectrum name as title on first subplot of each row
            if col == 0:
                # Add sample/group name above the leftmost subplot
                ax.text(-0.15, 1.15, spectrum_name, transform=ax.transAxes,
                       fontsize=16, fontweight='bold', va='center', ha='right')

            # Mutation type as title (smaller for multi-row plots)
            ax.set_title(mut_type, fontweight='bold', fontsize=14)
            ax.set_xticks(x_pos)
            ax.set_xticklabels([c[0:3] for c in contexts], rotation=90, fontsize=6)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)

            # Y-label only on first column
            if col == 0:
                ax.set_ylabel(ylabel, fontsize=12)

    plt.tight_layout()

    if outpath:
        plt.savefig(outpath, dpi=dpi, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
