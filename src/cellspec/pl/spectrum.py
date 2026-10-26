"""Plot mutation spectra."""

import os

import anndata as ad
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

from cellspec.utils.constants import MUTATION_COLORS, MUTATION_COLORS_DARK, MUTATION_TYPES

# Default style knobs shared by _plot_single_spectrum and _plot_multiple_spectra.
# These describe the "ribbon" layout — six panels butted edge-to-edge, y-axis
# only on the leftmost panel, monospace trinucleotide x-tick labels.
_STYLE_DEFAULTS = {
    "bar_width": 0.8,
    "group_padding": 0.5,
    "left_pad": 0.25,
    "wspace": 0.0,
    "ylabel_fontsize": 16,
    "ytick_fontsize": 16,
    "xtick_fontsize": 8,
    "panel_title_fontsize": 14,
    "row_label_fontsize": 16,
    "suptitle_fontsize": 20,
    "legend_fontsize": 14,
}


def _resolve_palette(palette: str | list[str] | dict[str, str] | None) -> dict[str, str]:
    """
    Resolve palette specification to a color dictionary.

    Parameters
    ----------
    palette : str, list, dict, or None
        Palette specification:
        - None: use default MUTATION_COLORS
        - 'default': use default MUTATION_COLORS
        - 'dark': use MUTATION_COLORS_DARK (for dark backgrounds)
        - list of 6 colors: colors for ['C>A', 'C>G', 'C>T', 'T>A', 'T>C', 'T>G']
        - dict: mapping mutation types to colors (e.g., {'C>A': '#00BCD4', ...})

    Returns
    -------
    dict
        Mapping from mutation types to hex colors
    """
    if palette is None or palette == "default":
        return MUTATION_COLORS
    elif palette == "dark":
        return MUTATION_COLORS_DARK
    elif isinstance(palette, list):
        if len(palette) != 6:
            raise ValueError(f"Palette list must have exactly 6 colors, got {len(palette)}")
        return dict(zip(MUTATION_TYPES, palette, strict=True))
    elif isinstance(palette, dict):
        # Validate all mutation types are present
        missing = set(MUTATION_TYPES) - set(palette.keys())
        if missing:
            raise ValueError(f"Palette dict is missing mutation types: {missing}")
        return palette
    else:
        raise TypeError(
            f"palette must be None, 'default', 'dark', a list of 6 colors, or a dict mapping mutation types to colors. Got {type(palette)}"
        )


def spectrum(
    adata: ad.AnnData,
    key: str = "per_cell",
    *,
    cells: list[str] | None = None,
    normalize: bool = False,
    palette: str | list[str] | dict[str, str] | None = None,
    outpath: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 600,
    title: str | None = None,
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    transparent: bool = False,
    yticks: list[float] | np.ndarray | None = None,
    ytick_scale: float = 1.0,
    ytick_unit_label: str | None = None,
    show_mut_type_titles: bool = True,
    show_xticks: bool = True,
    show_legend: bool = False,
    bar_width: float = _STYLE_DEFAULTS["bar_width"],
    group_padding: float = _STYLE_DEFAULTS["group_padding"],
    left_pad: float = _STYLE_DEFAULTS["left_pad"],
    wspace: float = _STYLE_DEFAULTS["wspace"],
) -> None:
    """
    Plot a mutation spectrum stored by :func:`cellspec.tl.compute_spectrum`.

    The storage layout determines the plot: aggregation and grouping now live
    on ``tl.compute_spectrum`` and this function just reads whatever was
    written for ``key``.

    - ``adata.obsm[f'spectrum_{key}']`` — ``(n_cells, 96)`` DataFrame; renders
      as a multi-row ribbon (one row per cell). Use ``cells=`` to subset.
    - ``adata.uns[f'spectrum_{key}']`` — either a Series of length 96 (single
      ribbon) or a ``(96, n_groups)`` DataFrame (one row per group).

    Parameters
    ----------
    adata : ad.AnnData
    key : str, default 'per_cell'
        Key suffix. Looks up ``adata.obsm['spectrum_{key}']`` or
        ``adata.uns['spectrum_{key}']``.
    cells : list of str, optional
        Only honored when the spectrum is stored in ``adata.obsm`` (per-cell).
        Restricts the plot to the given cell names.
    normalize : bool, default False
        Convert each ribbon's counts to proportions (sum to 1).
    palette : str, list, dict, or None, optional
        Mutation-type color palette (see :func:`cellspec.pl.spectrum_from_df`).
    outpath : str, optional
        If given, saves the figure to this path (parent dirs auto-created).
    figsize : tuple, optional
        Figure size. Defaults to ``(12, 3 * n_rows)``.
    dpi : int, default 600
    title, ylabel, ylim, transparent, yticks, ytick_scale, ytick_unit_label,
    show_mut_type_titles, show_xticks, show_legend, bar_width, group_padding,
    left_pad, wspace :
        Style knobs; see :func:`cellspec.pl.spectrum_from_df`.

    Examples
    --------
    >>> import cellspec as spc
    >>> # Per-cell matrix: plot one row per cell (subset first for large N)
    >>> spc.tl.compute_spectrum(adata)
    >>> spc.pl.spectrum(adata, key="per_cell", cells=adata.obs_names[:5])
    >>>
    >>> # Pooled sum across all cells (Series)
    >>> spc.tl.compute_spectrum(adata, aggregate="sum")
    >>> spc.pl.spectrum(adata, key="sum")
    >>>
    >>> # Sum by group (one ribbon per group)
    >>> spc.tl.compute_spectrum(adata, aggregate="sum", groupby="lineage")
    >>> spc.pl.spectrum(adata, key="sum_by_lineage")
    >>>
    >>> # Private spectrum per lineage
    >>> spc.tl.private_mutations(adata, groupby="lineage", store_key="private")
    >>> spc.tl.compute_spectrum(adata, aggregate="private_spectrum", private_key="private")
    >>> spc.pl.spectrum(adata, key="private_spectrum_from_private")
    """
    spectrum_key = f"spectrum_{key}"
    in_obsm = spectrum_key in adata.obsm
    in_uns = spectrum_key in adata.uns

    if not in_obsm and not in_uns:
        raise ValueError(
            f"'{spectrum_key}' not found in adata.obsm or adata.uns. "
            f"Run spc.tl.compute_spectrum(..., key='{key}') first."
        )
    if in_obsm and in_uns:
        raise ValueError(
            f"'{spectrum_key}' is set in both adata.obsm and adata.uns. Drop one before plotting to disambiguate."
        )

    if outpath:
        parent_dir = os.path.dirname(outpath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    style_kwargs = {
        "normalize": normalize,
        "palette": palette,
        "outpath": outpath,
        "dpi": dpi,
        "ylabel": ylabel,
        "ylim": ylim,
        "transparent": transparent,
        "yticks": yticks,
        "ytick_scale": ytick_scale,
        "ytick_unit_label": ytick_unit_label,
        "show_mut_type_titles": show_mut_type_titles,
        "show_xticks": show_xticks,
        "show_legend": show_legend,
        "bar_width": bar_width,
        "group_padding": group_padding,
        "left_pad": left_pad,
        "wspace": wspace,
    }

    if in_obsm:
        per_cell_df = adata.obsm[spectrum_key]
        if cells is not None:
            per_cell_df = per_cell_df.loc[cells, :]
        spectra_dict = {idx: per_cell_df.loc[idx, :] for idx in per_cell_df.index}
        _plot_multiple_spectra(
            spectra_dict,
            title=title,
            figsize=figsize or (12, 3 * max(1, len(spectra_dict))),
            **style_kwargs,
        )
        return

    obj = adata.uns[spectrum_key]
    if cells is not None:
        raise ValueError("'cells' only applies to per-cell spectra stored in adata.obsm.")

    if isinstance(obj, pd.Series):
        _plot_single_spectrum(obj, title=title, figsize=figsize or (12, 3), **style_kwargs)
    elif isinstance(obj, pd.DataFrame):
        spectra_dict = {col: obj[col] for col in obj.columns}
        _plot_multiple_spectra(
            spectra_dict,
            title=title,
            figsize=figsize or (12, 3 * max(1, len(spectra_dict))),
            **style_kwargs,
        )
    else:
        raise TypeError(f"adata.uns['{spectrum_key}'] has unsupported type {type(obj)}")


def spectrum_from_df(
    spectrum_df: pd.DataFrame,
    normalize: bool = False,
    palette: str | list[str] | dict[str, str] | None = None,
    title: str | None = None,
    outpath: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 600,
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    transparent: bool = False,
    yticks: list[float] | np.ndarray | None = None,
    ytick_scale: float = 1.0,
    ytick_unit_label: str | None = None,
    show_mut_type_titles: bool = True,
    show_xticks: bool = True,
    show_legend: bool = False,
    bar_width: float = _STYLE_DEFAULTS["bar_width"],
    group_padding: float = _STYLE_DEFAULTS["group_padding"],
    left_pad: float = _STYLE_DEFAULTS["left_pad"],
    wspace: float = _STYLE_DEFAULTS["wspace"],
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
    palette : str, list, dict, or None, optional
        Color palette (see spc.pl.spectrum for details)
    title : str, optional
        Plot title
    outpath : str, optional
        Full output path for saving plot (e.g., 'results/spectrum.png').
        Parent directory will be created if it doesn't exist.
        If None, plot is displayed but not saved.
    figsize : tuple, optional
        Figure size. If None, auto-calculated based on number of spectra
    dpi : int, default 600
        DPI for saved figure
    ylabel : str, optional
        Y-axis label. If None, uses "Mutation count" or "Proportion" based on normalize
    ylim : tuple of (float, float), optional
        Y-axis limits as (ymin, ymax). If None, automatically calculated from data.
    transparent : bool, default False
        Save figure with transparent background (only affects saved files)
    yticks, ytick_scale, ytick_unit_label, show_mut_type_titles, show_xticks,
    show_legend, bar_width, group_padding, left_pad, wspace :
        Style knobs forwarded to the plotting backend. See
        :func:`cellspec.pl.spectrum` for descriptions.

    Examples
    --------
    >>> import cellspec as spc
    >>> import pandas as pd
    >>>
    >>> # Plot a single spectrum (Series)
    >>> spectrum = adata.obsm["spectrum_somatic"].sum(axis=0)
    >>> spc.pl.spectrum_from_df(spectrum, title="Total Somatic Mutations")
    >>>
    >>> # Save plot to a file
    >>> spc.pl.spectrum_from_df(spectrum, outpath="results/spectrum.png")
    >>>
    >>> # Plot from custom DataFrame (96 contexts × groups)
    >>> private_spectra = adata.uns["spectrum_private"]
    >>> spc.pl.spectrum_from_df(private_spectra, title="Private Mutations", normalize=True)
    >>>
    >>> # Plot from subset of cells (cells × 96 contexts)
    >>> cell_subset = adata.obsm["spectrum_somatic"].loc[["cell1", "cell2", "cell3"]]
    >>> spc.pl.spectrum_from_df(cell_subset, title="Selected Cells")

    Notes
    -----
    This function automatically detects the DataFrame format and plots accordingly.
    For more control over grouping and aggregation from AnnData objects, use spc.pl.spectrum().
    """
    # Create parent directory if saving
    if outpath:
        parent_dir = os.path.dirname(outpath)
        if parent_dir:
            os.makedirs(parent_dir, exist_ok=True)

    # Handle Series (single spectrum)
    if isinstance(spectrum_df, pd.Series):
        if figsize is None:
            figsize = (12, 3)
        _plot_single_spectrum(
            spectrum_df,
            normalize=normalize,
            palette=palette,
            title=title,
            outpath=outpath,
            figsize=figsize,
            dpi=dpi,
            ylabel=ylabel,
            ylim=ylim,
            transparent=transparent,
            yticks=yticks,
            ytick_scale=ytick_scale,
            ytick_unit_label=ytick_unit_label,
            show_mut_type_titles=show_mut_type_titles,
            show_xticks=show_xticks,
            show_legend=show_legend,
            bar_width=bar_width,
            group_padding=group_padding,
            left_pad=left_pad,
            wspace=wspace,
        )
        if outpath:
            print(f"Saved: {outpath}")
        return

    # Handle DataFrame
    if not isinstance(spectrum_df, pd.DataFrame):
        raise TypeError(f"spectrum_df must be pd.DataFrame or pd.Series, got {type(spectrum_df)}")

    # Detect format based on shape
    n_rows, n_cols = spectrum_df.shape

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
        figsize = (12, 3 * n_spectra)

    _plot_multiple_spectra(
        spectra_dict,
        normalize=normalize,
        palette=palette,
        title=title,
        outpath=outpath,
        figsize=figsize,
        dpi=dpi,
        ylabel=ylabel,
        ylim=ylim,
        transparent=transparent,
        yticks=yticks,
        ytick_scale=ytick_scale,
        ytick_unit_label=ytick_unit_label,
        show_mut_type_titles=show_mut_type_titles,
        show_xticks=show_xticks,
        show_legend=show_legend,
        bar_width=bar_width,
        group_padding=group_padding,
        left_pad=left_pad,
        wspace=wspace,
    )
    if outpath:
        print(f"Saved: {outpath}")


def compare_spectra(
    adata: ad.AnnData,
    keys: list[str],
    normalize: bool = False,
    palette: str | list[str] | dict[str, str] | None = None,
    outpath: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 600,
    title: str | None = None,
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    transparent: bool = False,
    yticks: list[float] | np.ndarray | None = None,
    ytick_scale: float = 1.0,
    ytick_unit_label: str | None = None,
    show_mut_type_titles: bool = True,
    show_xticks: bool = True,
    show_legend: bool = False,
    bar_width: float = _STYLE_DEFAULTS["bar_width"],
    group_padding: float = _STYLE_DEFAULTS["group_padding"],
    left_pad: float = _STYLE_DEFAULTS["left_pad"],
    wspace: float = _STYLE_DEFAULTS["wspace"],
) -> None:
    """
    Compare multiple spectra on a single stacked figure.

    Each ``key`` is resolved against ``adata.obsm`` and ``adata.uns``:

    - per-cell matrix in ``adata.obsm['spectrum_{key}']`` → summed across all
      cells to produce one row
    - Series in ``adata.uns['spectrum_{key}']`` → used as-is (one row)
    - DataFrame ``(96, n_groups)`` in ``adata.uns['spectrum_{key}']`` → each
      group becomes its own row, labeled ``'{key}_{group}'``

    Parameters
    ----------
    adata : ad.AnnData
    keys : list of str
    normalize : bool, default False
        Normalize each row to proportions. For advanced normalization (by
        callable sites, coverage, etc.), use :func:`cellspec.tl.normalize_spectrum`
        first and pass the normalized keys.
    palette : str, list, dict, or None, optional
    outpath : str, optional
    figsize : tuple, optional
    dpi : int, default 600
    title : str, optional
    ylabel : str, optional
    ylim : tuple of (float, float), optional
    transparent : bool, default False
    yticks, ytick_scale, ytick_unit_label, show_mut_type_titles, show_xticks,
    show_legend, bar_width, group_padding, left_pad, wspace :
        Style knobs forwarded to the plotting backend. See
        :func:`cellspec.pl.spectrum` for descriptions.

    Examples
    --------
    >>> import cellspec as spc
    >>> # Compare two per-cell spectra, pooled across all cells
    >>> spc.pl.compare_spectra(adata, keys=["germline", "somatic"], normalize=True, outpath="figures/compare.png")
    >>>
    >>> # Compare per-group private spectra across keys
    >>> spc.pl.compare_spectra(adata, keys=["private_spectrum_from_private"])
    """
    spectra_dict: dict[str, pd.Series] = {}
    for key in keys:
        spectrum_key = f"spectrum_{key}"
        in_obsm = spectrum_key in adata.obsm
        in_uns = spectrum_key in adata.uns

        if not in_obsm and not in_uns:
            raise ValueError(f"'{spectrum_key}' not found in adata.obsm or adata.uns")
        if in_obsm and in_uns:
            raise ValueError(f"'{spectrum_key}' is set in both adata.obsm and adata.uns")

        if in_obsm:
            spectra_dict[key] = adata.obsm[spectrum_key].sum(axis=0)
        else:
            obj = adata.uns[spectrum_key]
            if isinstance(obj, pd.Series):
                spectra_dict[key] = obj
            elif isinstance(obj, pd.DataFrame):
                for group in obj.columns:
                    spectra_dict[f"{key}_{group}"] = obj[group]
            else:
                raise TypeError(f"adata.uns['{spectrum_key}'] has unsupported type {type(obj)}")

    _plot_multiple_spectra(
        spectra_dict,
        normalize=normalize,
        palette=palette,
        title=title or "Spectrum Comparison",
        outpath=outpath,
        figsize=figsize,
        dpi=dpi,
        ylabel=ylabel,
        ylim=ylim,
        transparent=transparent,
        yticks=yticks,
        ytick_scale=ytick_scale,
        ytick_unit_label=ytick_unit_label,
        show_mut_type_titles=show_mut_type_titles,
        show_xticks=show_xticks,
        show_legend=show_legend,
        bar_width=bar_width,
        group_padding=group_padding,
        left_pad=left_pad,
        wspace=wspace,
    )


def _plot_single_spectrum(
    spectrum: pd.Series,
    normalize: bool = False,
    palette: str | list[str] | dict[str, str] | None = None,
    title: str | None = None,
    outpath: str | None = None,
    figsize: tuple[float, float] = (12, 3),
    dpi: int = 600,
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    transparent: bool = False,
    yticks: list[float] | np.ndarray | None = None,
    ytick_scale: float = 1.0,
    ytick_unit_label: str | None = None,
    show_mut_type_titles: bool = True,
    show_xticks: bool = True,
    show_legend: bool = False,
    bar_width: float = _STYLE_DEFAULTS["bar_width"],
    group_padding: float = _STYLE_DEFAULTS["group_padding"],
    left_pad: float = _STYLE_DEFAULTS["left_pad"],
    wspace: float = _STYLE_DEFAULTS["wspace"],
) -> None:
    """Plot a single mutation spectrum in the ribbon layout."""
    if normalize and spectrum.sum() > 0:
        spectrum = spectrum / spectrum.sum()

    fig, axes = plt.subplots(1, 6, figsize=figsize, sharey=True)

    color_map = _resolve_palette(palette)
    colors = [color_map[t] for t in MUTATION_TYPES]

    if ylim is not None:
        y_min, y_max = ylim
    else:
        y_min = 0
        y_max = (spectrum.max() * 1.15) if spectrum.max() > 0 else 1.0

    if ylabel is None:
        ylabel = "Proportion" if normalize else "Mutation count"
    if ytick_unit_label is not None:
        ylabel = f"{ylabel} ({ytick_unit_label})"

    for col, mut_type in enumerate(MUTATION_TYPES):
        ax = axes[col]

        contexts = [c for c in spectrum.index if f"{c[1]}>{c[5]}" == mut_type]
        counts = [spectrum[c] for c in contexts]

        if col == 0:
            x_pos = np.arange(len(contexts)) + left_pad
            ax.set_xlim(-0.5, len(contexts) - 0.5 + left_pad + group_padding)
        else:
            x_pos = np.arange(len(contexts))
            ax.set_xlim(-0.5 - group_padding, len(contexts) - 0.5 + group_padding)

        ax.bar(x_pos, counts, color=colors[col], width=bar_width, edgecolor="none")

        ax.set_ylim(y_min, y_max)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        if show_mut_type_titles:
            ax.set_title(mut_type, fontweight="bold", fontsize=_STYLE_DEFAULTS["panel_title_fontsize"])

        if show_xticks:
            ax.set_xticks(x_pos)
            ax.set_xticklabels(
                [c[:3] for c in contexts],
                rotation=90,
                fontsize=_STYLE_DEFAULTS["xtick_fontsize"],
                family="monospace",
                ha="center",
                va="top",
            )
            ax.tick_params(axis="x", length=0, pad=2)
        else:
            ax.set_xticks([])

        if col == 0:
            ax.tick_params(axis="y", labelsize=_STYLE_DEFAULTS["ytick_fontsize"])
            ax.set_ylabel(ylabel, fontsize=_STYLE_DEFAULTS["ylabel_fontsize"])
            if yticks is not None:
                ax.set_yticks(yticks)
            if ytick_scale != 1.0:
                ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _, s=ytick_scale: f"{v / s:g}"))
        else:
            ax.tick_params(axis="y", left=False, labelleft=False)
            ax.spines["left"].set_visible(False)

    if show_legend:
        handles = [patches.Patch(facecolor=color_map[t], label=t) for t in MUTATION_TYPES]
        fig.legend(
            handles=handles,
            ncol=6,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, 1.05 if not title else 1.0),
            fontsize=_STYLE_DEFAULTS["legend_fontsize"],
            handlelength=1.2,
            columnspacing=2.0,
            handletextpad=0.5,
        )

    if title:
        # Push suptitle above the panel-title row so it doesn't collide with
        # the C>A / C>G / … labels that `show_mut_type_titles` draws.
        suptitle_y = 1.25 if show_mut_type_titles else 1.1
        fig.suptitle(title, fontsize=_STYLE_DEFAULTS["suptitle_fontsize"], fontweight="bold", y=suptitle_y)

    plt.subplots_adjust(wspace=wspace)

    if outpath:
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight", transparent=transparent)
        plt.close()
    else:
        plt.show()


def _plot_multiple_spectra(
    spectra_dict: dict[str, pd.Series],
    normalize: bool = False,
    palette: str | list[str] | dict[str, str] | None = None,
    title: str | None = None,
    outpath: str | None = None,
    figsize: tuple[float, float] | None = None,
    dpi: int = 600,
    ylabel: str | None = None,
    ylim: tuple[float, float] | None = None,
    transparent: bool = False,
    yticks: list[float] | np.ndarray | None = None,
    ytick_scale: float = 1.0,
    ytick_unit_label: str | None = None,
    show_mut_type_titles: bool = True,
    show_xticks: bool = True,
    show_legend: bool = False,
    bar_width: float = _STYLE_DEFAULTS["bar_width"],
    group_padding: float = _STYLE_DEFAULTS["group_padding"],
    left_pad: float = _STYLE_DEFAULTS["left_pad"],
    wspace: float = _STYLE_DEFAULTS["wspace"],
) -> None:
    """Plot multiple spectra in a stacked ribbon layout (one row per sample/group)."""
    n_spectra = len(spectra_dict)

    if figsize is None:
        figsize = (12, 3 * n_spectra)

    fig, axes = plt.subplots(n_spectra, 6, figsize=figsize, sharey="row")
    if n_spectra == 1:
        axes = axes.reshape(1, -1)

    color_map = _resolve_palette(palette)
    colors = [color_map[t] for t in MUTATION_TYPES]

    if ylim is not None:
        y_min, y_max = ylim
    else:
        all_values = [s.values for s in spectra_dict.values()]
        if normalize:
            all_values = [v / v.sum() if v.sum() > 0 else v for v in all_values]
        max_value = max(v.max() for v in all_values)
        y_min = 0
        y_max = (max_value * 1.15) if max_value > 0 else 1.0

    # In multi-row mode we use the spectrum name as the row's ylabel; the
    # unit-label (e.g. "Mutation count") is only drawn when the caller
    # passes it explicitly, since the row label already identifies the axis.
    ylabel_explicit = ylabel is not None
    if ytick_unit_label is not None:
        ylabel = f"{ylabel or ''} ({ytick_unit_label})".lstrip()
        ylabel_explicit = True

    for row, (spectrum_name, spectrum) in enumerate(spectra_dict.items()):
        if normalize and spectrum.sum() > 0:
            spectrum = spectrum / spectrum.sum()

        for col, mut_type in enumerate(MUTATION_TYPES):
            ax = axes[row, col]

            contexts = [c for c in spectrum.index if f"{c[1]}>{c[5]}" == mut_type]
            counts = [spectrum[c] for c in contexts]

            if col == 0:
                x_pos = np.arange(len(contexts)) + left_pad
                ax.set_xlim(-0.5, len(contexts) - 0.5 + left_pad + group_padding)
            else:
                x_pos = np.arange(len(contexts))
                ax.set_xlim(-0.5 - group_padding, len(contexts) - 0.5 + group_padding)

            ax.bar(x_pos, counts, color=colors[col], width=bar_width, edgecolor="none")

            ax.set_ylim(y_min, y_max)
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

            # Per-panel mutation-type titles only on the top row.
            if show_mut_type_titles and row == 0:
                ax.set_title(mut_type, fontweight="bold", fontsize=_STYLE_DEFAULTS["panel_title_fontsize"])

            # X-ticks only on the bottom row so the ribbon reads vertically.
            if show_xticks and row == n_spectra - 1:
                ax.set_xticks(x_pos)
                ax.set_xticklabels(
                    [c[:3] for c in contexts],
                    rotation=90,
                    fontsize=_STYLE_DEFAULTS["xtick_fontsize"],
                    family="monospace",
                    ha="center",
                    va="top",
                )
                ax.tick_params(axis="x", length=0, pad=2)
            else:
                ax.set_xticks([])

            if col == 0:
                ax.tick_params(axis="y", labelsize=_STYLE_DEFAULTS["ytick_fontsize"])
                row_label = str(spectrum_name)
                if ylabel_explicit and ylabel:
                    row_label = f"{row_label}\n{ylabel}"
                ax.set_ylabel(
                    row_label,
                    fontsize=_STYLE_DEFAULTS["row_label_fontsize"],
                    fontweight="bold",
                )
                if yticks is not None:
                    ax.set_yticks(yticks)
                if ytick_scale != 1.0:
                    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _, s=ytick_scale: f"{v / s:g}"))
            else:
                ax.tick_params(axis="y", left=False, labelleft=False)
                ax.spines["left"].set_visible(False)

    # Compute dynamic y positions for suptitle / legend. Multi-row figures put
    # the top axis at a fractional y that depends on n_spectra; targeting
    # 1 + k/n keeps a constant physical gap above the top row regardless of
    # how many rows are stacked below it.
    extra_top = 0.10 if show_mut_type_titles else 0.04
    legend_y = 1 + (extra_top / n_spectra)
    suptitle_y = legend_y + (0.08 / n_spectra if show_legend else 0.0)
    suptitle_y = max(suptitle_y, 1 + (0.18 / n_spectra) if show_mut_type_titles else 1 + (0.08 / n_spectra))

    if show_legend:
        handles = [patches.Patch(facecolor=color_map[t], label=t) for t in MUTATION_TYPES]
        fig.legend(
            handles=handles,
            ncol=6,
            frameon=False,
            loc="lower center",
            bbox_to_anchor=(0.5, legend_y),
            fontsize=_STYLE_DEFAULTS["legend_fontsize"],
            handlelength=1.2,
            columnspacing=2.0,
            handletextpad=0.5,
        )

    if title:
        fig.suptitle(title, fontsize=_STYLE_DEFAULTS["suptitle_fontsize"], fontweight="bold", y=suptitle_y)

    plt.subplots_adjust(wspace=wspace)

    if outpath:
        plt.savefig(outpath, dpi=dpi, bbox_inches="tight", transparent=transparent)
        plt.close()
    else:
        plt.show()
