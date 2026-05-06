"""Filter variants and cells in AnnData objects."""

import anndata as ad
import numpy as np
from scipy.sparse import issparse

from cellspec.utils.context import parse_variant_id


def filter_variants(
    adata: ad.AnnData, min_cells: int | None = None, min_fraction: float | None = None, inplace: bool = True
) -> ad.AnnData | None:
    """
    Filter variants based on presence across cells/samples.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with variants
    min_cells : int, optional
        Minimum number of cells/samples a variant must be present in
    min_fraction : float, optional
        Minimum fraction of cells/samples a variant must be present in (0-1)
    inplace : bool, default True
        Modify adata in place or return copy

    Returns
    -------
    ad.AnnData or None
        Filtered AnnData (if inplace=False), otherwise None

    Examples
    --------
    >>> import cellspec as spc
    >>> # Keep variants present in at least 3 cells
    >>> spc.pp.filter_variants(adata, min_cells=3)

    >>> # Keep variants present in at least 1% of cells
    >>> spc.pp.filter_variants(adata, min_fraction=0.01)

    Notes
    -----
    Presence is determined by .X (binary presence/absence matrix).
    """
    if not inplace:
        adata = adata.copy()

    # Get presence matrix
    X = adata.X
    if issparse(X):
        presence_counts = np.array(X.sum(axis=0)).flatten()
    else:
        presence_counts = X.sum(axis=0)

    # Build mask
    mask = np.ones(adata.n_vars, dtype=bool)

    if min_cells is not None:
        mask &= presence_counts >= min_cells

    if min_fraction is not None:
        min_cells_from_fraction = int(np.ceil(adata.n_obs * min_fraction))
        mask &= presence_counts >= min_cells_from_fraction

    # Filter
    adata._inplace_subset_var(mask)

    if not inplace:
        return adata


def filter_cells(
    adata: ad.AnnData, min_variants: int | None = None, max_variants: int | None = None, inplace: bool = True
) -> ad.AnnData | None:
    """
    Filter cells/samples based on number of variants detected.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with variants
    min_variants : int, optional
        Minimum number of variants a cell must have
    max_variants : int, optional
        Maximum number of variants a cell must have
    inplace : bool, default True
        Modify adata in place or return copy

    Returns
    -------
    ad.AnnData or None
        Filtered AnnData (if inplace=False), otherwise None

    Examples
    --------
    >>> import cellspec as spc
    >>> # Keep cells with at least 10 variants
    >>> spc.pp.filter_cells(adata, min_variants=10)

    >>> # Remove cells with too many variants (potential doublets)
    >>> spc.pp.filter_cells(adata, min_variants=10, max_variants=1000)

    Notes
    -----
    Variant counts are from .X (binary presence/absence matrix).
    """
    if not inplace:
        adata = adata.copy()

    # Get variant counts per cell
    X = adata.X
    if issparse(X):
        variant_counts = np.array(X.sum(axis=1)).flatten()
    else:
        variant_counts = X.sum(axis=1)

    # Build mask
    mask = np.ones(adata.n_obs, dtype=bool)

    if min_variants is not None:
        mask &= variant_counts >= min_variants

    if max_variants is not None:
        mask &= variant_counts <= max_variants

    # Filter
    adata._inplace_subset_obs(mask)

    if not inplace:
        return adata


def filter_by_coverage(
    adata: ad.AnnData,
    min_depth: int,
    max_depth: int | None = None,
    min_fraction: float | None = None,
    min_cells: int | None = None,
    layer: str = "DP",
    min_alt_depth: int | None = None,
    max_alt_depth: int | None = None,
    inplace: bool = True,
) -> ad.AnnData | None:
    """
    Filter variants based on sequencing depth (coverage) across cells/samples.

    Keeps only variants where a minimum number or fraction of cells/samples meet
    the depth threshold.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with variants (must have 'DP' and/or 'AD' layer depending on mode)
    min_depth : int
        Minimum depth threshold per cell. Interpretation depends on `layer`:
        - layer='DP': Minimum total depth (reference + alternate reads)
        - layer='AD': Minimum alternate allele depth
        - layer='both': Minimum total depth (use with min_alt_depth)
    max_depth : int, optional
        Maximum depth threshold per cell. Interpretation depends on `layer`:
        - layer='DP': Maximum total depth (reference + alternate reads)
        - layer='AD': Maximum alternate allele depth
        - layer='both': Maximum total depth (use with max_alt_depth)
        If None, no upper limit is applied. Useful for filtering potential
        PCR duplicates, mapping artifacts, or high-coverage outliers.
    min_fraction : float, optional
        Minimum fraction of cells that must meet the depth threshold (0-1).
        Cannot be used together with min_cells. If neither min_fraction nor
        min_cells is specified, defaults based on layer:
        - layer='DP': defaults to 1.0 (ALL cells, for quality control)
        - layer='AD' or 'both': defaults to min_cells=1 (ANY cell with variant)
    min_cells : int, optional
        Minimum number of cells that must meet the depth threshold.
        Cannot be used together with min_fraction. Useful when you want
        "at least N cells" regardless of total cell count.
    layer : str, default 'DP'
        Which layer(s) to use for filtering:
        - 'DP': Filter by total depth only (default, backward compatible)
        - 'AD': Filter by alternate allele depth only
        - 'both': Require both DP and AD thresholds
    min_alt_depth : int, optional
        When layer='both', the minimum alternate allele depth required.
        If None, defaults to min_depth. Ignored when layer='DP' or layer='AD'.
    max_alt_depth : int, optional
        When layer='both', the maximum alternate allele depth allowed.
        If None, no upper limit is applied. Ignored when layer='DP' or layer='AD'.
    inplace : bool, default True
        Modify adata in place or return copy

    Returns
    -------
    ad.AnnData or None
        Filtered AnnData (if inplace=False), otherwise None

    Examples
    --------
    >>> import cellspec as spc
    >>> # Keep only variants with total depth >= 10 in ALL cells (default for DP)
    >>> spc.pp.filter_by_coverage(adata, min_depth=10)
    >>>
    >>> # Filter by depth range: 10 <= DP <= 100 in ALL cells
    >>> spc.pp.filter_by_coverage(adata, min_depth=10, max_depth=100)
    >>>
    >>> # Keep variants with DP >= 10 in at least 80% of cells
    >>> spc.pp.filter_by_coverage(adata, min_depth=10, min_fraction=0.8)
    >>>
    >>> # Keep variants with DP >= 10 in at least 5 cells
    >>> spc.pp.filter_by_coverage(adata, min_depth=10, min_cells=5)
    >>>
    >>> # Filter by alternate allele depth - keep if ANY cell has AD >= 3 (default for AD)
    >>> spc.pp.filter_by_coverage(adata, min_depth=3, layer="AD")
    >>>
    >>> # Filter by AD range: 3 <= AD <= 50 in at least 1 cell
    >>> spc.pp.filter_by_coverage(adata, min_depth=3, max_depth=50, layer="AD")
    >>>
    >>> # Filter by AD - require at least 2 cells with AD >= 3
    >>> spc.pp.filter_by_coverage(adata, min_depth=3, min_cells=2, layer="AD")
    >>>
    >>> # Require both DP >= 10 AND AD >= 3 in at least 1 cell (default for 'both')
    >>> spc.pp.filter_by_coverage(adata, min_depth=10, min_alt_depth=3, layer="both")
    >>>
    >>> # Complex filter: 10 <= DP <= 100 AND 3 <= AD <= 50 in at least 90% of cells
    >>> spc.pp.filter_by_coverage(
    ...     adata, min_depth=10, max_depth=100, min_alt_depth=3, max_alt_depth=50, layer="both", min_fraction=0.9
    ... )

    Notes
    -----
    This function is useful for ensuring high-quality variant calls by removing
    sites with insufficient or excessive coverage.

    **Default behavior by layer:**

    - **layer='DP'**: Defaults to min_fraction=1.0 (ALL cells). This is stringent
      and ensures every cell has adequate total depth at retained variants, which
      is important for quality control.

    - **layer='AD' or 'both'**: Defaults to min_cells=1 (ANY cell). This is
      appropriate for variant detection, as you typically want to retain variants
      that appear in at least one cell with sufficient alternate allele support.

    **Using max_depth:**

    Setting a maximum depth threshold helps filter out problematic sites with
    abnormally high coverage, which may indicate:
    - PCR duplicates
    - Mapping artifacts (e.g., reads mapping to repetitive regions)
    - Copy number variations or collapsed repeats
    - Sequencing errors or technical artifacts

    For example, if median coverage is ~30x, sites with >100x coverage may be
    problematic. Use max_depth to exclude these outliers.

    **Filtering modes:**

    - **DP mode**: Filters based on total sequencing depth. Useful for ensuring
      adequate coverage regardless of genotype.

    - **AD mode**: Filters based on alternate allele depth only. Useful when you
      want to retain variants with sufficient evidence for the variant allele in
      at least some cells. Default behavior (min_cells=1) keeps variants present
      in any cell.

    - **both mode**: Applies both filters simultaneously. Useful when you need
      both adequate total coverage AND sufficient alternate allele support.
      For example, requiring DP >= 10 and AD >= 3 ensures good coverage overall
      while also requiring at least 3 reads supporting the variant.

    **Choosing min_cells vs min_fraction:**

    - Use **min_cells** when you want "at least N cells" regardless of dataset size
      (e.g., "at least 3 cells" works the same for 10-cell or 1000-cell datasets)

    - Use **min_fraction** when you want a proportion (e.g., "at least 10% of cells"
      scales with dataset size)

    Requires the 'DP' layer (and 'AD' layer if using layer='AD' or layer='both')
    in adata.layers, which are automatically created by spc.pp.load_vcf() when
    loading VCF files.
    """
    if not inplace:
        adata = adata.copy()

    # Validate layer parameter
    valid_layers = {"DP", "AD", "both"}
    if layer not in valid_layers:
        raise ValueError(f"layer must be one of {valid_layers}, got '{layer}'")

    # Validate min_fraction and min_cells parameters
    if min_fraction is not None and min_cells is not None:
        raise ValueError("Cannot specify both min_fraction and min_cells. Use one or the other.")

    # Set defaults based on layer if neither specified
    if min_fraction is None and min_cells is None:
        if layer == "DP":
            # For DP filtering, default to requiring all cells (quality control)
            min_fraction = 1.0
        else:
            # For AD or both, default to requiring at least 1 cell (variant detection)
            min_cells = 1

    # Check for required layers
    if layer in ["DP", "both"]:
        if "DP" not in adata.layers:
            raise ValueError(
                "'DP' layer not found in adata.layers. "
                "This layer is required for coverage filtering and is created by spc.pp.load_vcf()."
            )

    if layer in ["AD", "both"]:
        if "AD" not in adata.layers:
            raise ValueError(
                "'AD' layer not found in adata.layers. "
                "This layer is required for alternate allele depth filtering and is created by spc.pp.load_vcf()."
            )

    # Calculate minimum number of cells required
    if min_fraction is not None:
        min_cells_required = int(np.ceil(adata.n_obs * min_fraction))
    else:
        min_cells_required = min_cells

    # Filter based on selected layer
    if layer == "DP":
        # Filter by total depth
        DP = adata.layers["DP"]
        if issparse(DP):
            DP = DP.toarray()

        # Apply both min and max thresholds
        meets_threshold = DP >= min_depth
        if max_depth is not None:
            meets_threshold = meets_threshold & (DP <= max_depth)

        n_cells_passing = meets_threshold.sum(axis=0)
        mask = n_cells_passing >= min_cells_required

    elif layer == "AD":
        # Filter by alternate allele depth
        AD = adata.layers["AD"]
        if issparse(AD):
            AD = AD.toarray()

        # Apply both min and max thresholds
        meets_threshold = AD >= min_depth
        if max_depth is not None:
            meets_threshold = meets_threshold & (AD <= max_depth)

        n_cells_passing = meets_threshold.sum(axis=0)
        mask = n_cells_passing >= min_cells_required

    elif layer == "both":
        # Filter by both DP and AD
        DP = adata.layers["DP"]
        AD = adata.layers["AD"]

        if issparse(DP):
            DP = DP.toarray()
        if issparse(AD):
            AD = AD.toarray()

        # Set thresholds (use min_depth/max_depth for AD if specific thresholds not provided)
        ad_min_threshold = min_alt_depth if min_alt_depth is not None else min_depth
        ad_max_threshold = max_alt_depth if max_alt_depth is not None else max_depth

        # Apply DP thresholds
        meets_dp_threshold = DP >= min_depth
        if max_depth is not None:
            meets_dp_threshold = meets_dp_threshold & (DP <= max_depth)

        # Apply AD thresholds
        meets_ad_threshold = AD >= ad_min_threshold
        if ad_max_threshold is not None:
            meets_ad_threshold = meets_ad_threshold & (AD <= ad_max_threshold)

        # Both conditions must be met
        meets_both = meets_dp_threshold & meets_ad_threshold

        n_cells_passing = meets_both.sum(axis=0)
        mask = n_cells_passing >= min_cells_required

    # Filter
    adata._inplace_subset_var(mask)

    if not inplace:
        return adata


def filter_to_snps(adata: ad.AnnData, chrom_prefix: str | None = None, inplace: bool = True) -> ad.AnnData | None:
    """
    Filter to only single nucleotide variants (SNPs).

    Removes indels and multi-allelic sites, keeping only sites where
    both reference and alternate alleles are single nucleotides.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with variants
    chrom_prefix : str, optional
        Chromosome prefix for parsing variant IDs. If None (default), accepts
        any chromosome naming (e.g., 'chr1', 'I', '1'). Use 'chr' for
        human/mouse data if you want to be strict.
    inplace : bool, default True
        Modify adata in place or return copy

    Returns
    -------
    ad.AnnData or None
        Filtered AnnData (if inplace=False), otherwise None

    Examples
    --------
    >>> import cellspec as spc
    >>> # Default: works with any chromosome naming
    >>> adata = spc.pp.load_vcf("variants.vcf.gz")
    >>> spc.pp.filter_to_snps(adata)
    >>> print(f"Retained {adata.n_vars} SNPs")
    >>>
    >>> # For human/mouse data, can specify prefix
    >>> spc.pp.filter_to_snps(adata, chrom_prefix="chr")

    Notes
    -----
    This is typically run before trinucleotide context annotation,
    as trinuc contexts are only defined for SNPs.

    Works with any organism's chromosome naming:
    - Human/mouse: 'chr1-12345-A>T'
    - C. elegans: 'I-12345-A>T'
    - Drosophila: '2L-12345-A>T'
    """
    if not inplace:
        adata = adata.copy()

    # Check each variant
    is_snp = []
    for variant_id in adata.var_names:
        parsed = parse_variant_id(variant_id, chrom_prefix)
        if parsed is None:
            is_snp.append(False)
        else:
            chrom, pos, ref_base, alt_base = parsed
            # SNP: both ref and alt are single nucleotides (A, C, G, or T)
            # Exclude variants with missing ALT (represented as ".")
            is_valid_base = lambda b: len(b) == 1 and b in "ACGT"
            is_snp.append(is_valid_base(ref_base) and is_valid_base(alt_base))

    is_snp = np.array(is_snp)

    # Filter
    adata._inplace_subset_var(is_snp)

    if not inplace:
        return adata
