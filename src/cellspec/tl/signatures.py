"""Mutational signature analysis using SigProfiler."""

import os
import tempfile
import warnings

import anndata as ad
import pandas as pd

from cellspec.utils.sigprofiler_io import (
    cleanup_temp_files,
    parse_cosmic_assignment_results,
)


def fit_cosmic_signatures(
    adata: ad.AnnData,
    key: str = "spectrum",
    groupby: str | None = None,
    aggregate: str = "sum",
    cosmic_version: float = 3.5,
    genome_build: str = "GRCh37",
    output_key: str | None = None,
    signature_database: str | None = None,
    exclude_signature_subgroups: list[str] | None = None,
    make_plots: bool = False,
    context_type: str = "96",
    verbose: bool = True,
    inplace: bool = True,
) -> pd.DataFrame | None:
    """
    Fit known COSMIC mutational signatures to mutation spectra.

    Uses SigProfilerAssignment to decompose mutation spectra into
    contributions from known COSMIC signatures.

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with computed spectrum
    key : str, default 'spectrum'
        Key for spectrum in adata.obsm[f'spectrum_{key}']
    groupby : str, optional
        Column name from adata.obs to group samples by (e.g., 'lineage', 'sample_id').
        If provided, aggregates spectrum within each group before fitting signatures.
        Recommended for sparse single-cell data.
    aggregate : str, default 'sum'
        How to aggregate spectra within groups: 'sum' or 'mean'.
        Only used if groupby is provided.
    cosmic_version : float, default 3.5
        COSMIC signature version (1, 2, 3, 3.1, 3.2, 3.3, 3.4, 3.5)
    genome_build : str, default 'GRCh37'
        Reference genome ('GRCh37', 'GRCh38', 'mm9', 'mm10', 'c_elegans', etc.)
    output_key : str, optional
        Key for storing results. If None, uses f'{key}_cosmic'
    signature_database : str, optional
        Path to custom signature database (TSV file)
    exclude_signature_subgroups : list, optional
        Signature subgroups to exclude (e.g., ['MMR_deficiency_signatures'])
    make_plots : bool, default False
        Generate SigProfiler's default plots
    context_type : str, default '96'
        Mutation context type ('96', '288', '1536', 'DINUC', 'ID')
    verbose : bool, default True
        Print progress messages
    inplace : bool, default True
        Store results in adata

    Returns
    -------
    pd.DataFrame or None
        Signature activities (samples/groups × signatures) if inplace=False

    Stores in adata
    ---------------
    adata.obsm[f'cosmic_signatures_{output_key}'] : pd.DataFrame (if groupby=None)
        Signature activities/exposures per cell (cells × COSMIC_signatures)

    adata.uns[f'cosmic_signatures_{output_key}'] : pd.DataFrame (if groupby provided)
        Signature activities/exposures per group (groups × COSMIC_signatures)

    adata.uns[f'cosmic_signatures_{output_key}_reference'] : pd.DataFrame
        Reference COSMIC signature matrix (96 contexts × signatures)

    adata.uns[f'cosmic_signatures_{output_key}_metadata'] : dict
        Metadata including cosmic_version, genome_build, groupby, and fit quality

    Examples
    --------
    >>> import cellspec as spc
    >>> # Fit signatures to individual cells (if enough mutations per cell)
    >>> spc.tl.fit_cosmic_signatures(adata, key="somatic")
    >>> print(adata.obsm["cosmic_signatures_somatic_cosmic"])
    >>>
    >>> # Fit signatures to aggregated groups (recommended for sparse data)
    >>> spc.tl.fit_cosmic_signatures(adata, key="somatic", groupby="lineage")
    >>> print(adata.uns["cosmic_signatures_somatic_cosmic"])  # Results per lineage
    >>>
    >>> # Use C. elegans genome
    >>> spc.tl.fit_cosmic_signatures(adata, key="somatic", genome_build="c_elegans")
    >>>
    >>> # Aggregate by sample with mean instead of sum
    >>> spc.tl.fit_cosmic_signatures(adata, key="somatic", groupby="sample_id", aggregate="mean")
    >>>
    >>> # Use COSMIC v3.4 with custom exclusions
    >>> spc.tl.fit_cosmic_signatures(
    ...     adata, key="somatic", cosmic_version=3.4, exclude_signature_subgroups=["Possible_sequencing_artifacts"]
    ... )

    Notes
    -----
    SigProfilerAssignment works best with samples having >50 mutations per sample.

    **For sparse single-cell data (few mutations per cell):**
    Use the `groupby` parameter to aggregate mutations within groups:
    - `groupby='lineage'` - Fit signatures to lineage-level aggregates
    - `groupby='sample_id'` - Fit signatures per sample
    - `groupby='clone'` - Fit signatures per clone

    **Storage location:**
    - Without groupby: Results in adata.obsm (per-cell signature activities)
    - With groupby: Results in adata.uns (per-group signature activities)

    The function uses a forward stagewise algorithm with non-negative least
    squares (NNLS) to find the optimal combination of COSMIC signatures.
    """
    try:
        from SigProfilerAssignment import Analyzer as Analyze
    except ImportError as e:
        raise ImportError(
            "SigProfilerAssignment is required for signature fitting. Install with: pip install SigProfilerAssignment"
        ) from e

    if output_key is None:
        output_key = f"{key}_cosmic"

    if verbose:
        print(f"Fitting COSMIC v{cosmic_version} signatures to spectrum '{key}'...")

    # Get or create spectrum for fitting
    spectrum_df = adata.obsm[f"spectrum_{key}"]

    # Handle groupby aggregation
    if groupby is not None:
        if groupby not in adata.obs.columns:
            raise ValueError(
                f"groupby column '{groupby}' not found in adata.obs. Available columns: {list(adata.obs.columns)}"
            )

        if aggregate not in ["sum", "mean"]:
            raise ValueError(f"aggregate must be 'sum' or 'mean', got '{aggregate}'")

        if verbose:
            print(f"Aggregating spectrum by '{groupby}' using {aggregate}...")

        # Aggregate spectrum within groups
        grouped_spectra = []
        group_names = []
        for group_name, group_indices in adata.obs.groupby(groupby).groups.items():
            # Get cells in this group that are in spectrum_df
            group_cells = [cell for cell in group_indices if cell in spectrum_df.index]
            if len(group_cells) > 0:
                if aggregate == "sum":
                    group_spectrum = spectrum_df.loc[group_cells, :].sum(axis=0)
                else:  # mean
                    group_spectrum = spectrum_df.loc[group_cells, :].mean(axis=0)
                grouped_spectra.append(group_spectrum)
                group_names.append(str(group_name))

        # Create aggregated spectrum DataFrame
        spectrum_df = pd.DataFrame(grouped_spectra, index=group_names)

        if verbose:
            print(f"Created {len(group_names)} aggregated samples from {len(groupby)} groups")

    # Create temporary directories for I/O
    temp_input_dir = tempfile.mkdtemp(prefix="cellspec_sigprofiler_input_")
    temp_output_dir = tempfile.mkdtemp(prefix="cellspec_sigprofiler_output_")

    try:
        # Write spectrum to temporary file for SigProfiler
        input_matrix_path = os.path.join(temp_input_dir, "matrix.txt")
        from cellspec.utils.constants import get_canonical_96_order

        canonical_order = get_canonical_96_order()
        spectrum_df = spectrum_df[canonical_order].astype(int)
        spectrum_df.to_csv(input_matrix_path, sep="\t", index=True, index_label="Sample")

        if verbose:
            print(f"Converted spectrum to SigProfiler format: {input_matrix_path}")
            n_samples = spectrum_df.shape[0]
            total_mutations = spectrum_df.sum().sum()
            print(f"Analyzing {n_samples} samples with {int(total_mutations)} total mutations")

        # Check if samples have sufficient mutations
        sample_mutation_counts = spectrum_df.sum(axis=1)
        low_mutation_samples = (sample_mutation_counts < 50).sum()
        if low_mutation_samples > 0:
            warnings.warn(
                f"{low_mutation_samples}/{len(sample_mutation_counts)} samples have <50 mutations. "
                f"SigProfilerAssignment works best with >50 mutations per sample. "
                f"Consider aggregating sparse samples by lineage/clone.",
                UserWarning,
                stacklevel=2,
            )

        # Run SigProfilerAssignment
        if verbose:
            print("Running SigProfilerAssignment (this may take a few minutes)...")

        Analyze.cosmic_fit(
            samples=input_matrix_path,
            output=temp_output_dir,
            input_type="matrix",
            context_type=context_type,
            cosmic_version=cosmic_version,
            genome_build=genome_build,
            signature_database=signature_database,
            exclude_signature_subgroups=exclude_signature_subgroups,
            export_probabilities=False,
            make_plots=make_plots,
            sample_reconstruction_plots=False,  # Disable to reduce output
            verbose=verbose,
        )

        if verbose:
            print("Parsing results...")

        # Parse results
        sample_names = spectrum_df.index.tolist()
        activities_df, signatures_df, metadata = parse_cosmic_assignment_results(temp_output_dir, sample_names)

        if verbose:
            n_active_sigs = (activities_df.sum(axis=0) > 0).sum()
            print(f"Found {n_active_sigs} active signatures across {len(sample_names)} samples")

        # Store results in adata
        if inplace:
            # Store activities - location depends on whether we grouped
            if groupby is not None:
                # Grouped results go in .uns (not per-cell)
                adata.uns[f"cosmic_signatures_{output_key}"] = activities_df
                storage_location = f"adata.uns['cosmic_signatures_{output_key}']"
            else:
                # Per-cell results go in .obsm
                adata.obsm[f"cosmic_signatures_{output_key}"] = activities_df
                storage_location = f"adata.obsm['cosmic_signatures_{output_key}']"

            # Store reference signatures if available
            if signatures_df is not None:
                adata.uns[f"cosmic_signatures_{output_key}_reference"] = signatures_df

            # Store metadata
            adata.uns[f"cosmic_signatures_{output_key}_metadata"] = {
                "method": "SigProfilerAssignment",
                "cosmic_version": cosmic_version,
                "genome_build": genome_build,
                "context_type": context_type,
                "spectrum_key": key,
                "groupby": groupby,
                "aggregate": aggregate if groupby else None,
                "n_samples": len(sample_names),
                "n_signatures": activities_df.shape[1],
                "total_mutations": int(total_mutations),
                **metadata,
            }

            if verbose:
                print("Results stored in:")
                print(f"  - {storage_location}")
                print(f"  - adata.uns['cosmic_signatures_{output_key}_metadata']")
                if signatures_df is not None:
                    print(f"  - adata.uns['cosmic_signatures_{output_key}_reference']")

            return None
        else:
            return activities_df

    finally:
        # Clean up temporary files
        cleanup_temp_files(temp_input_dir)
        cleanup_temp_files(temp_output_dir)
