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


def _calculate_signature_proportions(exposures_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate proportions of each signature from absolute exposures.

    Parameters
    ----------
    exposures_df : pd.DataFrame
        Signature exposures (samples × signatures)

    Returns
    -------
    pd.DataFrame
        Signature proportions (samples × signatures), values sum to 1.0 per sample
    """
    # Calculate proportions by dividing each row by its sum
    proportions = exposures_df.div(exposures_df.sum(axis=1), axis=0)
    return proportions


def fit_cosmic_signatures(
    adata: ad.AnnData,
    key: str = "spectrum",
    groupby: str | None = None,
    aggregate: str = "sum",
    cosmic_version: float = 3.4,
    genome_build: str = "GRCh38",
    output_key: str | None = None,
    output_dir: str | None = None,
    signature_database: str | None = None,
    exclude_signature_subgroups: list[str] | None = None,
    make_plots: bool = True,
    sample_reconstruction_plots: bool = False,
    context_type: str = "96",
    exome: bool = False,
    nnls_add_penalty: float = 0.0,
    nnls_remove_penalty: float = 0.0,
    initial_remove_penalty: float = 0.0,
    cpu: int = 1,
    add_background_signatures: bool = True,
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
    cosmic_version : float, default 3.4
        COSMIC signature version (1, 2, 3, 3.1, 3.2, 3.3, 3.4, 3.5)
    genome_build : str, default 'GRCh38'
        Reference genome ('GRCh37', 'GRCh38', 'mm9', 'mm10', 'c_elegans', etc.)
    output_key : str, optional
        Key for storing results. If None, uses f'{key}_cosmic'
    output_dir : str, optional
        Directory to save SigProfilerAssignment outputs (plots, logs, etc.).
        If None, uses a temporary directory that is cleaned up after parsing results.
        If provided, results are preserved for inspection.
    signature_database : str, optional
        Path to custom signature database (TSV file)
    exclude_signature_subgroups : list, optional
        Signature subgroups to exclude (e.g., ['MMR_deficiency_signatures'])
    make_plots : bool, default True
        Generate SigProfiler's default plots
    sample_reconstruction_plots : bool, default False
        Generate per-sample reconstruction plots
    context_type : str, default '96'
        Mutation context type ('96', '288', '1536', 'DINUC', 'ID')
    exome : bool, default False
        Use exome-specific signature database
    nnls_add_penalty : float, default 0.0
        Penalty for adding new signatures (0.0 = no penalty)
    nnls_remove_penalty : float, default 0.0
        Penalty for removing signatures (0.0 = no penalty)
    initial_remove_penalty : float, default 0.0
        Initial penalty for signature removal
    cpu : int, default 1
        Number of CPUs to use
    add_background_signatures : bool, default True
        Include background signatures in fitting
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
    adata.obsm[f'{output_key}_cosmic_fit'] : pd.DataFrame (if groupby=None)
        Signature exposures per cell (cells × COSMIC_signatures)

    adata.obsm[f'{output_key}_cosmic_fit_proportions'] : pd.DataFrame (if groupby=None)
        Signature proportions per cell (cells × COSMIC_signatures)

    adata.uns[f'{output_key}_cosmic_fit'] : pd.DataFrame (if groupby provided)
        Signature exposures per group (groups × COSMIC_signatures)

    adata.uns[f'{output_key}_cosmic_fit_proportions'] : pd.DataFrame (if groupby provided)
        Signature proportions per group (groups × COSMIC_signatures)

    adata.uns['cosmic_signatures'] : pd.DataFrame
        Reference COSMIC signature matrix (signatures × 96 contexts)

    adata.uns[f'{output_key}_cosmic_fit_metadata'] : dict
        Metadata including cosmic_version, genome_build, groupby, and fit quality

    Examples
    --------
    >>> import cellspec as spc
    >>> # Fit signatures to individual cells (if enough mutations per cell)
    >>> spc.tl.fit_cosmic_signatures(adata, key="spectrum")
    >>> print(adata.obsm["spectrum_cosmic_fit"])  # Absolute counts
    >>> print(adata.obsm["spectrum_cosmic_fit_proportions"])  # Proportions
    >>>
    >>> # Fit signatures to aggregated groups (recommended for sparse data)
    >>> spc.tl.fit_cosmic_signatures(adata, key="spectrum", groupby="Mutation")
    >>> print(adata.uns["spectrum_Mutation_cosmic_fit"])  # Counts per Mutation group
    >>> print(adata.uns["spectrum_Mutation_cosmic_fit_proportions"])  # Proportions
    >>>
    >>> # Use C. elegans genome
    >>> spc.tl.fit_cosmic_signatures(adata, key="spectrum", genome_build="c_elegans")
    >>>
    >>> # Aggregate by sample with mean instead of sum
    >>> spc.tl.fit_cosmic_signatures(adata, key="spectrum", groupby="sample_id", aggregate="mean")
    >>>
    >>> # Use COSMIC v3.4 with custom exclusions
    >>> spc.tl.fit_cosmic_signatures(
    ...     adata, key="spectrum", cosmic_version=3.4, exclude_signature_subgroups=["Possible_sequencing_artifacts"]
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
        # Build output key: {key}_{groupby (if provided)}
        if groupby is not None:
            output_key = f"{key}_{groupby}"
        else:
            output_key = key

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
            n_unique_groups = adata.obs[groupby].nunique()
            print(f"Created {len(group_names)} aggregated samples from {n_unique_groups} unique groups")

    # Set up input/output directories
    if output_dir is None:
        # Use temporary directories that will be cleaned up
        temp_input_dir = tempfile.mkdtemp(prefix="cellspec_sigprofiler_input_")
        temp_output_dir = tempfile.mkdtemp(prefix="cellspec_sigprofiler_output_")
        cleanup_dirs = True
    else:
        # User-specified output directory - preserve results
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        temp_input_dir = str(output_path / "input")
        temp_output_dir = str(output_path / "output")
        Path(temp_input_dir).mkdir(parents=True, exist_ok=True)
        Path(temp_output_dir).mkdir(parents=True, exist_ok=True)
        cleanup_dirs = False

        if verbose:
            print(f"SigProfiler outputs will be saved to: {output_dir}")

    try:
        # Validate that spectrum has 96 columns
        if spectrum_df.shape[1] != 96:
            raise ValueError(
                f"Spectrum has {spectrum_df.shape[1]} columns, expected 96 trinucleotide contexts. "
                f"Ensure the spectrum was computed with trinucleotide context annotation "
                f"(spc.pp.annotate_contexts()) and filtering to SNPs (spc.pp.filter_to_snps())."
            )

        # Validate no NaN or negative values
        if spectrum_df.isna().any().any():
            raise ValueError("Spectrum contains NaN values. Ensure all mutations have valid trinucleotide contexts.")
        if (spectrum_df < 0).any().any():
            raise ValueError("Spectrum contains negative values. Mutation counts must be non-negative.")

        # Convert spectrum to SigProfiler format (transpose and convert mutation notation)
        # This creates a temporary AnnData-like structure with the spectrum
        input_matrix_path = os.path.join(temp_input_dir, "matrix.txt")
        # Create a temporary minimal AnnData with just the spectrum we need
        # (handles both grouped and ungrouped cases)
        import anndata as ad

        from cellspec.utils.sigprofiler_io import spectrum_to_sigprofiler_matrix

        temp_adata = ad.AnnData(obs=pd.DataFrame(index=spectrum_df.index))
        temp_adata.obsm[f"spectrum_{key}"] = spectrum_df

        # Write matrix in SigProfiler format (96 mutation types × N samples with A[C>A]A notation)
        spectrum_to_sigprofiler_matrix(temp_adata, key, output_path=input_matrix_path)

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
            exome=exome,
            export_probabilities=False,
            make_plots=make_plots,
            sample_reconstruction_plots=sample_reconstruction_plots,
            collapse_to_SBS96=False,  # We already provide SBS96 format
            nnls_add_penalty=nnls_add_penalty,
            nnls_remove_penalty=nnls_remove_penalty,
            initial_remove_penalty=initial_remove_penalty,
            cpu=cpu,
            add_background_signatures=add_background_signatures,
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

        # Calculate proportions from absolute exposures
        proportions_df = _calculate_signature_proportions(activities_df)

        # Store results in adata
        if inplace:
            # Store exposures and proportions - location depends on whether we grouped
            if groupby is not None:
                # Grouped results go in .uns (not per-cell)
                adata.uns[f"{output_key}_cosmic_fit"] = activities_df
                adata.uns[f"{output_key}_cosmic_fit_proportions"] = proportions_df
                storage_location = f"adata.uns['{output_key}_cosmic_fit']"
                proportions_location = f"adata.uns['{output_key}_cosmic_fit_proportions']"
            else:
                # Per-cell results go in .obsm
                adata.obsm[f"{output_key}_cosmic_fit"] = activities_df
                adata.obsm[f"{output_key}_cosmic_fit_proportions"] = proportions_df
                storage_location = f"adata.obsm['{output_key}_cosmic_fit']"
                proportions_location = f"adata.obsm['{output_key}_cosmic_fit_proportions']"

            # Store reference signatures if available (shared across all keys)
            if signatures_df is not None:
                adata.uns["cosmic_signatures"] = signatures_df

            # Store metadata
            adata.uns[f"{output_key}_cosmic_fit_metadata"] = {
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
                print(f"  - {proportions_location}")
                print(f"  - adata.uns['{output_key}_cosmic_fit_metadata']")
                if signatures_df is not None:
                    print("  - adata.uns['cosmic_signatures']")

            return None
        else:
            return activities_df

    finally:
        # Clean up temporary files only if not user-specified
        if cleanup_dirs:
            cleanup_temp_files(temp_input_dir)
            cleanup_temp_files(temp_output_dir)
