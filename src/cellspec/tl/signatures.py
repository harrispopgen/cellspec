"""Mutational signature analysis using SigProfiler."""

import os
import tempfile
import warnings

import anndata as ad
import pandas as pd

from cellspec.utils.sigprofiler_io import (
    cleanup_temp_files,
    parse_cosmic_assignment_results,
    spectrum_to_sigprofiler_matrix,
)


def fit_cosmic_signatures(
    adata: ad.AnnData,
    key: str = "spectrum",
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
        Signature activities (cells × signatures) if inplace=False

    Stores in adata
    ---------------
    adata.obsm[f'cosmic_signatures_{output_key}'] : pd.DataFrame
        Signature activities/exposures per cell (cells × COSMIC_signatures)

    adata.uns[f'cosmic_signatures_{output_key}_reference'] : pd.DataFrame
        Reference COSMIC signature matrix (96 contexts × signatures)

    adata.uns[f'cosmic_signatures_{output_key}_metadata'] : dict
        Metadata including cosmic_version, genome_build, and fit quality

    Examples
    --------
    >>> import cellspec as spc
    >>> # Basic usage - fit COSMIC v3.5 signatures
    >>> spc.tl.fit_cosmic_signatures(adata, key="somatic")
    >>> print(adata.obsm["cosmic_signatures_somatic_cosmic"])
    >>>
    >>> # Use C. elegans genome
    >>> spc.tl.fit_cosmic_signatures(adata, key="somatic", genome_build="c_elegans")
    >>>
    >>> # Use COSMIC v3.4 with custom exclusions
    >>> spc.tl.fit_cosmic_signatures(
    ...     adata, key="somatic", cosmic_version=3.4, exclude_signature_subgroups=["Possible_sequencing_artifacts"]
    ... )

    Notes
    -----
    SigProfilerAssignment works best with samples having >50 mutations.
    For sparse single-cell data, consider:
    1. Aggregating by lineage/clone using compute_spectrum(groupby='lineage')
    2. Using private_mutations() to analyze lineage-specific signatures

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

    # Create temporary directories for I/O
    temp_input_dir = tempfile.mkdtemp(prefix="cellspec_sigprofiler_input_")
    temp_output_dir = tempfile.mkdtemp(prefix="cellspec_sigprofiler_output_")

    try:
        # Convert spectrum to SigProfiler format
        input_matrix_path = os.path.join(temp_input_dir, "matrix.txt")
        spectrum_to_sigprofiler_matrix(adata, key, input_matrix_path)

        if verbose:
            print(f"Converted spectrum to SigProfiler format: {input_matrix_path}")
            n_samples = adata.obsm[f"spectrum_{key}"].shape[0]
            total_mutations = adata.obsm[f"spectrum_{key}"].sum().sum()
            print(f"Analyzing {n_samples} samples with {int(total_mutations)} total mutations")

        # Check if samples have sufficient mutations
        sample_mutation_counts = adata.obsm[f"spectrum_{key}"].sum(axis=1)
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
        sample_names = adata.obsm[f"spectrum_{key}"].index.tolist()
        activities_df, signatures_df, metadata = parse_cosmic_assignment_results(temp_output_dir, sample_names)

        if verbose:
            n_active_sigs = (activities_df.sum(axis=0) > 0).sum()
            print(f"Found {n_active_sigs} active signatures across {len(sample_names)} samples")

        # Store results in adata
        if inplace:
            # Store activities (cells × signatures)
            adata.obsm[f"cosmic_signatures_{output_key}"] = activities_df

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
                "n_samples": len(sample_names),
                "n_signatures": activities_df.shape[1],
                "total_mutations": int(total_mutations),
                **metadata,
            }

            if verbose:
                print("Results stored in:")
                print(f"  - adata.obsm['cosmic_signatures_{output_key}']")
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
