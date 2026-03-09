"""Helper functions for SigProfiler I/O and format conversion."""

import os
import tempfile
from pathlib import Path

import anndata as ad
import pandas as pd

from cellspec.utils.constants import get_canonical_96_order


def spectrum_to_sigprofiler_matrix(
    adata: ad.AnnData,
    key: str,
    output_path: str | None = None,
) -> str:
    """
    Convert cellspec spectrum to SigProfiler matrix format.

    SigProfiler expects a tab-delimited file with:
    - Rows: Sample IDs (cell names)
    - Columns: 96 trinucleotide contexts in canonical order
    - Values: Integer mutation counts

    Parameters
    ----------
    adata : ad.AnnData
        AnnData object with computed spectrum
    key : str
        Key for spectrum in adata.obsm[f'spectrum_{key}']
    output_path : str, optional
        Path to write matrix file. If None, creates temporary file.

    Returns
    -------
    str
        Path to the written matrix file

    Raises
    ------
    ValueError
        If spectrum key not found in adata.obsm
    """
    spectrum_key = f"spectrum_{key}"

    if spectrum_key not in adata.obsm:
        raise ValueError(f"'{spectrum_key}' not found in adata.obsm. Run spc.tl.compute_spectrum() first.")

    spectrum_df = adata.obsm[spectrum_key]

    # Ensure canonical column order
    canonical_order = get_canonical_96_order()

    # Check if all expected columns are present
    missing_cols = set(canonical_order) - set(spectrum_df.columns)
    if missing_cols:
        raise ValueError(
            f"Spectrum is missing expected 96-channel columns. "
            f"Missing: {sorted(list(missing_cols)[:5])}... "
            f"Ensure spectrum was computed with trinucleotide context annotation."
        )

    # Reorder columns to canonical order
    spectrum_df = spectrum_df[canonical_order]

    # Convert to integer counts (SigProfiler expects integers)
    spectrum_df = spectrum_df.astype(int)

    # Create output path if not provided
    if output_path is None:
        # Create temporary file
        temp_dir = tempfile.mkdtemp(prefix="cellspec_sigprofiler_")
        output_path = os.path.join(temp_dir, "matrix.txt")

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write tab-delimited file
    # SigProfiler expects sample names in first column with header
    spectrum_df.to_csv(output_path, sep="\t", index=True, index_label="Sample")

    return output_path


def parse_cosmic_assignment_results(
    results_dir: str,
    sample_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Parse SigProfilerAssignment cosmic_fit results.

    Parameters
    ----------
    results_dir : str
        Directory containing SigProfilerAssignment output
    sample_names : list of str
        Original sample names (for validation)

    Returns
    -------
    tuple of (activities_df, signatures_df, metadata)
        - activities_df: Signature activities per sample (samples × signatures)
        - signatures_df: COSMIC signature reference matrix (96 contexts × signatures)
        - metadata: Dict with fit quality metrics

    Raises
    ------
    FileNotFoundError
        If expected output files not found
    ValueError
        If results cannot be parsed
    """
    results_path = Path(results_dir)

    # SigProfilerAssignment output structure (as of v1.0+):
    # output/
    #   Assignment_Solution/
    #     Activities/Assignment_Solution_Activities.txt
    #     Signatures/Assignment_Solution_Signatures.txt
    #     Solution_Stats/Assignment_Solution_Samples_Stats.txt

    assignment_dir = results_path / "Assignment_Solution"
    activities_dir = assignment_dir / "Activities"
    signatures_dir = assignment_dir / "Signatures"
    stats_dir = assignment_dir / "Solution_Stats"

    # Check if Assignment_Solution exists
    if not assignment_dir.exists():
        raise FileNotFoundError(
            f"Could not find Assignment_Solution directory in {results_dir}. "
            f"Available subdirectories: {[d.name for d in results_path.iterdir() if d.is_dir()]}"
        )

    # Look for activities file
    activities_file = activities_dir / "Assignment_Solution_Activities.txt"
    if not activities_file.exists():
        # Fallback to searching
        activities_files = list(activities_dir.glob("*Activities*.txt")) if activities_dir.exists() else []
        if not activities_files:
            raise FileNotFoundError(
                f"Could not find Assignment_Solution_Activities.txt in {activities_dir}. "
                f"Available files: {[f.name for f in activities_dir.glob('*.txt')] if activities_dir.exists() else []}"
            )
        activities_file = activities_files[0]

    # Read activities (samples × signatures)
    activities_df = pd.read_csv(activities_file, sep="\t", index_col=0)

    # Validate sample names match
    if not set(activities_df.index).issubset(set(sample_names)):
        missing = set(activities_df.index) - set(sample_names)
        raise ValueError(
            f"Activities file contains unexpected sample names: {list(missing)[:5]}. Expected samples from input data."
        )

    # Look for signature reference file
    signatures_file = signatures_dir / "Assignment_Solution_Signatures.txt"
    signatures_df = None
    if signatures_file.exists():
        signatures_df = pd.read_csv(signatures_file, sep="\t", index_col=0)
    elif signatures_dir.exists():
        # Fallback to searching
        signatures_files = list(signatures_dir.glob("*Signatures*.txt"))
        if signatures_files:
            signatures_df = pd.read_csv(signatures_files[0], sep="\t", index_col=0)

    # Look for statistics/metadata
    metadata = {}
    stats_file = stats_dir / "Assignment_Solution_Samples_Stats.txt"
    if stats_file.exists():
        try:
            stats_df = pd.read_csv(stats_file, sep="\t", index_col=0)
            # Store reconstruction metrics if available
            if "Cosine Similarity" in stats_df.columns:
                metadata["reconstruction_cosine_similarity"] = stats_df["Cosine Similarity"].to_dict()
        except (FileNotFoundError, pd.errors.ParserError, KeyError, ValueError):
            pass  # Stats file format varies, skip if can't parse

    return activities_df, signatures_df, metadata


def cleanup_temp_files(file_path: str) -> None:
    """
    Clean up temporary files created for SigProfiler.

    Parameters
    ----------
    file_path : str
        Path to temporary file or directory to remove
    """
    path = Path(file_path)

    if path.exists():
        if path.is_file():
            path.unlink()
            # Try to remove parent if it's a temp directory
            parent = path.parent
            if parent.name.startswith("cellspec_sigprofiler_"):
                try:
                    parent.rmdir()
                except OSError:
                    pass  # Directory not empty, leave it
        elif path.is_dir():
            import shutil

            shutil.rmtree(path)
