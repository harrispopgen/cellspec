"""Helper functions for SigProfiler I/O and format conversion."""

import os
import tempfile
from pathlib import Path

import anndata as ad
import pandas as pd

from cellspec.utils.constants import get_canonical_96_order


def convert_to_sigprofiler_format(mutation_type: str) -> str:
    """
    Convert cellspec mutation format to SigProfiler format.

    Converts from 'XYZ>XWZ' to 'X[Y>W]Z' format.
    Example: 'ACA>AAA' -> 'A[C>A]A'

    Parameters
    ----------
    mutation_type : str
        Mutation in cellspec format (e.g., 'ACA>AAA')

    Returns
    -------
    str
        Mutation in SigProfiler format (e.g., 'A[C>A]A')
    """
    # Format is XYZ>XWZ where Y is mutated base
    parts = mutation_type.split(">")
    if len(parts) != 2 or len(parts[0]) != 3 or len(parts[1]) != 3:
        raise ValueError(f"Invalid mutation type format: {mutation_type}")

    left = parts[0]  # XYZ
    right = parts[1]  # XWZ

    # Extract: X[Y>W]Z
    return f"{left[0]}[{left[1]}>{right[1]}]{left[2]}"


def convert_from_sigprofiler_format(mutation_type: str) -> str:
    """
    Convert SigProfiler mutation format to cellspec format.

    Converts from 'X[Y>W]Z' to 'XYZ>XWZ' format.
    Example: 'A[C>A]A' -> 'ACA>AAA'

    Parameters
    ----------
    mutation_type : str
        Mutation in SigProfiler format (e.g., 'A[C>A]A')

    Returns
    -------
    str
        Mutation in cellspec format (e.g., 'ACA>AAA')
    """
    # Format is X[Y>W]Z
    if "[" not in mutation_type or "]" not in mutation_type:
        raise ValueError(f"Invalid SigProfiler format: {mutation_type}")

    base_5 = mutation_type[0]
    base_3 = mutation_type[-1]
    change = mutation_type[mutation_type.index("[") + 1 : mutation_type.index("]")]
    ref, alt = change.split(">")

    return f"{base_5}{ref}{base_3}>{base_5}{alt}{base_3}"


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

    # TRANSPOSE: SigProfiler expects (96 mutation types × N samples), not (N samples × 96 contexts)
    spectrum_df = spectrum_df.T

    # Convert row names (mutation types) from cellspec format (ACA>AAA) to SigProfiler format (A[C>A]A)
    sigprofiler_index = [convert_to_sigprofiler_format(idx) for idx in spectrum_df.index]
    spectrum_df.index = sigprofiler_index

    # Create output path if not provided
    if output_path is None:
        # Create temporary file
        temp_dir = tempfile.mkdtemp(prefix="cellspec_sigprofiler_")
        output_path = os.path.join(temp_dir, "matrix.txt")

    # Ensure parent directory exists
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)

    # Write tab-delimited file
    # Format: mutation types as rows, samples as columns
    # Index has no name (just empty string in header)
    spectrum_df.to_csv(output_path, sep="\t", index=True)

    return output_path


def parse_cosmic_assignment_results(
    results_dir: str,
    sample_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """
    Parse SigProfilerAssignment cosmic_fit results.

    SigProfilerAssignment output structure:
    - Activities: Contains fitted COSMIC signature profiles (signatures × 96 contexts)
    - Signatures: Contains reconstructed sample spectra (samples × 96 contexts)
    - Solution_Stats: Contains fit quality metrics per sample

    Parameters
    ----------
    results_dir : str
        Directory containing SigProfilerAssignment output
    sample_names : list of str
        Original sample names (for validation)

    Returns
    -------
    tuple of (activities_df, cosmic_signatures_df, metadata)
        - activities_df: Reconstructed sample spectra (samples × 96 contexts) in cellspec format
        - cosmic_signatures_df: COSMIC signatures found (signatures × 96 contexts) in cellspec format
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

    # Read Signatures file (contains reconstructed sample spectra: samples × 96 contexts)
    signatures_file = signatures_dir / "Assignment_Solution_Signatures.txt"
    if not signatures_file.exists():
        # Fallback to searching
        signatures_files = list(signatures_dir.glob("*Signatures*.txt")) if signatures_dir.exists() else []
        if not signatures_files:
            raise FileNotFoundError(
                f"Could not find Assignment_Solution_Signatures.txt in {signatures_dir}. "
                f"Available files: {[f.name for f in signatures_dir.glob('*.txt')] if signatures_dir.exists() else []}"
            )
        signatures_file = signatures_files[0]

    # Signatures file format: (96 mutation types × N samples)
    reconstructed_spectra = pd.read_csv(signatures_file, sep="\t", index_col=0)

    # Convert row names (mutation types) from SigProfiler format (A[C>A]A) back to cellspec format (ACA>AAA)
    cellspec_index = [convert_from_sigprofiler_format(idx) for idx in reconstructed_spectra.index]
    reconstructed_spectra.index = cellspec_index

    # TRANSPOSE to get (N samples × 96 contexts) format that cellspec uses
    reconstructed_spectra = reconstructed_spectra.T

    # Validate sample names
    if not set(reconstructed_spectra.index).issubset(set(sample_names)):
        missing = set(reconstructed_spectra.index) - set(sample_names)
        raise ValueError(
            f"Signatures file contains unexpected sample names: {list(missing)[:5]}. Expected: {sample_names}"
        )

    # Read Activities file (contains COSMIC signature profiles: 96 mutation types × N signatures)
    activities_file = activities_dir / "Assignment_Solution_Activities.txt"
    cosmic_signatures_df = None
    if activities_file.exists():
        cosmic_signatures_df = pd.read_csv(activities_file, sep="\t", index_col=0)
        # Convert row names (mutation types) from SigProfiler format back to cellspec format
        cellspec_index = [convert_from_sigprofiler_format(idx) for idx in cosmic_signatures_df.index]
        cosmic_signatures_df.index = cellspec_index
        # Transpose to get (N signatures × 96 contexts) format
        cosmic_signatures_df = cosmic_signatures_df.T

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

    return reconstructed_spectra, cosmic_signatures_df, metadata


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
