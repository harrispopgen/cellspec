"""Trinucleotide context utilities."""

import re
import numpy as np
from Bio.Seq import Seq
from scipy.stats import hypergeom
from typing import Optional, Tuple


def reverse_complement(seq: str) -> str:
    """
    Return the reverse complement of a DNA sequence.

    Parameters
    ----------
    seq : str
        DNA sequence

    Returns
    -------
    str
        Reverse complement of input sequence

    Examples
    --------
    >>> reverse_complement('ATG')
    'CAT'
    """
    return str(Seq(seq).reverse_complement())


def strand_standardize_trinuc(ref_trinuc: str, alt_base: str) -> str:
    """
    Convert a trinucleotide substitution into pyrimidine-based form (per COSMIC convention).

    If the central base is a purine (A/G), reverse complements both the trinucleotide and alt.
    This standardizes mutations to always report pyrimidine (C/T) reference alleles.

    Parameters
    ----------
    ref_trinuc : str
        Reference trinucleotide context (e.g., 'TCG')
    alt_base : str
        Alternate allele base (e.g., 'T')

    Returns
    -------
    str
        Standardized 3mer substitution string (e.g., 'TCG>TTG')

    Examples
    --------
    >>> strand_standardize_trinuc('TCG', 'T')
    'TCG>TTG'
    >>> strand_standardize_trinuc('AGT', 'C')  # A is purine, gets flipped to T
    'ACT>GCT'
    """
    if ref_trinuc[1] in "AG":
        trinuc_rc = reverse_complement(ref_trinuc)
        alt_rc = reverse_complement(alt_base)
        return f"{trinuc_rc}>{trinuc_rc[0]}{alt_rc}{trinuc_rc[2]}"
    else:
        return f"{ref_trinuc}>{ref_trinuc[0]}{alt_base}{ref_trinuc[2]}"


def parse_variant_id(variant_id: str, chrom_prefix: Optional[str] = None) -> Optional[Tuple[str, int, str, str]]:
    """
    Parse a variant ID string into its components.

    Parameters
    ----------
    variant_id : str
        Variant ID in format 'chr1-12345-A>T' or 'I-12345-A>T'
    chrom_prefix : str, optional
        Expected chromosome prefix. If None, accepts any chromosome naming.
        Use 'chr' for human/mouse, or None for organisms like C. elegans.

    Returns
    -------
    tuple or None
        (chrom, pos, ref, alt) if valid, None otherwise

    Examples
    --------
    >>> parse_variant_id('chr1-12345-A>T', chrom_prefix='chr')
    ('chr1', 12345, 'A', 'T')
    >>> parse_variant_id('I-12345-A>T', chrom_prefix=None)
    ('I', 12345, 'A', 'T')
    """
    # Build regex pattern based on chrom_prefix
    if chrom_prefix is not None:
        # Expect specific prefix
        pattern = rf"({chrom_prefix}[\w]+)-(\d+)-([ACGT.]+)>([ACGT.]+)"
    else:
        # Accept any chromosome name (alphanumeric, including roman numerals)
        pattern = r"([\w]+)-(\d+)-([ACGT.]+)>([ACGT.]+)"

    match = re.match(pattern, variant_id)
    if not match:
        return None
    chrom, pos, ref_base, alt_base = match.groups()
    return chrom, int(pos), ref_base, alt_base


def classify_mutation_type(mut_context: str) -> str:
    """
    Classify a mutation context into one of the 6 major substitution types.

    Parameters
    ----------
    mut_context : str
        Mutation context (e.g., 'ACA>AAA')

    Returns
    -------
    str
        Mutation type: 'C>A', 'C>G', 'C>T', 'T>A', 'T>C', or 'T>G'

    Examples
    --------
    >>> classify_mutation_type('ACA>AAA')
    'C>A'
    """
    if len(mut_context) < 7:
        raise ValueError(f"Invalid mutation context: {mut_context}")

    ref_base = mut_context[1]
    alt_base = mut_context[-2]
    return f"{ref_base}>{alt_base}"


def compute_vaf(ad: np.ndarray, dp: np.ndarray, target_dp: int = 10) -> np.ndarray:
    """
    Compute Variant Allele Frequency with hypergeometric downsampling for coverage normalization.

    This function normalizes VAF calculations across sites with unequal coverage by
    downsampling high-coverage sites to a target depth using hypergeometric sampling.

    Parameters
    ----------
    ad : np.ndarray
        Alternate allele depth (read counts supporting alternate allele)
    dp : np.ndarray
        Total depth (total read counts)
    target_dp : int, default 10
        Target depth for downsampling

    Returns
    -------
    np.ndarray
        Variant allele frequencies (VAF) normalized to target depth

    Examples
    --------
    >>> ad = np.array([5, 20, 3])
    >>> dp = np.array([50, 100, 30])
    >>> vaf = compute_vaf(ad, dp, target_dp=10)
    """
    ad = ad.astype(int)
    dp = dp.astype(int)

    # Create mask for rows that need downsampling
    needs_downsampling = dp > target_dp

    # Initialize with original values
    downsampled_ad = ad.copy()

    # Only apply hypergeometric to rows that need it
    if needs_downsampling.any():
        downsampled_ad[needs_downsampling] = hypergeom.rvs(
            dp[needs_downsampling],
            ad[needs_downsampling],
            target_dp
        )

    downsampled_dp = np.minimum(dp, target_dp)
    return downsampled_ad / downsampled_dp
