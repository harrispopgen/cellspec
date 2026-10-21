"""Trinucleotide context utilities."""

import re

import numpy as np
from Bio.Seq import Seq
from scipy.stats import hypergeom


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
    >>> reverse_complement("ATG")
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
    >>> strand_standardize_trinuc("TCG", "T")
    'TCG>TTG'
    >>> strand_standardize_trinuc("AGT", "C")  # A is purine, gets flipped to T
    'ACT>GCT'
    """
    if ref_trinuc[1] in "AG":
        trinuc_rc = reverse_complement(ref_trinuc)
        alt_rc = reverse_complement(alt_base)
        return f"{trinuc_rc}>{trinuc_rc[0]}{alt_rc}{trinuc_rc[2]}"
    else:
        return f"{ref_trinuc}>{ref_trinuc[0]}{alt_base}{ref_trinuc[2]}"


def parse_variant_id(variant_id: str, chrom_prefix: str | None = None) -> tuple[str, int, str, str] | None:
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
    >>> parse_variant_id("chr1-12345-A>T", chrom_prefix="chr")
    ('chr1', 12345, 'A', 'T')
    >>> parse_variant_id("I-12345-A>T", chrom_prefix=None)
    ('I', 12345, 'A', 'T')
    >>> parse_variant_id("CP116366.1-12345-A>T", chrom_prefix=None)
    ('CP116366.1', 12345, 'A', 'T')
    """
    # Build regex pattern based on chrom_prefix
    if chrom_prefix is not None:
        # Expect specific prefix (include dots for version numbers)
        pattern = rf"({chrom_prefix}[\w.]+)-(\d+)-([ACGT.]+)>([ACGT.]+)"
    else:
        # Accept any chromosome name (alphanumeric, dots for version numbers, including roman numerals)
        pattern = r"([\w.]+)-(\d+)-([ACGT.]+)>([ACGT.]+)"

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
    >>> classify_mutation_type("ACA>AAA")
    'C>A'
    """
    if len(mut_context) < 7:
        raise ValueError(f"Invalid mutation context: {mut_context}")

    ref_base = mut_context[1]
    alt_base = mut_context[-2]
    return f"{ref_base}>{alt_base}"


def compute_vaf(ad: np.ndarray, dp: np.ndarray, min_dp: int, normalize: bool = False) -> np.ndarray:
    """
    Compute Variant Allele Frequency with an explicit depth floor.

    Entries with ``dp < min_dp`` are returned as ``NaN`` — those sites don't
    have enough coverage to estimate VAF reliably. For entries with
    ``dp >= min_dp``:

    - ``normalize=False`` (default) returns the raw ``ad / dp``.
    - ``normalize=True`` hypergeometrically downsamples the alt-depth to
      ``min_dp`` and returns ``down_ad / min_dp``, making VAFs comparable
      across sites with heterogeneous coverage.

    Parameters
    ----------
    ad : np.ndarray
        Alternate allele depth (read counts supporting alternate allele).
    dp : np.ndarray
        Total depth (total read counts).
    min_dp : int
        Minimum DP floor. Sites with ``dp < min_dp`` return ``NaN``; when
        ``normalize=True``, this is also the target depth for downsampling.
    normalize : bool, default False
        Apply hypergeometric downsampling at ``min_dp`` for valid sites.

    Returns
    -------
    np.ndarray
        VAF per entry, with ``NaN`` for entries that failed the ``min_dp``
        threshold.

    Examples
    --------
    >>> ad = np.array([5, 20, 3])
    >>> dp = np.array([50, 100, 5])
    >>> compute_vaf(ad, dp, min_dp=10)  # raw VAF, last → NaN
    array([0.1 , 0.2 ,  nan], dtype=float32)
    >>> compute_vaf(ad, dp, min_dp=10, normalize=True)  # downsampled to 10
    """
    ad = np.asarray(ad, dtype=int)
    dp = np.asarray(dp, dtype=int)

    vaf = np.full(ad.shape, np.nan, dtype=np.float32)
    valid = dp >= min_dp
    if not np.any(valid):
        return vaf

    ad_v = ad[valid]
    dp_v = dp[valid]

    if normalize:
        down_ad = ad_v.copy()
        needs_down = dp_v > min_dp
        if needs_down.any():
            down_ad[needs_down] = hypergeom.rvs(dp_v[needs_down], ad_v[needs_down], min_dp)
        vaf[valid] = down_ad / min_dp
    else:
        vaf[valid] = ad_v / dp_v

    return vaf
