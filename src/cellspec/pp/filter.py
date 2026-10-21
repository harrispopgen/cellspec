"""Preprocessing-stage filters.

Only ``filter_to_snps`` lives in ``pp`` — it is required before trinucleotide
context annotation. Depth- and carrier-based filtering lives in ``tl`` and
depends on precomputed stats; see :func:`cellspec.tl.filter_sites` and
:func:`cellspec.tl.filter_cells`.
"""

import anndata as ad
import numpy as np

from cellspec.utils.context import parse_variant_id


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

    is_snp = []
    for variant_id in adata.var_names:
        parsed = parse_variant_id(variant_id, chrom_prefix)
        if parsed is None:
            is_snp.append(False)
            continue
        _, _, ref_base, alt_base = parsed
        is_snp.append(len(ref_base) == 1 and ref_base in "ACGT" and len(alt_base) == 1 and alt_base in "ACGT")

    adata._inplace_subset_var(np.array(is_snp))

    if not inplace:
        return adata
