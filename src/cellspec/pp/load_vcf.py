"""Load VCF files into AnnData objects."""

import anndata as ad
import numpy as np
from cyvcf2 import VCF
from scipy.sparse import csr_matrix
from tqdm import tqdm


def load_vcf(
    vcf_path: str, show_progress: bool = True, sparse: bool = True, skip_missing_alt: bool = True
) -> ad.AnnData:
    """
    Load a VCF file into an AnnData object.

    This function converts a joint-called VCF (bulk or single-cell) into an AnnData object
    with sparse matrices for memory efficiency. The resulting object has:
    - .X: Genotype calls (0=HOM_REF, 1=HET, 2=UNKNOWN, 3=HOM_ALT)
    - .layers['DP']: Total depth per variant per sample/cell
    - .layers['AD']: Alternate allele depth per variant per sample/cell
    - .var_names: Variant IDs in format 'chr-pos-ref>alt' (pos is 1-based, matching VCF)
    - .var['chrom']: Chromosome name for each variant
    - .var['pos']: Genomic position for each variant (1-based, matching VCF)
    - .obs_names: Sample/cell names from VCF

    Duplicate variants (same CHROM-POS-REF>ALT) are detected and only the first
    occurrence is kept, with a warning issued.

    Parameters
    ----------
    vcf_path : str
        Path to VCF file (can be .vcf or .vcf.gz)
    show_progress : bool, default True
        Show progress bar during loading
    sparse : bool, default True
        Use sparse matrices (recommended for single-cell scale data)
    skip_missing_alt : bool, default True
        Skip variants with missing ALT alleles (represented as "." in VCF).
        If False, these variants are included with "." as the ALT allele.

    Returns
    -------
    ad.AnnData
        AnnData object with variant calls

    Examples
    --------
    >>> import cellspec as spc
    >>> # Load VCF, skipping variants without ALT alleles (default)
    >>> adata = spc.pp.load_vcf("joint_calls.vcf.gz")
    >>> print(f"Loaded {adata.n_vars} variants across {adata.n_obs} samples/cells")
    >>>
    >>> # Include variants with missing ALT alleles
    >>> adata = spc.pp.load_vcf("joint_calls.vcf.gz", skip_missing_alt=False)

    Notes
    -----
    For large single-cell datasets (>100 cells), sparse=True is highly recommended
    to reduce memory usage. Most variants are not present in most cells, so the
    data is naturally sparse.

    Genotype encoding in .X:
    - 0: HOM_REF (0/0)
    - 1: HET (0/1)
    - 2: UNKNOWN (./.)
    - 3: HOM_ALT (1/1)

    Variant positions use 1-based indexing matching VCF format (POS field).

    By default, variants without ALT alleles (represented as "." in VCF) are
    skipped during loading. Set skip_missing_alt=False to include them with
    variant IDs like 'chr1-1000-A>.'

    If duplicate variant records are found (same CHROM-POS-REF>ALT), only the
    first occurrence is kept. This can happen when merging VCFs or from
    overlapping variant calls. Use bcftools norm -d exact to deduplicate VCFs.
    """
    # Load VCF
    vcf = VCF(vcf_path)

    # Get sample/cell names
    cell_names = np.array(vcf.samples)
    n_cells = len(cell_names)

    # Stream through VCF once, filtering and processing as we go
    variant_ids = []
    chroms = []
    positions = []
    depth_data = []
    alt_data = []
    genotype_data = []
    n_skipped = 0
    n_duplicates = 0
    seen_variants = set()

    # Use tqdm without total for indeterminate progress bar
    iterator = tqdm(vcf, desc="Loading VCF", unit=" sites") if show_progress else vcf

    for variant in iterator:
        # Check if ALT allele exists
        has_alt = variant.ALT and len(variant.ALT) > 0

        # Skip if missing ALT and user wants to skip these
        if skip_missing_alt and not has_alt:
            n_skipped += 1
            continue

        # Create variant ID using POS (1-based, matching VCF)
        alt_allele = variant.ALT[0] if has_alt else "."
        variant_id = f"{variant.CHROM}-{variant.POS}-{variant.REF}>{alt_allele}"

        # Check for duplicates (keep first occurrence)
        if variant_id in seen_variants:
            n_duplicates += 1
            continue

        seen_variants.add(variant_id)
        variant_ids.append(variant_id)
        chroms.append(variant.CHROM)
        positions.append(variant.POS)

        # Get depths
        # cyvcf2 uses various sentinel values for missing data:
        # INT32_MIN (-2147483648), -1, etc.
        # Replace all negative values with 0 for proper handling of missing depth data
        alt_dp = variant.gt_alt_depths
        ref_dp = variant.gt_ref_depths

        # Replace any negative sentinel values with 0
        alt_dp = np.where(alt_dp < 0, 0, alt_dp)
        ref_dp = np.where(ref_dp < 0, 0, ref_dp)

        tot_dp = np.add(alt_dp, ref_dp)

        # Get genotypes (0=HOM_REF, 1=HET, 2=UNKNOWN, 3=HOM_ALT)
        genotypes = variant.gt_types

        # Collect data
        depth_data.append(tot_dp)
        alt_data.append(alt_dp)
        genotype_data.append(genotypes.astype(np.float32))

    # Report filtering if variants were skipped
    if skip_missing_alt and n_skipped > 0 and show_progress:
        print(f"Skipped {n_skipped} sites with missing ALT alleles")

    # Warn about duplicates
    if n_duplicates > 0 and show_progress:
        import warnings

        warnings.warn(
            f"Found {n_duplicates} duplicate variant records in VCF (same CHROM-POS-REF>ALT). "
            f"Kept first occurrence of each. Consider deduplicating your VCF with: "
            f"bcftools norm -d exact input.vcf.gz -Oz -o output.vcf.gz",
            UserWarning,
            stacklevel=2,
        )

    variant_ids = np.array(variant_ids)
    n_variants = len(variant_ids)

    # Check if any variants were loaded
    if n_variants == 0:
        raise ValueError(
            f"No variants loaded from {vcf_path}. "
            f"{'All variants had missing ALT alleles. ' if skip_missing_alt else ''}"
            "Check your VCF file or set skip_missing_alt=False."
        )

    # Create AnnData object
    # Stack data (variants × cells) and transpose to (cells × variants)
    if show_progress:
        print(f"Building matrices for {n_variants} sites × {n_cells} samples...")
    genotype_mat = np.stack(genotype_data).T
    depth_mat = np.stack(depth_data).T
    alt_mat = np.stack(alt_data).T

    if sparse:
        # Convert to sparse matrices
        if show_progress:
            print("Converting to sparse format...")
        X = csr_matrix(genotype_mat, dtype=np.float32)
        adata = ad.AnnData(X)
        adata.layers["DP"] = csr_matrix(depth_mat, dtype=np.float32)
        adata.layers["AD"] = csr_matrix(alt_mat, dtype=np.float32)
    else:
        # Use dense arrays
        X = genotype_mat
        adata = ad.AnnData(X)
        adata.layers["DP"] = depth_mat
        adata.layers["AD"] = alt_mat

    # Set names
    adata.obs_names = cell_names
    adata.var_names = variant_ids

    # Add genomic position annotations
    adata.var["chrom"] = chroms
    adata.var["pos"] = positions

    # Add basic metadata
    adata.uns["vcf_source"] = vcf_path
    adata.uns["n_variants"] = n_variants
    adata.uns["n_cells"] = n_cells

    if show_progress:
        print(f"Done! Loaded {n_variants} sites × {n_cells} samples")

    return adata
