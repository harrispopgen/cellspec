"""Load VCF files into AnnData objects."""

import anndata as ad
import numpy as np
from cyvcf2 import VCF
from scipy.sparse import coo_matrix, csr_matrix, vstack
from tqdm import tqdm

# Variants per CSR chunk during sparse build. Bounds peak buffer memory at
# roughly chunk_size * n_cells * 8 bytes for the COO triples before each
# flush; tune up for higher density data, down for very wide cohorts.
_CHUNK_SIZE = 50_000


def load_vcf(
    vcf_path: str, show_progress: bool = True, sparse: bool = True, skip_missing_alt: bool = True
) -> ad.AnnData:
    """
    Load a VCF file into an AnnData object.

    This function converts a joint-called VCF (bulk or single-cell) into an AnnData object
    with sparse matrices for memory efficiency. The resulting object has:
    - .X: Genotype calls (0=HOM_REF or missing, 1=HET, 3=HOM_ALT)
    - .layers['DP']: Total depth per variant per sample/cell
    - .layers['AD']: Alternate allele depth per variant per sample/cell
    - .var_names: Variant IDs in format 'chr-pos-ref>alt' (pos is 1-based, matching VCF)
    - .var['chrom']: Chromosome name for each variant
    - .var['pos']: Genomic position for each variant (1-based, matching VCF)
    - .obs_names: Sample/cell names from VCF

    The input VCF **must have one ALT allele per record** — multi-allelic
    records raise ``ValueError``. Preprocess with
    ``bcftools norm -m-any input.vcf.gz | bcftools norm -d exact``
    to split multi-allelics and drop any duplicates the split produces.

    Duplicate variants (same CHROM-POS-REF>ALT) are detected and only the first
    occurrence is kept, with a warning issued.

    Parameters
    ----------
    vcf_path : str
        Path to VCF file (can be .vcf or .vcf.gz)
    show_progress : bool, default True
        Show progress bar during loading
    sparse : bool, default True
        Use sparse matrices (recommended for single-cell scale data). When True,
        matrices are built incrementally in CSR chunks so peak memory scales with
        the number of nonzero entries rather than (n_cells × n_variants).
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
    For large single-cell datasets, sparse=True (the default) is required at
    realistic joint-VCF scale. Memory usage is then O(nnz) — proportional to
    the actual number of called (cell, variant) entries rather than the dense
    matrix size — making the in-memory footprint comparable to (and often
    smaller than) the source VCF.gz on disk.

    Genotype encoding in .X:
    - 0: HOM_REF *or* missing (use ``layers['DP'] > 0`` to distinguish)
    - 1: HET (0/1)
    - 3: HOM_ALT (1/1)

    UNKNOWN VCF calls (``./.``) and sites with no read coverage are treated as
    missing: their entries in .X, .layers['DP'], and .layers['AD'] are all 0.
    To identify the cells/sites where a real call was made, mask on
    ``adata.layers['DP'] > 0``.

    Variant positions use 1-based indexing matching VCF format (POS field).

    By default, variants without ALT alleles (represented as "." in VCF) are
    skipped during loading. Set skip_missing_alt=False to include them with
    variant IDs like 'chr1-1000-A>.'

    If duplicate variant records are found (same CHROM-POS-REF>ALT), only the
    first occurrence is kept. This can happen when merging VCFs or from
    overlapping variant calls. Use bcftools norm -d exact to deduplicate VCFs.

    **Multi-allelic sites.** cellspec stores one ALT per variant and uses
    cyvcf2's per-sample AD / DP, which are only well-defined when each VCF
    record carries a single ALT. Multi-allelic records raise a ``ValueError``
    on first encounter rather than silently keeping the first ALT and
    discarding the rest. The recommended preprocessing pipeline is::

        bcftools norm -m-any input.vcf.gz | bcftools norm -d exact -Oz -o output.vcf.gz

    ``-m-any`` splits every multi-allelic record into one record per ALT.
    ``-d exact`` then drops any records where CHROM/POS/REF/ALT are all
    identical to an earlier record (which can happen when the input VCF
    merged sources that already contained the biallelic form of a split).
    """
    vcf = VCF(vcf_path)

    cell_names = np.array(vcf.samples)
    n_cells = len(cell_names)

    variant_ids = []
    chroms = []
    positions = []
    n_skipped = 0
    n_duplicates = 0
    seen_variants = set()

    # Sparse path: COO triples per layer, flushed to CSR chunks every
    # _CHUNK_SIZE variants. Avoids materializing dense (variants × cells)
    # intermediates that OOM at single-cell joint-VCF scale.
    dp_rows, dp_cols, dp_vals = [], [], []
    ad_rows, ad_cols, ad_vals = [], [], []
    x_rows, x_cols, x_vals = [], [], []
    dp_chunks, ad_chunks, x_chunks = [], [], []
    local_idx = 0

    # Dense path: per-variant arrays, stacked at end (small bulk-VCF use case).
    depth_data = []
    alt_data = []
    genotype_data = []

    def flush_chunk(n_rows):
        """Build CSR chunk from current COO buffers and reset them."""
        if n_rows == 0:
            return
        for rows, cols, vals, chunks in (
            (dp_rows, dp_cols, dp_vals, dp_chunks),
            (ad_rows, ad_cols, ad_vals, ad_chunks),
            (x_rows, x_cols, x_vals, x_chunks),
        ):
            if rows:
                r = np.concatenate(rows)
                c = np.concatenate(cols)
                v = np.concatenate(vals)
            else:
                r = np.array([], dtype=np.int32)
                c = np.array([], dtype=np.int32)
                v = np.array([], dtype=np.float32)
            m = coo_matrix((v, (r, c)), shape=(n_rows, n_cells), dtype=np.float32).tocsr()
            chunks.append(m)
            rows.clear()
            cols.clear()
            vals.clear()

    iterator = tqdm(vcf, desc="Loading VCF", unit=" sites") if show_progress else vcf

    for variant in iterator:
        has_alt = variant.ALT and len(variant.ALT) > 0

        if skip_missing_alt and not has_alt:
            n_skipped += 1
            continue

        if has_alt and len(variant.ALT) > 1:
            raise ValueError(
                f"Multi-allelic record at {variant.CHROM}:{variant.POS} "
                f"({variant.REF}>{','.join(variant.ALT)}). cellspec requires one ALT "
                f"allele per VCF record (AD/DP semantics are ambiguous otherwise). "
                f"Split multi-allelic sites before loading, e.g.:\n"
                f"    bcftools norm -m-any input.vcf.gz | "
                f"bcftools norm -d exact -Oz -o output.vcf.gz"
            )

        alt_allele = variant.ALT[0] if has_alt else "."
        variant_id = f"{variant.CHROM}-{variant.POS}-{variant.REF}>{alt_allele}"

        if variant_id in seen_variants:
            n_duplicates += 1
            continue

        seen_variants.add(variant_id)
        variant_ids.append(variant_id)
        chroms.append(variant.CHROM)
        positions.append(variant.POS)

        # cyvcf2 uses various sentinel values for missing data (INT32_MIN, -1).
        # Replace any negative values with 0.
        alt_dp = variant.gt_alt_depths
        ref_dp = variant.gt_ref_depths
        alt_dp = np.where(alt_dp < 0, 0, alt_dp)
        ref_dp = np.where(ref_dp < 0, 0, ref_dp)
        tot_dp = alt_dp + ref_dp
        gt = np.asarray(variant.gt_types)  # 0=HOM_REF, 1=HET, 2=UNKNOWN, 3=HOM_ALT

        # Treat UNKNOWN (./.) calls as missing across all layers.
        is_called = gt != 2

        if sparse:
            # DP: nonzero total depth at called sites
            nz = np.flatnonzero((tot_dp > 0) & is_called)
            if nz.size:
                dp_rows.append(np.full(nz.size, local_idx, dtype=np.int32))
                dp_cols.append(nz.astype(np.int32, copy=False))
                dp_vals.append(tot_dp[nz].astype(np.float32, copy=False))

            # AD: nonzero alt-allele depth at called sites
            nz = np.flatnonzero((alt_dp > 0) & is_called)
            if nz.size:
                ad_rows.append(np.full(nz.size, local_idx, dtype=np.int32))
                ad_cols.append(nz.astype(np.int32, copy=False))
                ad_vals.append(alt_dp[nz].astype(np.float32, copy=False))

            # X: store HET (1) and HOM_ALT (3) only. HOM_REF (0) collapses to
            # the sparse default; UNKNOWN (2) is dropped on purpose so .X==0
            # consistently means "HOM_REF or missing."
            nz = np.flatnonzero((gt == 1) | (gt == 3))
            if nz.size:
                x_rows.append(np.full(nz.size, local_idx, dtype=np.int32))
                x_cols.append(nz.astype(np.int32, copy=False))
                x_vals.append(gt[nz].astype(np.float32, copy=False))

            local_idx += 1
            if local_idx >= _CHUNK_SIZE:
                flush_chunk(local_idx)
                local_idx = 0
        else:
            # Dense path: mask UNKNOWN entries to 0 in all layers; restrict
            # X to HET/HOM_ALT so semantics match the sparse path.
            tot_dp_masked = np.where(is_called, tot_dp, 0).astype(np.float32, copy=False)
            alt_dp_masked = np.where(is_called, alt_dp, 0).astype(np.float32, copy=False)
            gt_x = np.where((gt == 1) | (gt == 3), gt, 0).astype(np.float32, copy=False)
            depth_data.append(tot_dp_masked)
            alt_data.append(alt_dp_masked)
            genotype_data.append(gt_x)

    if sparse:
        flush_chunk(local_idx)
        local_idx = 0

    if skip_missing_alt and n_skipped > 0 and show_progress:
        print(f"Skipped {n_skipped} sites with missing ALT alleles")

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

    if n_variants == 0:
        raise ValueError(
            f"No variants loaded from {vcf_path}. "
            f"{'All variants had missing ALT alleles. ' if skip_missing_alt else ''}"
            "Check your VCF file or set skip_missing_alt=False."
        )

    if show_progress:
        print(f"Building matrices for {n_variants} sites × {n_cells} samples...")

    if sparse:
        # Each chunk has shape (chunk_variants, n_cells); vstack and transpose
        # to the AnnData convention (n_cells, n_variants).
        def _combine(chunks):
            if not chunks:
                return csr_matrix((n_cells, 0), dtype=np.float32)
            return vstack(chunks, format="csr").T.tocsr()

        X = _combine(x_chunks)
        DP = _combine(dp_chunks)
        AD = _combine(ad_chunks)
    else:
        X = np.stack(genotype_data).T
        DP = np.stack(depth_data).T
        AD = np.stack(alt_data).T

    adata = ad.AnnData(X=X)
    adata.layers["DP"] = DP
    adata.layers["AD"] = AD

    adata.obs_names = cell_names
    adata.var_names = variant_ids

    adata.var["chrom"] = chroms
    adata.var["pos"] = positions

    adata.uns["vcf_source"] = vcf_path
    adata.uns["n_variants"] = n_variants
    adata.uns["n_cells"] = n_cells

    if show_progress:
        print(f"Done! Loaded {n_variants} sites × {n_cells} samples")

    return adata
