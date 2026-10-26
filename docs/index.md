# `cellspec`: Single-cell mutation spectrum analysis

[![Tests][badge-tests]][tests]
[![Documentation][badge-docs]][documentation]

[badge-tests]: https://img.shields.io/github/actions/workflow/status/harrispopgen/cellspec/test.yaml?branch=main
[badge-docs]: https://img.shields.io/readthedocs/cellspec

A python package for analyzing variant calls from high throughput single cell genome sequencing experiments. Provides a convenient `scanpy` style API for loading joint calling vcf files into `anndata` objects, and performing downstream processing and analysis tasks, including:

* coverage analysis and filtering
* Annotation ancestral trinucleotide sequence context of SNPs
* Computing trinucleotide mutation spectra
* Visualizing mutation spectra

*in development*:

* sequencing error / artifact correction
* Mutation signature fitting and de novo signature discovery
* Phylogenetic analysis
    * distance based
    * maximum liklihood
    * Bayseian
* eQTL analysis (using genome-transcriptome coassay data)


## Installation

Install the latest development version:

```bash
git clone https://github.com/harrispopgen/cellspec.git
cd cellspec
pip install -e .
```

## Getting started

As an homage to the **s**emi **p**ermeable **c**apsule technology that spurred the need for this package, I encourage the following convention when importing `cellspec`:

```python
import cellspec as spc
```

**`cellspec`** uses the {class}`~anndata.AnnData` class to store joint calling data.

```{image} https://raw.githubusercontent.com/scverse/anndata/main/docs/_static/img/anndata_schema.svg
:width: 500px
```

> At the most basic level, an {class}`~anndata.AnnData` object `adata` stores a data matrix `adata.X`, annotation of observations `adata.obs` and variables `adata.var` as `pd.DataFrame` and unstructured annotation `adata.uns` as `dict`. Names of observations and variables can be accessed via `adata.obs_names` and `adata.var_names`, respectively. {class}`~anndata.AnnData` objects can be sliced like dataframes, for example, `adata_subset = adata[:, list_of_gene_names]`.

In **`cellspec`**, observations are cells (or samples), and variables are bi-allelic sites. Genotype calls from the vcf file are stored in `adata.X`, and depth information in `adata.layers`. Total read depth at each site in each cell is stored in `adata.layers["DP"]`, and alternate allele read depth is stored in `adata.layers["AD"]`.

To load a vcf into anndata:

```python
adata = spc.pp.load_vcf(filename)
```

**One ALT per record required.** `pp.load_vcf` stores a single ALT per
variant and uses per-record AD/DP, so multi-allelic VCF lines raise a
`ValueError`. Split them first:

```bash
bcftools norm -m-any input.vcf.gz | bcftools norm -d exact -Oz -o output.vcf.gz
```

`-m-any` splits each multi-allelic record into one record per ALT; the
second pass `-d exact` drops any records whose CHROM/POS/REF/ALT all
match an earlier record (which can happen when a merged VCF already
contained the biallelic form of a split).

This initial step can take a somewhat long time, especially for datasets with a lot of alleles. As such, it a good idea to save your data in .h5ad format for more convenient loading in the future:

```python
adata.write_h5ad(filename)
```

### A typical workflow

Once loaded, a standard workflow is: restrict to SNPs, annotate the
ancestral trinucleotide context, compute per-site and per-cell QC
statistics, filter from those stats, and then compute the spectrum.

```python
# Preprocessing — SNPs + trinucleotide context
spc.pp.filter_to_snps(adata)
spc.pp.annotate_contexts(adata, fasta_path="reference.fa")

# QC stats land on adata.var / adata.obs
spc.tl.compute_site_stats(adata)                 # dp_total, dp_mean, dp_median, dp_min, dp_max, carrier_freq
spc.tl.compute_cell_stats(adata, min_depth=10)   # total_dp, mean_dp_per_site, callable_sites

# Filtering reads from the precomputed columns above
spc.tl.filter_sites(adata, min_bulk_dp=100, min_carrier_freq=0.01)
spc.tl.filter_cells(adata, min_callable_sites=1000)

# Spectrum
spc.tl.compute_spectrum(adata, key="somatic")
spc.pl.spectrum(adata, key="somatic", aggregate="sum")
```

Each `filter_sites` / `filter_cells` call appends its applied thresholds
to `adata.uns['filter_sites_history']` / `['filter_cells_history']`, so
downstream functions (e.g. `tl.compute_bulk_vaf`) can warn when a later
`min_dp` is below an already-applied floor.

### Variant allele frequencies

`cellspec` provides per-cell and pseudobulk VAF estimators. Both default to
**raw** VAF (`AD / DP`) with a `min_dp` floor — any entry below `min_dp`
becomes `NaN`. Setting `normalize=True` applies hypergeometric
downsampling at `min_dp` for coverage-uniform comparison across sites.

```python
# Per-cell VAF layer, stored in adata.layers['VAF']
spc.tl.add_vaf_layer(adata)                      # raw AD / DP; min_dp from 25th percentile
spc.tl.add_vaf_layer(adata, min_dp=20, normalize=True)   # downsampled to 20

# Pseudobulk VAF per site, stored in adata.var[key]. Use distinct keys to
# keep raw and normalized side-by-side.
spc.tl.compute_bulk_vaf(adata, key="bulk_vaf_raw")
spc.tl.compute_bulk_vaf(adata, normalize=True, key="bulk_vaf_norm")
```

If `min_dp` is omitted, it defaults to the 25th percentile of DP (per
(cell, site) entry for `add_vaf_layer`, per site for `compute_bulk_vaf`).

### Mutational signatures

Fit COSMIC mutational signatures to a computed spectrum via
`tl.fit_cosmic_signatures` (wraps `SigProfilerAssignment`):

```python
# Per-cell fit (works best when each cell carries many mutations)
spc.tl.fit_cosmic_signatures(adata, key="somatic")

# For sparse single-cell data, aggregate within groups first
spc.tl.fit_cosmic_signatures(adata, key="somatic", groupby="lineage")
```

Please refer to the [documentation][] and [tutorials][] for more instruction,
and the [API documentation][] for information on specific functionality.

## Contents

```{toctree}
:maxdepth: 2

api/index
tutorials/index
changelog
contributing
references
```

[tests]: https://github.com/harrispopgen/cellspec/actions/workflows/test.yaml
[documentation]: https://cellspec.readthedocs.io
[tutorials]: https://cellspec.readthedocs.io/en/latest/tutorials/index.html
[api documentation]: https://cellspec.readthedocs.io/en/latest/api.html
