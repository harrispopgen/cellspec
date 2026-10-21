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

This initial step can take a somewhat long time, especially for datasets with a lot of alleles. As such, it a good idea to save your data in .h5ad format for more convenient loading in the future:

```python
adata.write_h5ad(filename)
```

### A typical workflow

Once loaded, a standard trinucleotide spectrum workflow looks like this:

```python
# Restrict to SNPs and annotate the ancestral trinucleotide context
spc.pp.filter_to_snps(adata)
spc.pp.annotate_contexts(adata, fasta_path="reference.fa")

# Quality-filter by coverage. Default (layer='DP') requires every cell to
# meet min_depth; see `pp.filter_by_coverage` for AD-based and depth-range
# filtering (e.g. layer='both', max_depth=...).
spc.pp.filter_by_coverage(adata, min_depth=10)

# Compute a per-cell 96-channel spectrum (stored in adata.obsm['spectrum_somatic'])
spc.tl.compute_spectrum(adata, key="somatic")

# Plot the aggregate spectrum across all cells
spc.pl.spectrum(adata, key="somatic", aggregate="sum")
```

### Variant allele frequencies

`cellspec` provides two coverage-normalized VAF estimators:

```python
# Per-cell VAF layer (stored in adata.layers['VAF'])
spc.tl.add_vaf_layer(adata, target_dp=10)

# Pseudobulk VAF per site (stored in adata.var['bulk_vaf'])
spc.tl.compute_bulk_vaf(adata)
```

Both use hypergeometric downsampling to a target depth so that sites with
different coverage are directly comparable. `compute_bulk_vaf` is often a
convenient first-pass filter for picking out high-confidence variants
across a cohort before computing per-cell spectra.

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
