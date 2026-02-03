# cellspec: Single-cell mutation spectrum analysis

**cellspec** is a Python package for analyzing mutation spectra from bulk or single-cell variant calling data. It provides a scanpy-style API for computing 96-channel trinucleotide mutation spectra, calculating mutation rates, and visualizing mutation patterns.

## Key features

- **Load and process VCF files** into AnnData objects
- **Compute mutation spectra** with flexible filtering and counting strategies
- **Calculate mutation rates** normalized by callable sites
- **Identify private mutations** specific to cells or lineages
- **Visualize spectra** with publication-ready plots

## Installation

Install from PyPI:

```bash
pip install cellspec
```

Or install the development version:

```bash
git clone https://github.com/harrispopgen/cellspec.git
cd cellspec
pip install -e .
```

## Quick start

```python
import cellspec as spc

# Load VCF and annotate variants
adata = spc.pp.load_vcf('variants.vcf.gz')
spc.pp.annotate_contexts(adata, 'reference.fa')
spc.pp.filter_to_snps(adata)

# Compute mutation spectrum
spc.tl.compute_spectrum(adata, min_depth=10, key='somatic')

# Visualize
spc.pl.spectrum(adata, key='somatic', normalize=True)
```

## Citation

If you use cellspec in your research, please cite:

```
[Citation information to be added]
```

## Contents

```{toctree}
:maxdepth: 2

api/index
tutorials/index
changelog
contributing
references
```
