"""
cellspec: Cell-level mutation spectrum analysis

A scanpy-style API for computing and analyzing mutation spectra from
bulk or single-cell variant calls stored in AnnData objects.

The API is organized into three modules:
- pp: Preprocessing (loading VCFs, annotation, filtering)
- tl: Tools (spectrum computation, rate calculations, private mutations)
- pl: Plotting (visualization)

Data structure:
- adata.X stores genotype calls: 0=HOM_REF, 1=HET, 2=UNKNOWN, 3=HOM_ALT

Example usage:
    import cellspec as spc
    import anndata as ad

    # Preprocessing
    adata = spc.pp.load_vcf('variants.vcf.gz')
    spc.pp.annotate_contexts(adata, fasta_path='reference.fa')
    spc.pp.filter_to_snps(adata)

    # Tools
    spc.tl.compute_spectrum(adata, min_depth=10, key='somatic')
    spc.tl.compute_callable_sites(adata, min_depth=10)
    spc.tl.compute_rates(adata, spectrum_key='somatic')

    # Plotting
    spc.pl.spectrum(adata, key='somatic', aggregate='sum')
"""

from importlib.metadata import version

from . import pl, pp, tl, utils

__all__ = ["pl", "pp", "tl", "utils"]

__version__ = version("cellspec")
