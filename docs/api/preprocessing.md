# Preprocessing: `pp`

Functions required before QC and analysis: loading VCF files, SNP filtering,
and trinucleotide context annotation. Depth- and carrier-based filtering
lives in `tl` and operates on precomputed stats — see
[Tools](tools.md).

## Loading data

```{eval-rst}
.. module:: cellspec.pp
.. currentmodule:: cellspec

.. autosummary::
    :toctree: ../generated

    pp.load_vcf
```

## SNP filtering

```{eval-rst}
.. autosummary::
    :toctree: ../generated

    pp.filter_to_snps
```

## Annotation

```{eval-rst}
.. autosummary::
    :toctree: ../generated

    pp.annotate_contexts
```
