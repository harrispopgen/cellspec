# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- `tl.fit_cosmic_signatures`: fit COSMIC mutational signatures to spectra via
  SigProfilerAssignment, with optional `groupby` aggregation and direct
  `Series`/`DataFrame` spectrum input.
- `tl.compute_bulk_vaf`: pseudobulk VAF across samples with hypergeometric
  downsampling, stored in `adata.var['bulk_vaf']`.
- `tl.add_vaf_layer`: per-cell VAF layer with coverage-normalized downsampling
  (moved from `pp` to `tl`).
- `tl.compute_spectrum_from_mask`: 96-channel spectrum counts from a boolean
  variant mask or a trinucleotide-context series.
- Per-group spectrum mode in `tl.compute_spectrum` via `private_key`, plotted
  automatically by `pl.spectrum` and `pl.compare_spectra`.
- Alt-depth and depth-range filtering in `pp.filter_by_coverage`
  (`layer='AD'`/`'both'`, `max_depth`, `min_alt_depth`, `max_alt_depth`).
- Customizable color palettes (`palette=` with `'default'`, `'dark'`, list, or
  dict) and transparent-background export in `pl.spectrum` /
  `pl.spectrum_from_df` / `pl.compare_spectra`.
- `chrom` and `pos` columns on `adata.var` from `pp.load_vcf`.

### Changed

- `pp.load_vcf` builds sparse layers incrementally in CSR chunks, bounding
  peak memory to roughly the final nonzero count rather than
  `n_cells × n_variants`.
- `tl.compute_callable_sites` moved into the `normalize` module (replaces the
  removed `compute_rates`).
- Spectrum plots (`pl.spectrum`, `pl.spectrum_from_df`, `pl.compare_spectra`)
  switched to a ribbon layout: six mutation-type panels butt edge-to-edge
  (`wspace=0`), y-axis is drawn only on the leftmost panel, trinucleotide
  x-tick labels are rendered in monospace, and the per-panel colored
  indicator rectangle has been removed. Default `figsize` is now `(12, 3)`
  per spectrum row. Multi-row plots use the spectrum / group name as the
  leftmost ylabel (any explicit `ylabel=` is stacked underneath).
- New plotting style knobs on all three functions: `yticks`, `ytick_scale`,
  `ytick_unit_label` (e.g. `ytick_scale=1e3, ytick_unit_label='thousands'`),
  `show_mut_type_titles` (default `True`), `show_xticks` (default `True`),
  `show_legend` (default `False`), `bar_width`, `group_padding`, `left_pad`,
  `wspace`.

### Fixed

- `pl.spectrum_from_df` now creates missing parent directories when writing
  to `outpath`.
- `pp.load_vcf` masks negative/invalid `AD`/`DP` sentinel values from cyvcf2
  to 0.
- `tl.compute_bulk_vaf` handles the edge case where all sites have zero depth.
