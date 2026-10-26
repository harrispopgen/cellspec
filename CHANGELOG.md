# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog][],
and this project adheres to [Semantic Versioning][].

[keep a changelog]: https://keepachangelog.com/en/1.0.0/
[semantic versioning]: https://semver.org/spec/v2.0.0.html

## [Unreleased]

### Added

- `tl.compute_spectrum` grew an `aggregate` argument with four modes:
  `None` (per-cell matrix, default), `'sum'`, `'private_spectrum'`, and
  `'randomize'`. Groupby and private-mask handling moved out of `pl.spectrum`
  into this function. The result is stored at `adata.obsm['spectrum_{key}']`
  when `aggregate=None` and `adata.uns['spectrum_{key}']` otherwise.
- `tl.compute_spectrum(aggregate='randomize', ...)`: single-carrier
  assignment (Beichman et al.) — each variant with ≥1 carrier contributes
  one count, randomly credited to a carrier cell (or its group when
  `groupby` is given). Prevents common polymorphisms from inflating a
  group's spectrum. Deterministic when `groupby=None`; `random_state`
  controls the per-group draw.
- `tl.compute_spectrum(aggregate='private_spectrum', private_key=...)`:
  spectrum of each owner's private variants, each variant counted exactly
  once per owner. Reads the mask from
  `adata.uns['{private_key}_mutations']`; inherits the per-cell vs
  per-group ownership from the prior `tl.private_mutations` call.
- `tl.compute_spectrum` gained a `csv_path=` argument that writes the
  computed Series/DataFrame to disk (creating parent dirs if needed).
- Auto-generated default keys: `compute_spectrum(aggregate=None)` →
  `'per_cell'`; `aggregate='sum'` → `'sum'`; with `groupby='lineage'` →
  `'sum_by_lineage'`; `aggregate='private_spectrum'` →
  `'private_spectrum_from_{private_key}'`. Resolves the old
  `spectrum_spectrum` double-prefix.
- `tl.compute_site_stats`: per-site depth summaries (`dp_total`, `dp_mean`,
  `dp_median`, `dp_min`, `dp_max`), carrier frequency (`carrier_freq`) and
  absolute carrier count (`n_carriers`) on `adata.var`. Supersedes the stats
  paths that lived inside the old `pp.filter_variants` /
  `pp.filter_by_coverage`.
- `tl.compute_cell_stats`: per-cell depth summaries (`total_dp`,
  `mean_dp_per_site`, and `callable_sites` when `min_depth` is given) on
  `adata.obs`. Replaces `tl.compute_callable_sites`.
- `tl.filter_sites`: variant-axis filtering from precomputed `.var` columns.
  Takes keyword thresholds (`min_bulk_dp`, `max_bulk_dp`, `min_mean_dp`,
  `max_mean_dp`, `min_median_dp`, `max_median_dp`, `min_carrier_freq`,
  `max_carrier_freq`, `min_n_carriers`, `max_n_carriers`) and errors with a
  remediation hint if the required column is missing.
- `tl.filter_cells`: cell-axis filtering from precomputed `.obs` columns
  (`min_total_dp`/`max_total_dp`, `min_mean_dp_per_site`/`max_mean_dp_per_site`,
  `min_callable_sites`/`max_callable_sites`).
- Filter history: every `tl.filter_sites` / `tl.filter_cells` call appends
  its applied thresholds to `adata.uns['filter_sites_history']` /
  `['filter_cells_history']`. Downstream functions can read this to detect
  inconsistent later settings.
- `tl.fit_cosmic_signatures`: fit COSMIC mutational signatures to spectra via
  SigProfilerAssignment, with optional `groupby` aggregation and direct
  `Series`/`DataFrame` spectrum input.
- `tl.compute_bulk_vaf`: pseudobulk VAF across samples with hypergeometric
  downsampling, stored in `adata.var[key]`.
- `tl.add_vaf_layer`: per-cell VAF layer (moved from `pp` to `tl`).
- `tl.compute_spectrum_from_mask`: 96-channel spectrum counts from a boolean
  variant mask or a trinucleotide-context series.
- Per-group spectrum mode in `tl.compute_spectrum` via `private_key`, plotted
  automatically by `pl.spectrum` and `pl.compare_spectra`.
- Customizable color palettes (`palette=` with `'default'`, `'dark'`, list, or
  dict) and transparent-background export in `pl.spectrum` /
  `pl.spectrum_from_df` / `pl.compare_spectra`.
- `chrom` and `pos` columns on `adata.var` from `pp.load_vcf`.

### Changed

- **`pl.spectrum` simplified (breaking)**: dropped `groupby`, `aggregate`,
  and `outdir` arguments. The function now autodetects what to plot from
  the storage location — `adata.obsm[f'spectrum_{key}']` renders as a
  multi-row per-cell ribbon (subset with `cells=`), while
  `adata.uns[f'spectrum_{key}']` is plotted as either a single Series or a
  per-group DataFrame. All grouping and aggregation now live on
  `tl.compute_spectrum`.
- **`pl.compare_spectra` simplified (breaking)**: dropped the `aggregate`
  argument. Per-cell storage is summed; `uns` Series/DataFrames are read
  directly, with DataFrame columns expanded into per-group rows.
- `tl.compute_spectrum` default `key` changed from `'spectrum'` to the
  auto-generated name described above. Existing callers that passed an
  explicit `key` are unaffected; callers relying on `obsm['spectrum_spectrum']`
  need to either pass `key='spectrum'` or update to the new default.
- Private-mutation spectra now count each private variant exactly once per
  owner (per cell or per group), replacing the previous naive per-cell sum
  that inflated group spectra by the number of cells in the group.
- **VAF semantics (breaking)**: `tl.add_vaf_layer`, `tl.compute_bulk_vaf`,
  and `utils.compute_vaf` now take `min_dp` instead of `target_dp`. Entries
  with `dp < min_dp` return `NaN` (previously they were silently clamped).
  The default is now the **25th percentile** of nonzero DP (per (cell, site)
  entry for `add_vaf_layer`, per site for `compute_bulk_vaf`).
- **VAF default is now raw AD / DP**, not hypergeometrically downsampled.
  Pass `normalize=True` to opt in to downsampling at `min_dp`. This also
  applies to `utils.compute_vaf`.
- `tl.compute_bulk_vaf` gained a `key` argument so raw and normalized
  pseudobulk VAFs can live side-by-side on `adata.var` (e.g.
  `'bulk_vaf_raw'`, `'bulk_vaf_norm'`).
- `tl.compute_bulk_vaf` warns when its `min_dp` is below a previously
  applied `tl.filter_sites(min_bulk_dp=...)` floor.
- `tl.add_vaf_layer` warns when more than `nan_warn_threshold` (default
  50%) of DP>0 entries fall below `min_dp`.
- `pp.load_vcf` builds sparse layers incrementally in CSR chunks, bounding
  peak memory to roughly the final nonzero count rather than
  `n_cells × n_variants`.
- `pp.load_vcf` now raises `ValueError` on the first multi-allelic record
  it encounters instead of silently keeping only the first ALT. The error
  points at the recommended preprocessing pipeline
  (`bcftools norm -m-any input.vcf.gz | bcftools norm -d exact`). This
  prevents a silent-data-corruption class of bug where the stored `AD`
  no longer reflects the first ALT once cyvcf2 is handed a multi-allelic
  record.
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

### Removed

- `tl.normalize_spectrum` — unused in the rest of the package; callers
  should divide the stored spectrum themselves (e.g. by `.sum(axis=1)` for
  proportions or by an `.obs` column for per-callable-site rates).
- `tl.compute_spectrum_from_mask` — the "I have a boolean variant mask,
  give me 96-channel counts" case is now served by
  `tl.compute_spectrum(..., variant_mask=mask, aggregate='sum')`.
- `pp.filter_variants`, `pp.filter_cells`, `pp.filter_by_coverage` —
  replaced by `tl.filter_sites` / `tl.filter_cells` operating on columns
  from `tl.compute_site_stats` / `tl.compute_cell_stats`. `pp.filter_to_snps`
  is the only filter that remains in `pp`.
- `tl.compute_callable_sites` — folded into `tl.compute_cell_stats`
  (pass `min_depth=...` to produce the same `adata.obs['callable_sites']`
  column that `tl.normalize_spectrum(..., obs_column='callable_sites')`
  consumes).

### Fixed

- `pl.spectrum_from_df` now creates missing parent directories when writing
  to `outpath`.
- `pp.load_vcf` masks negative/invalid `AD`/`DP` sentinel values from cyvcf2
  to 0.
- `tl.compute_bulk_vaf` handles the edge case where all sites have zero depth.
