# Project overview

In this repo is python package I am developing for analyzing variant calls from high throughput single cell genome sequencing experiments. It provides a convenient `scanpy` style API for loading joint calling vcf files into `anndata` objects, and performing downstream processing and analysis tasks, including:

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

# Codebase

The package is built from the [`scverse` cookie cutter template](https://github.com/scverse/cookiecutter-scverse) and I aim to follow the guidelines required to publish a package in the [`scverse` ecosystem](https://github.com/scverse/ecosystem-packages?tab=readme-ov-file).

### `scverse` requirements:

Mandatory checklist for adding packages to `scverse`:

- [ ] The code is publicly available under an [OSI-approved](https://opensource.org/licenses/alphabetical) license
- [ ] The package provides versioned releases
- [ ] The package can be installed from a standard registry (e.g. PyPI, conda-forge, bioconda)
- [ ] Automated tests cover essential functions of the package and a reasonable range of inputs and conditions [^1]
- [ ] Continuous integration (CI) automatically executes these tests on each push or pull request [^2]
- [ ] The package provides API documentation via a website or README[^3]
- [ ] The package uses scverse datastructures where appropriate (i.e. AnnData, MuData or SpatialData and their modality-specific extensions)
- [ ] I am an author or maintainer of the tool and agree on listing the package on the scverse website

Recommendations I am following:

- [ ] The package provides tutorials (or "vignettes") that help getting users started quickly
- [ ] The package uses the [scverse cookiecutter template](https://github.com/scverse/cookiecutter-scverse).

The package documentation uses Sphinx with MyST (Markdown support) and builds via ReadTheDocs. I aim to provide useful vignettes (in addition to standard api docs) on this website.
