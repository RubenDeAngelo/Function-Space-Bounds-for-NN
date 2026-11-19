# Rights and attribution

The repository's MIT licence applies to the research software and its software
documentation, configuration and numerical record files. The release terms were
specified by Moritz Maibaum, a repository collaborator, for this paper release.

Scientific figure artwork in `LaTeX/` retains the authors' copyright. The
manuscript text and manuscript PDFs are not distributed in this code release;
the software licence does not relicense the paper.
No Springer publishing agreement is asserted by this pre-submission release.

The software authors are Johannes Gründler, Moritz Maibaum and Philipp Petersen.
The preserved corner experiment originated in Moritz Maibaum's RICAM project;
its original source, checkpoint hashes and restricted use in this paper are
recorded in `Code/Function-Space-Bounds-for-NN/experiments/pde_revision_20261112/corner_source/provenance.json`.
Raw historical records may contain quantities outside the present paper's scope.

This release vendors no Mothercode software. It uses the standalone frozen
backends and the original pre-migration code base. External packages (GMP, MPFR,
gmpy2, NumPy, Matplotlib, PyTorch and other optional dependencies) retain their
own licences; their implementations are not relicensed by this repository.
System libraries and Python package installations are not included in the source
distribution. Historical platform-specific executables are omitted from this distribution;
portable builds use the included research source and independently installed libraries.
Their original hashes remain in the provenance manifests and EXCLUDED_BINARIES.json.
