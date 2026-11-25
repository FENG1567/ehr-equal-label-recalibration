# Repository publication and release

This public repository is available at https://github.com/FENG1567/ehr-equal-label-recalibration. The deposited code release is tagged `v1.0.0` and corresponds to editor package `20261125-r9`; the manuscript's scientific content and supplementary tables are unchanged from r8. The current figure display authority is `data/presentation/figure_display_index.json` (version `2026.11.25-r7`). Historical r4 numbering is retained only as provenance.

The release includes `.github/`, `.gitignore` and `.gitattributes` so that CI, output exclusions and line-ending rules travel with the source. Install the fully pinned public dependencies from `envs/requirements-public.lock.txt`, then run the release check, verify-only mode, unit tests and aggregate display reproduction described in `README.md`. The hosted workflow uses the same lock file and reproduces aggregate outputs; it does not refit clinical models.

Two byte-preserved aggregate CSVs exceed GitHub's 25 MiB browser upload limit: `Data15/sites/site_adaptation_repeat_scores.csv` (51,447,891 bytes; about 49.1 MiB) and `Data5/reviewed_NWICU_SICdb/transport_calibration_draws.csv` (45,323,976 bytes; about 43.2 MiB). Both remain below GitHub's 100 MB regular-file limit and are not patient predictions. If preparing a replacement upload, use GitHub Desktop or Git CLI. Do not add raw databases or checkpoint binaries to satisfy a missing-input check.

For a future code release, update the version and release URL in `CITATION.cff` and availability statements, refresh package checksums with `python scripts/update_checksums.py` after metadata edits, and rerun the checks documented in `README.md`. Keep protected scientific-input hashes unchanged. No DOI is assigned in this repository.

Official references: https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github ; https://creativecommons.org/licenses/by/4.0/ .
