# Equal-label recalibration of transferred EHR risk models

Reproducibility materials for *Residual Differences in In-Hospital Mortality Prediction After Recalibrating Transferred EHR Models With the Same Local Labels*. Public code release: [v1.0.0](https://github.com/FENG1567/ehr-equal-label-recalibration/tree/v1.0.0). It accompanies editor package **20261125-r9**, which adds the Code availability statement to the r8 scientific content. The current figure display index is **2026.11.25-r7**. Authors: Shuo Feng, Chao Shi, Tianming Liu, Haoran Ding, Fenghe Hang, Zhenghao Huang and Liwei Liu. Contact: 18940835066@163.com.

## Quick start

Use Python 3.12 in a clean environment. The full public lock pins direct and transitive dependencies:

```bash
python -m venv .venv
# Linux/macOS: source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r envs/requirements-public.lock.txt
python scripts/check_release.py
python scripts/reproduce.py --verify-only
python -m unittest discover -s tests -v
python scripts/reproduce.py --output outputs
```

This renders six main and six supplementary figures from aggregate CSVs, exporting publication figures as PDF, SVG, PNG and TIFF with preview PNGs. It rebuilds the two main tables and nine supplementary physical tables from aggregate CSVs and checks them cell by cell against `data/presentation/tables.json`. The submitted English manuscript table bodies were checked against those aggregate specifications for editor package r9; the Supplementary Information table bodies were checked for r8, which is unchanged in r9. The comparison normalizes whitespace, line breaks and Unicode minus/dash forms. Table notes or captions may differ after external file-pointer cleanup; see `provenance/submission-document-bindings.json` for the document hashes and results. Main Table 2 is an eight-row raw-to-Platt summary; its complete 24-row calibration grid is retained in Supplementary Tables 2A–2D (six rows by six columns per cohort-budget block). Supplementary Tables appear in this order: clinical characteristics (S1), complete calibration grid (S2A–S2D), Data22 equal-label summary (S3), temporal assessment (S4), SICdb (S5), and point-only log loss (S6). The older seven-table snapshot is preserved as `tables_source_r2.json` for canonical comparison. The workflow does not access patient records, refit models or rerun the patient-cluster bootstrap. Numeric table cells and plotted results derive from deposited aggregate data; PDF bytes and fonts can differ by platform. `reference/figures/` contains the 12 byte-preserved v12 PDFs indexed by the current r7 display index. Generated outputs are ignored by Git.

## What is included

| Directory | Contents |
|---|---|
| `data/aggregates/Data1`–`Data22` | Authors' aggregate results; low-support site rows omitted as documented in `docs/PUBLIC_DATA_POLICY.md` |
| `data/presentation` | Frozen historical figure selectors, current r7 figure display index, revised English table snapshot and prior seven-table canonical snapshot |
| `src/presentation` | Aggregate table rebuilders and the two v12 figure renderers |
| `archive/original` | Original analysis source, retaining historical paths and safeguards |
| `config` | Variable mapping and clinical-stage/input metadata |
| `docs` | Data access, clinical workflow, limitations and data dictionary |
| `provenance` | Source-code hashes, result-to-code mapping, figure bindings, submission-document table bindings and release input hashes |
| `tests` | Aggregate-to-table consistency and release integrity tests |

For statistical refitting from clinical data, use `docs/CLINICAL_REPRODUCTION.md`. This requires independently authorized database access and the original pipeline input layout. Archived scripts are not advertised as a portable one-command clinical rerun. Historical checkpoint membership cannot be authenticated retrospectively; the newer source-only and calendar-separated analyses have separate provenance. No historical uncertainty is repaired by renaming a file or reconstructing a candidate roster.

## Licensing and availability

Project-authored software is licensed under **MIT**. Authors' aggregate results and displays are licensed under **CC BY 4.0**. See `LICENSING.md` and `THIRD_PARTY_NOTICES.md`; external databases and libraries keep their original terms.

Analysis code, software requirements and aggregate results for reproducing the figures and tables are available at [the v1.0.0 release](https://github.com/FENG1567/ehr-equal-label-recalibration/tree/v1.0.0). Instructions for analyses requiring the original clinical databases are in `docs/CLINICAL_REPRODUCTION.md`. The public workflow reproduces aggregate displays; it does not claim a full clinical-data rerun. Open site-level data follow `docs/PUBLIC_DATA_POLICY.md`.

## Historical figure-label provenance

`data/presentation/figure_display_index.json` (version 2026.11.25-r7) is the current authority for figure labels. The r4 renumbering map retained in that file and in `provenance/figure-source-bindings.json` documents a historical display change only: prior main labels Figure1–Figure6 map to r4 Figure2, Figure6, Figure1, Figure3, Figure4 and Figure5; prior supplementary labels FigureS1–FigureS6 map to r4 FigureS3, FigureS5, FigureS4, FigureS2, FigureS6 and FigureS1. Historical source identities and the frozen `figures.json` source bundle remain unchanged. This mapping is retained as provenance and does not identify the current display index.

Two aggregate CSVs exceed GitHub's 25 MiB browser upload limit, so use GitHub Desktop or Git CLI if uploading a replacement copy: `Data15/sites/site_adaptation_repeat_scores.csv` (51,447,891 bytes) and `Data5/reviewed_NWICU_SICdb/transport_calibration_draws.csv` (45,323,976 bytes). Both remain below GitHub's 100 MB regular-file limit.
