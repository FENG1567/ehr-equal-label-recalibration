# Public release validation

Prepared 2026-11-25 for manuscript version 20261125-r4.

## r4 display synchronization

The r4 staging update changes presentation labels and table order only. Six main and six supplementary PDFs were renamed through temporary paths; their bytes and individual SHA-256 values are preserved. Renderer output stems, figure contracts, source maps, the display index, and source bindings now use r4 numbering. Historical source-figure identities and the frozen figures.json bundle remain unchanged.

Main Table 1 and Main Table 2 data cells are unchanged. Captions and English table notes follow the WPS document plan. The calibration grid is shown in Supplementary Tables 2A–2D. Supplementary table order is clinical characteristics (S1), calibration blocks (S2A–S2D), Data22 equal-label results (S3), temporal assessment (S4), SICdb (S5), and point-only log loss (S6). The seven-table tables_source_r2.json snapshot remains unchanged and is still checked by the table rebuilder.

This stage did not rerender the 12 PDFs. Prior direct-render and contact-sheet evidence applies because all PDF bytes are unchanged. The table DOCX and its evidence binding await final manuscript-package synchronization. No clinical refit, patient-level bootstrap, or hosted CI run is claimed.

## Validation commands

The public release tests and tables-only reproduction run against this staged package. Release input checks verify the updated file manifest and checksums. These checks verify aggregate display reproduction only; they do not independently validate the clinical analyses.
