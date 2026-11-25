# MIMIC eligible availability audit addendum

Run: `v5_availability_eligible_20261123T172333Z_3860102`

This report aggregates the persisted v4 per-stay availability audit for all 94,343 frozen MIMIC sequence stays, the reviewed 73,936-row `eligible_24h` subset, and split-specific all-sequence and eligible subsets. The CSV contains aggregate counts and denominators only.

The audit is a composite proxy built from cached fixed-selection retained-token, frozen-bin, and old-versus-full sampled-key fields. It does not directly attribute every late-store event category and does not establish bedside or prospective clinical input availability. Old-versus-full rebuild key drift is a sampling/rebuild change and must not be presented as a pure clinical-availability effect. Historical Data6 `raw_record_time` counts are candidate records, not retained-token counts.

No raw event table was opened; bins were not recomputed; patient-level identifiers are not included in the output. See `manifest.json` for source hashes, exact denominator definitions, gate/split checks, and full-cache reconciliation.
