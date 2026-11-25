# Clinical analysis reconstruction

## Distinguish two reproduction levels

`scripts/reproduce.py` is a tested public aggregate-display workflow. It does not recompute the original cohort, fitted models, recalibrators or patient bootstrap. The source archive permits inspection and reconstruction of those analyses after legally obtaining the clinical inputs; no clean end-to-end clinical rerun was performed while preparing this release.

Original scripts retain the layout `/data/admin/fengshuo/ehr_fm_20261102` and dated run subdirectories. Many verify complete input hashes before use. Changing these paths/hashes globally without checking their meaning could invalidate provenance. Use a private Linux workspace/container mounted at that layout, or explicitly port the code with a recorded diff. Do not copy the historical virtual environments or administrator/systemd launchers. The original scientific source, result manifests and dependency inventory are deposited; machine-specific resource wrappers are not scientific dependencies.

```bash
python scripts/clinical_workspace.py --inventory
python scripts/clinical_workspace.py --prepare /your/private/new-workspace
python scripts/clinical_workspace.py --check /your/private/new-workspace
```

Preparation copies only the original code and variable mapping into an empty or new directory. It never downloads records, trains a model, modifies the production server, copies patient data or changes privileges. The check lists missing private inputs. Preparation is not a claim that the clinical pipeline has run. Keep this workspace outside the public repository.

## Stage order

1. Obtain the six versioned source databases under their original conditions. Rebuild events/cohorts using `code/extract.py` and sequences/splits using `code/build_data.py`; preserve source table versions, vocabularies, zero/padding distinctions, bins and one-patient roles.
2. Recreate v4 outcome/landmark eligibility and repaired eICU/NWICU inputs. `temporal/code`, `availability_repair` and `engine/code/cohort_registry_v4.py` define source paths and native endpoint semantics. SICdb remains exploratory. Do not silently replace primary eligibility by administrative exit ordering.
3. The historical main comparisons require the specific old frozen checkpoints and readout/eligibility inputs identified in Data12. They are not included or claimed to be reconstructable exactly. Old-member reconstruction in Data21 is conditional. A newly trained model is a new run, not the historical checkpoint.
4. For auditable source-only sensitivity: v5 prepare_v2_unicode → v5_clean_pretraining_gpu → v5_clean_pretraining_evaluate; retain all three seeds and control arms. The v5 scripts give the actual masks, source-only binning, training and C-selection rules.
5. Recreate v5 calibration/sites, repaired fixed-head controls/source-selected heads, direct-token decoder, retained-token and clinical support, and the ICU-death bridge. The Data13–19 mapping is in `provenance/result-code-map.csv`.
6. For calendar-separated sensitivity use v7 temporal_availability_v7.py with its `--stage prepare`, `--stage train`, `--stage evaluate` interface and its private prerequisites. The archived original has a small `--self-test`, but this release has not run a new temporal model. For Data22 use clean_calibration_extension_v7.py with the source-only run it references. An earlier run's hashed predictions/protocol files are prerequisites, not public aggregate substitutes.
7. Recompute summaries and compare against all source CSVs using the same populations, label plans, fit-failure policy and paired inferential units. Preserve unfavorable results, seed-specific output and fallback fits. Replot using the public display workflow only after those comparisons.

## Inputs not deposited

Raw records; per-patient/episode sequences, labels and metadata; eligibility rosters and identity-aligned embeddings/predictions; adaptation membership draws; training-member rosters; private checkpoints and fitted heads; original clinical environment images. These must be regenerated or separately obtained with authorization. Their available hashes and aggregate provenance remain in Data12/14/20/21 and other manifests. No synthetic records are substituted for real clinical inputs.

## Environment and limits

Python 3.12.3 and the observed CPU libraries are recorded in `envs/analysis-runtime-observed.json`. Earlier engine/GPU scripts used distinct environments; Data12 records PyTorch metadata. `envs/requirements-clinical-observed.txt` is an observed core inventory, not a portable solver lock. The original scripts' static imports appear in `provenance/original-source-manifest.json`; obtain optional FEMR/MOTOR under its own terms. The independent public-display environment has a tested frozen dependency list. No prospective or bedside validity is asserted by any reproduction step.
