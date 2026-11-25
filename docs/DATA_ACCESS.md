# Data access

| Reused source | Version in manuscript | Official location | Reproduction requirement |
|---|---|---|---|
| MIMIC-IV | 3.1 | https://physionet.org/content/mimiciv/3.1/ | Obtain under the source's access/training/agreement conditions |
| eICU Collaborative Research Database | 2.0 | https://physionet.org/content/eicu-crd/2.0/ | Obtain under the source's access/training/agreement conditions |
| Northwestern ICU (NWICU) | 0.1.0 | https://physionet.org/content/nwicu-northwestern-icu/0.1.0/ | Obtain under the source's current terms |
| SICdb | 1.0.8 | https://physionet.org/content/sicdb/1.0.8/ | Original endpoint is an exploratory case-period mortality proxy |
| INSPIRE | 1.4.2 | https://physionet.org/content/inspire/1.4.2/ | Obtain under original credential/agreement terms; no credential bypass |
| PhysioNet Challenge 2012 | 1.0.0 | https://physionet.org/content/challenge-2012/1.0.0/ | Use the original record IDs; verified patient identity is unavailable |

This repository redistributes none of those individual records. Follow the linked source record's terms at the time of access. Controlled-access users must obtain their own authorization. This repository's MIT/CC BY licenses do not license database records or checkpoints. The deposited CSV/JSONs are aggregate research outputs rather than replacement copies of the original databases.

The data root structure expected by the archived extraction code is listed by `python scripts/clinical_workspace.py --inventory`. Root extraction conventions and table/field mappings are preserved in `archive/original/code/extract.py` and `config/variable_mapping.csv`. Do not upload an edited paths file containing secrets or patient products.
