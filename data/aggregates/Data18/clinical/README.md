# Current eligible clinical summary

Aggregate-only summaries for stays with `eligible_24h=true` in the reviewed v4.2 gates. Gate rows are matched to frozen sequence NPZ rows by a strict one-to-one `(db, stay_id, subject)` key; gate input order is not assumed. No patient-level rows or raw identifiers are exported.

Age values retain source top-coding (MIMIC/NWICU 91; eICU 90); top-coded values are not exact ages. Sex uses source mapping 1=male and 0=female; missing values remain unknown. Token statistics use NPZ `L` only among current eligible stays; cap-128 counts `L >= 128`. No verified derived severity composite was provided, so no score is reported; APACHE/SOFA were not read or inferred.

MIMIC split-2 counts include only currently eligible stays whose frozen NPZ row is assigned to split 2 (split SHA in manifest). They are not presented as model performance counts.
