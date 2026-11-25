# V5 sites analysis


Calibration curves use raw scores and the exact server-only Platt fit at 500 labels, repeat 0, on the reused v4.2 patient roles. The diagnostic smoother has three knots at assessment logit-score q10/q50/q90, two predictor degrees of freedom plus intercept (three coefficients), and 1000 patient-cluster bootstrap draws. Curves are limited to empirical p01-p99 support. Constant scores and Jeffreys fallbacks appear as a point reference.

Brier and logloss are both reported. P05/P95 across 200 adaptation draws describe repeat variability on the fixed assessment pool, separately from the patient bootstrap confidence interval. Paired gaps use the per-stay loss means across all 200 fits, followed by a shared patient-cluster bootstrap; patient-weighted estimates are a sensitivity analysis.

Across-site estimates use the same feasible site set within each budget. Micro summaries pool stays or unique global patients; macro summaries give each feasible site equal weight. Intervals resample global eICU patient clusters, so a patient who appears at multiple hospitals receives one shared bootstrap multiplicity.

Each across-site row reports feasible-site, assessment-stay, unique-patient and site-patient-membership denominators. Infeasible site cells retain both stay- and patient-weighting rows. The historical v4 site coverage reference is sourced from the original verified v4 metrics and manifest.

Cohort-level atomic checkpoints are retained under checkpoints/. The tables above are canonical only when the section jobmanifest reports status=complete.


Hospitals are selected only by eligible unique-patient count (largest 24; ascending canonical ID breaks ties). Global patient roles are locked before within-site sampling. Site results represent random within-site adaptation and make no temporal validation claim.

For the 206-hospital basic performance table, finite calibration-slope point coverage is distinct from the >=10-event and >=10-nonevent interval-stability rule. The rule does not exclude sites from point estimates or site selection.

## CSV schemas

### site_roster.csv

All 206 repaired eICU hospitals ranked by eligible unique-patient counts; the largest 24 are selected with an ascending hospital ID tie-break.

Rows: 206; SHA256: 9175e2814348d0316f751d53dd86ab08cd8cbf6a4159875abbc32c795cee049d.

| Column | Type |
|---|---|
| hospitalid | object |
| eligible_unique_patients | int64 |
| eligible_stays | int64 |
| selected_top24 | bool |
| eligible_patient_rank_largest_first | int64 |
| tie_break | object |

### site_basic_performance.csv

Recomputed basic FM and V+P LR metrics at all 206 repaired eICU hospitals; all point estimates retained regardless of event count.

Rows: 412; SHA256: 416c433abaf20568ff18e1fa7574639562ef6614673811a1f08be36b2f7f4815.

| Column | Type |
|---|---|
| scope | object |
| model_id | object |
| hospitalid | object |
| n_stays | int64 |
| n_patients | int64 |
| events | int64 |
| nonevents | int64 |
| prevalence | float64 |
| auroc | float64 |
| auroc_status | object |
| average_precision | float64 |
| average_precision_status | object |
| brier | float64 |
| logloss | float64 |
| calibration_intercept | float64 |
| calibration_slope | float64 |
| calibration_status | object |
| site_stability_flag_ge10_events_and_nonevents | bool |
| site_stability_rule_excludes_from_cohort | bool |
| model | object |
| metric_status | object |
| mean_predicted_probability | float64 |
| observed_event_rate | float64 |

### original_v4_site_reference.csv

Historical eICU hospital denominator, finite calibration-slope points and stability-flag count from the verified original v4 metrics and manifest.

Rows: 1; SHA256: 8b98686233143e5e3b12efc1c4b515f91b3b55e435b564449ee33d41e2be27d2.

| Column | Type |
|---|---|
| status | object |
| source_metrics_sha256 | object |
| source_manifest_sha256 | object |
| source_status | object |
| source_scope | object |
| model_id | object |
| hospital_denominator | int64 |
| finite_calibration_slope_points | int64 |
| stable_sites_ge10_events_and_nonevents | int64 |
| stable_site_rule | object |
| counts_source | object |

### site_adaptation_repeat_scores.csv

Full repeat-level 50/200/500 patient results for selected hospitals, both models, no-update/prevalence/Jeffreys/intercept/Platt, and stay- plus patient-weighted Brier/logloss.

Rows: 144000; SHA256: 49cc1efc932e65b87453ab9d0112c317c880625d3cead030e3cf0abd5fd8f22e.

| Column | Type |
|---|---|
| hospitalid | object |
| rank_by_eligible_patients | int64 |
| model | object |
| model_id | object |
| budget | int64 |
| repeat | int64 |
| method | object |
| events | int64 |
| nonevents | int64 |
| fit_ok | bool |
| fit_status | object |
| fallback_used | bool |
| fallback_probability | float64 |
| sample_patient_set_sha256 | object |
| sample_patient_index_stay_sha256 | object |
| sampling_seed | int64 |
| n_adaptation_patients | int64 |
| assessment_patients | int64 |
| assessment_stays | int64 |
| assessment_events | int64 |
| brier | float64 |
| logloss | float64 |
| brier_patient_weighted | float64 |
| logloss_patient_weighted | float64 |
| mean_predicted_probability | float64 |
| observed_event_rate | float64 |

### site_adaptation_summary.csv

Per-site, model, budget, method and weighting summary; infeasible cells retain both weighting rows and explicit adaptation/assessment denominators. Fit counts apply only to offset/intercept and Platt fits.

Rows: 1440; SHA256: 2d82cb05ab27ccbaac1d04a9f58e211af11707883092c6009cfd3a151079410c.

| Column | Type |
|---|---|
| hospitalid | object |
| rank_by_eligible_patients | int64 |
| model | object |
| model_id | object |
| budget | int64 |
| method | object |
| weighting | object |
| status | object |
| feasible_site | bool |
| available_adaptation_patients | int64 |
| assessment_patients | int64 |
| assessment_stays | int64 |
| assessment_site_patient_memberships | int64 |
| assessment_events | int64 |
| assessment_nonevents | int64 |
| fit_attempts | float64 |
| fit_successes | float64 |
| fit_failures | float64 |
| fit_failure_rate | float64 |
| brier_mean | float64 |
| brier_p05 | float64 |
| brier_p95 | float64 |
| logloss_mean | float64 |
| logloss_p05 | float64 |
| logloss_p95 | float64 |
| adaptation_draws | int64 |
| adaptation_event_p05 | float64 |
| adaptation_event_p95 | float64 |

### across_site_adaptation_summary.csv

Micro and macro stay- and patient-weighted losses over the same feasible site set at each budget, with explicit site/stay/patient denominators and 1000 shared global patient-cluster intervals.

Rows: 240; SHA256: ee5e964b491617777f0a0111fe21901e8255797c21eebfbd4bf794bc08b8087e.

| Column | Type |
|---|---|
| budget | int64 |
| model | object |
| model_id | object |
| method | object |
| metric | object |
| aggregation | object |
| weighting | object |
| status | object |
| selected_site_count | int64 |
| feasible_site_count | int64 |
| infeasible_site_count | int64 |
| feasible_assessment_stays | int64 |
| feasible_unique_assessment_patients | int64 |
| feasible_site_patient_memberships | int64 |
| global_cross_site_assessment_patients | int64 |
| adaptation_repeats_averaged | int64 |
| estimate | float64 |
| bootstrap_ci_95_lower | float64 |
| bootstrap_ci_95_upper | float64 |
| bootstrap_requested | int64 |
| bootstrap_valid | int64 |
| bootstrap_cluster_unit | object |
| fit_attempts | float64 |
| fit_successes | float64 |
| fit_failures | float64 |
| fit_failure_rate | float64 |

### site_role_flow.csv

Selected-site counts showing globally locked patient roles, adaptation and assessment support, and no temporal design.

Rows: 24; SHA256: e960579111a56b7642e69670df6f147b49b278738632381cf6f2b672c6785d06.

| Column | Type |
|---|---|
| hospitalid | object |
| rank_by_eligible_patients | int64 |
| eligible_unique_patients_all | int64 |
| global_adaptation_patients_in_site | int64 |
| global_assessment_patients_in_site | int64 |
| global_assessment_stays_in_site | int64 |
| patient_role_overlap | int64 |
| patients_crossing_hospitals_global_eicu | int64 |
| adaptation_index_stay | object |
| assessment_rule | object |
| index_stay_seed | int64 |
| sampling_seed_by_budget | object |
| time_design | object |
