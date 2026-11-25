# V5 calibration analysis


Calibration curves use raw scores and the exact server-only Platt fit at 500 labels, repeat 0, on the reused v4.2 patient roles. The diagnostic smoother has three knots at assessment logit-score q10/q50/q90, two predictor degrees of freedom plus intercept (three coefficients), and 1000 patient-cluster bootstrap draws. Curves are limited to empirical p01-p99 support. Constant scores and Jeffreys fallbacks appear as a point reference.

Brier and logloss are both reported. P05/P95 across 200 adaptation draws describe repeat variability on the fixed assessment pool, separately from the patient bootstrap confidence interval. Paired gaps use the per-stay loss means across all 200 fits, followed by a shared patient-cluster bootstrap; patient-weighted estimates are a sensitivity analysis.

The HGBT supplementary table is retained from verified v4 all-200-draw aggregate outputs for eICU and NWICU. Offset methods display as intercept; the original archive method is preserved in source_archive_method. These are historical summaries and were not recomputed on repaired-v5 scores.

Cohort-level atomic checkpoints are retained under checkpoints/. The tables above are canonical only when the section jobmanifest reports status=complete.


Hospitals are selected only by eligible unique-patient count (largest 24; ascending canonical ID breaks ties). Global patient roles are locked before within-site sampling. Site results represent random within-site adaptation and make no temporal validation claim.

For the 206-hospital basic performance table, finite calibration-slope point coverage is distinct from the >=10-event and >=10-nonevent interval-stability rule. The rule does not exclude sites from point estimates or site selection.

## CSV schemas

### global_recalibration_repeat_scores.csv

All 200 repeat-level raw, prevalence, Jeffreys, intercept and Platt results for each available budget; Brier/logloss and patient-weighted sensitivity.

Rows: 16000; SHA256: dd595b631e5b319682d76f33e86520fcb0471af588db6e438ba8539e0b7b4f8a.

| Column | Type |
|---|---|
| cohort | object |
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
| brier | float64 |
| logloss | float64 |
| brier_patient_weighted | float64 |
| logloss_patient_weighted | float64 |
| mean_predicted_probability | float64 |
| observed_event_rate | float64 |

### global_50_500_summary.csv

Complete 200-draw results at 50 and 500 labels, both proper scores, adaptation event P05/P95 and failed-fit/fallback counts.

Rows: 40; SHA256: 0391dd737bf30e805597da5ab7873032e98a1e6a63e21503016e8e83f87965c2.

| Column | Type |
|---|---|
| cohort | object |
| model | object |
| model_id | object |
| budget | int64 |
| method | object |
| attempts | int64 |
| fit_successes | float64 |
| fallback_count | int64 |
| events_p05 | float64 |
| events_p95 | float64 |
| brier_mean | float64 |
| brier_p05 | float64 |
| brier_p95 | float64 |
| logloss_mean | float64 |
| logloss_p05 | float64 |
| logloss_p95 | float64 |
| fit_attempts | float64 |
| fit_failures | float64 |
| fit_failure_rate | float64 |

### hgbt_supplementary_calibration_v4.csv

Verified v4 pooled eICU/NWICU HGBT calibration summaries across 200 adaptation draws; offset is displayed as intercept and preserved in source_archive_method. Historical aggregate only, not recomputed on repaired-v5 scores or site-level.

Rows: 56; SHA256: 4bb0f999ddc63a076910bd2578ac9c86f4c7fab1415d8e3739afa57f32533029.

| Column | Type |
|---|---|
| db | object |
| status | object |
| cohort_status | object |
| n_eligible_stays | float64 |
| n_index_patients | float64 |
| n_adaptation_patients | float64 |
| n_assessment_patients | float64 |
| n_assessment_stays | float64 |
| n_assessment_events | float64 |
| n_assessment_nonevents | float64 |
| assessment_precision_feasibility_200_events_200_nonevents | object |
| primary_budget_500_feasible_for_confirmation | object |
| primary_budget_feasibility_is_performance_blind | object |
| patient_overlap | float64 |
| bootstrap_cluster_unit | object |
| independent_patient_clustering_verified | object |
| one_random_index_stay_per_adaptation_patient | object |
| index_stay_selection_uses_outcome | object |
| stratified_by_patient_any_event | object |
| patient_pool_split_uses_outcomes | object |
| assessment_weighting | object |
| repeat_uncertainty_scope | object |
| bootstrap_uncertainty_scope | object |
| fixed_population_identity_sha256 | object |
| adaptation_index_patient_stay_sha256 | object |
| adaptation_patient_set_sha256 | object |
| assessment_population_identity_sha256 | object |
| assessment_patient_set_sha256 | object |
| seed | float64 |
| repeats | float64 |
| budgets | object |
| sampling_plan_sha256 | object |
| analysis_id | object |
| model_tag | object |
| reused_original_eicu_sampling_plan | bool |
| cross_version_curve_comparison_allowed | bool |
| budget | float64 |
| method | object |
| n_repeats | float64 |
| n_valid_repeats | float64 |
| n_budget_patients | float64 |
| n_budget_stays | float64 |
| n_zero_class_draws | float64 |
| n_one_class_draws | float64 |
| n_mixed_class_draws | float64 |
| events_min | float64 |
| events_median | float64 |
| events_max | float64 |
| nonevents_min | float64 |
| nonevents_median | float64 |
| nonevents_max | float64 |
| brier_mean | float64 |
| brier_p05 | float64 |
| brier_p95 | float64 |
| logloss_mean | float64 |
| logloss_p05 | float64 |
| logloss_p95 | float64 |
| negative_bss_rate_descriptive | float64 |
| fit_failure_rate | float64 |
| n_fit_fail | float64 |
| n_fit_success | float64 |
| n_fit_attempts | float64 |
| fit_failure_reasons | object |
| conditional_on_assessment_patients | object |
| assessment_prevalence_bss_is_deployable_reference | object |
| source_version | object |
| source_archive_method | object |
| source_artifact_path_internal | object |
| source_artifact_sha256 | object |
| source_endpoint_scope | object |
| supplement_scope | object |

### paired_loss_gap_bootstrap.csv

Raw gap, Platt gap, and post-gap minus raw-gap for Brier/logloss using saved per-stay means across all 200 fits; 1000 paired patient-cluster bootstraps and patient-weighted sensitivity.

Rows: 96; SHA256: c9ca9e88adb69bc6b381fa30c0c3ae00b32a1d08e81e3e0b8c30245777ea4037.

| Column | Type |
|---|---|
| cohort | object |
| budget | int64 |
| metric | object |
| contrast | object |
| weighting | object |
| estimate | float64 |
| bootstrap_ci_95_lower | float64 |
| bootstrap_ci_95_upper | float64 |
| bootstrap_requested | int64 |
| bootstrap_valid | int64 |
| assessment_stays | int64 |
| assessment_patients | int64 |
| adaptation_draws | int64 |
| adaptation_draw_p05 | float64 |
| adaptation_draw_p95 | float64 |
| adaptation_draw_mean | float64 |
| interval_scope | object |

### per_stay_repeat_mean_parity.csv

Parity checks that saved fit parameters recreate each server-only per-stay mean-loss array across 200 fits.

Rows: 64; SHA256: 55f258add5b2061271d517c918d0f3f1a2160a11f162127fa4e3388a6f5273e9.

| Column | Type |
|---|---|
| cohort | object |
| model | object |
| model_id | object |
| budget | int64 |
| method | object |
| metric | object |
| parity_ok | bool |
| max_abs_difference | float64 |

### calibration_curve_points.csv

Smooth risk curves and pointwise patient-cluster intervals for raw predictions and the fixed 500-label Platt repeat-0 fit.

Rows: 488; SHA256: 1bf903f9aca294c5616ad96a147bc370bf6e4d323e34450d76b0e8bc7ffdc090.

| Column | Type |
|---|---|
| cohort | object |
| model | object |
| model_id | object |
| method | object |
| curve_type | object |
| point_index | int64 |
| predicted_probability | float64 |
| smoothed_observed_risk | float64 |
| ci95_lower | float64 |
| ci95_upper | float64 |
| risk_support_low | float64 |
| risk_support_high | float64 |
| smoother_predictor_df | int64 |
| smoother_coefficient_count | int64 |
| knot_logit_q10 | float64 |
| knot_logit_q50 | float64 |
| knot_logit_q90 | float64 |
| bootstrap_requested | int64 |
| bootstrap_valid | int64 |
| bootstrap_invalid | int64 |
| curve_fit_status | object |

### calibration_risk_bins.csv

Empirical score-bin support, event counts, mean predictions and observed event fractions underlying the smooth curves.

Rows: 80; SHA256: 170d12f3de7b2bf05199830d861a88d7102b20ff96d5cfb401eca9d528eed85a.

| Column | Type |
|---|---|
| cohort | object |
| model | object |
| model_id | object |
| method | object |
| bin | int64 |
| n_stays | int64 |
| events | int64 |
| mean_predicted | float64 |
| observed | float64 |

### calibration_risk_summary.csv

Risk distribution, mean predicted and observed risks, Brier/logloss, curve status and fixed-fit provenance per model/cohort.

Rows: 8; SHA256: 9f87b9749af7620df9664db0ea3c50e4c4c620c7f5101f9bb54c1e922b47dd27.

| Column | Type |
|---|---|
| cohort | object |
| model | object |
| model_id | object |
| method | object |
| curve_status | object |
| n_stays | int64 |
| n_patients | int64 |
| events | int64 |
| nonevents | int64 |
| mean_predicted_probability | float64 |
| observed_event_rate | float64 |
| brier | float64 |
| logloss | float64 |
| p_min | float64 |
| p01 | float64 |
| p05 | float64 |
| p25 | float64 |
| p50 | float64 |
| p75 | float64 |
| p95 | float64 |
| p99 | float64 |
| p_max | float64 |
| smoother_predictor_df | int64 |
| smoother_coefficient_count | int64 |
| smoother_coefficients | object |
| knots_logit | object |
| probability_support | object |
| bootstrap_requested | int64 |
| bootstrap_valid | int64 |
| bootstrap_invalid | int64 |
| bootstrap_failures | object |
| fixed_calibrator_fallback | bool |
| assessment_scope | object |
| fixed_calibrator_budget | int64 |
| fixed_calibrator_repeat | int64 |
| platt_fit_ok | bool |
| platt_fit_status | object |
| global_sampling_plan_sha256 | object |
