import unittest

import pandas as pd

from status_consistent_v4_1 import (
    admit_order_flags,
    eicu_status_conflict,
    mimic_dod_bounds,
    nullable_status,
    reason_column,
    reconcile_hospital_ids,
)
from verify_status_consistent_v4_1 import assert_unchanged_columns


class StatusConsistentTests(unittest.TestCase):
    def test_eicu_unit_expired_hospital_alive_is_conflict(self):
        got = eicu_status_conflict(
            pd.Series(["Expired", "Alive", "Unknown", "Expired"]),
            pd.Series(["Alive", "Alive", "Alive", "Unknown"]),
        )
        self.assertEqual(got.tolist(), [True, False, False, False])

    def test_day_granular_dod_proven_death_and_overlap(self):
        dod = pd.Series(["2020-01-01", "2020-01-02", "2020-01-01", None])
        start = pd.Series(["2020-01-01 00:00", "2020-01-01 23:00", "2020-01-02 01:00", "2020-01-01 00:00"])
        got = mimic_dod_bounds(dod, start)
        self.assertEqual(got["source_patient_dod_death_by_24h_proven"].tolist(), [True, False, True, False])
        self.assertEqual(got["source_patient_dod_preindex_proven"].tolist(), [False, False, True, False])
        self.assertEqual(got["source_patient_dod_interval_overlaps_24h"].tolist(), [False, True, False, False])
        self.assertEqual(float(got.loc[1, "source_patient_dod_upper_bound_hr"]), 25.0)

    def test_admission_order_is_flagged_at_strict_boundaries(self):
        after_icu, after24 = admit_order_flags(pd.Series([0.0, 0.01, 24.0, 24.01, float("nan")]))
        self.assertEqual(after_icu.tolist(), [False, True, True, True, False])
        self.assertEqual(after24.tolist(), [False, False, False, True, False])

    def test_hospital_id_reconciliation_requires_complete_agreement(self):
        got = reconcile_hospital_ids(pd.Series([1, 206]), pd.Series([1.0, 206.0]))
        self.assertEqual(got.tolist(), ["1", "206"])
        with self.assertRaisesRegex(ValueError, "disagree"):
            reconcile_hospital_ids(pd.Series([1, 2]), pd.Series([1, 3]))
        with self.assertRaisesRegex(ValueError, "cover every row"):
            reconcile_hospital_ids(pd.Series([1, None]), pd.Series([1, 2]))

    def test_status_or_proven_timing_issue_outranks_strict_gap_reason(self):
        frame = pd.DataFrame({
            "source_time_status_conflict": [True],
            "source_landmark_issue": [False],
            "source_endpoint_unknown": [False],
            "source_y_mort_match": [True],
            "adult_eligible": [True],
            "input_window_le_24h_verified": [True],
            "input_available_24h": [True],
            "icu_exit_hr": [48.0],
            "hospital_exit_hr": [36.0],
            "source_postlandmark_ordering_issue": [True],
        })
        got = reason_column(frame, "eicu", pd.Series([False]), pd.Series([False]), "strict_v4_1")
        self.assertEqual(got.iloc[0], "source_time_status_conflict")

    def test_nullable_status_keeps_unknown_status_unknown(self):
        got = nullable_status(pd.Series(["Expired", "Alive", "Unknown"]))
        self.assertEqual(got.iloc[0], True)
        self.assertEqual(got.iloc[1], False)
        self.assertTrue(pd.isna(got.iloc[2]))

    def test_unchanged_nullable_numeric_columns_compare_with_missing_values(self):
        original = pd.DataFrame({
            "stay_id": [1, 2],
            "subject_id": [10, 20],
            "y_mort": [1, 0],
            "source_y_mort": pd.Series([1, pd.NA], dtype="Int64"),
            "source_y_mort_match": [True, False],
            "input_window_le_24h_verified": [True, True],
        })
        corrected = original.copy()
        assert_unchanged_columns(original, corrected, "mimic", "landmark_v4_1")


if __name__ == "__main__":
    unittest.main()
