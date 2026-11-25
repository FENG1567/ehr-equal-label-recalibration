"""Focused synthetic tests for the corrected Challenge 2012 parser."""
import unittest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_c2012_v4 import (
    RawRecord,
    _aggregate_hourly,
    _get_extractor_definitions,
    canonical_parameter,
    count_outcome_mismatches,
    normalize_records,
    outcome_label_from_survival,
    parse_record_text,
    parse_time_to_hour,
)


class C2012BuilderTests(unittest.TestCase):
    def test_unavailable_rule_labels_survive_pandas_nan_coercion(self):
        labels = pd.Series([0, 1, None, np.nan])
        self.assertEqual(count_outcome_mismatches(labels, [0, 1, 0, 1]), 2)

    def test_native_parameter_aliases_and_non_equivalent_labs(self):
        self.assertEqual(canonical_parameter("Na"), "sodium")
        self.assertEqual(canonical_parameter("K"), "potassium")
        self.assertEqual(canonical_parameter("Mg"), "magnesium")
        self.assertEqual(canonical_parameter("NISysABP"), "sbp")
        self.assertEqual(canonical_parameter("NIDiasABP"), "dbp")
        self.assertEqual(canonical_parameter("NIMAP"), "map")
        self.assertIsNone(canonical_parameter("HCT"))
        self.assertIsNone(canonical_parameter("HCO3"))
        self.assertIsNone(canonical_parameter("MechVent"))
        self.assertIsNone(canonical_parameter("SaO2"))

    def test_elapsed_hours_allow_values_after_midnight(self):
        self.assertEqual(parse_time_to_hour("24:00"), 24.0)
        self.assertAlmostEqual(parse_time_to_hour("35:19"), 35 + 19 / 60)
        self.assertAlmostEqual(parse_time_to_hour("00:30:30"), 0.5083333333333333)

    def test_native_series_format_uses_recordid_descriptor_row(self):
        text = """Time,Parameter,Value
00:00,RecordID,132539
00:00,Age,54
00:00,Gender,0
00:00,Height,-1
00:00,ICUType,4
00:00,Weight,-1
00:07,HR,73
"""
        record = parse_record_text(text, file_name="132539.txt")
        self.assertEqual(record.record_id, 132539)
        self.assertEqual([event[1] for event in record.events], ["Age", "Gender", "Height", "Weight", "HR"])

    def test_colon_header_format_remains_supported(self):
        text = """RecordID: 99
Age: 70
Gender: 1
Height: 175
Time,Parameter,Value
00:00,HR,80
"""
        record = parse_record_text(text, file_name="99.txt")
        self.assertEqual(record.record_id, 99)
        self.assertEqual(record.header["age"], "70")
        self.assertEqual(len(record.events), 1)

    def test_blank_source_parameter_is_audited_not_invented(self):
        record = parse_record_text('Time,Parameter,Value\n00:00,RecordID,1\n02:43,,80\n', file_name='1.txt')
        vars_, ranges = _get_extractor_definitions()
        _, qc = normalize_records([record], vars_, ranges)
        self.assertEqual(qc['unmapped_parameter_count_within_24h'][''], 1)

    def test_sentinel_and_unit_cleaning_precede_hourly_mean(self):
        vars_, ranges = _get_extractor_definitions()
        record = RawRecord(
            record_id=1,
            file_name="1.txt",
            header={"recordid": "1", "age": "50", "gender": "1", "height": "170"},
            events=[
                (0.10, "HR", "80"),
                (0.20, "HR", "-1"),
                (0.25, "Na", "140"),
                (0.25, "K", "4"),
                (0.25, "Mg", "1"),
                (0.25, "FiO2", "0.4"),
                (0.25, "NISysABP", "120"),
                (0.25, "NIDiasABP", "70"),
                (0.25, "NIMAP", "80"),
                (0.25, "HCT", "40"),
                (0.25, "HCO3", "24"),
                (0.50, "Weight", "80"),
                (24.0, "HR", "200"),
            ],
            source_sha256="",
            source_bytes=0,
        )
        events, qc = normalize_records([record], vars_, ranges)
        hourly = _aggregate_hourly(events)
        means = hourly.set_index("var")["val"]
        self.assertEqual(float(means["hr"]), 80.0)
        self.assertEqual(float(means["sodium"]), 140.0)
        self.assertEqual(float(means["potassium"]), 4.0)
        self.assertAlmostEqual(float(means["magnesium"]), 2.43)
        self.assertAlmostEqual(float(means["fio2"]), 40.0)
        self.assertEqual(float(means["sbp"]), 120.0)
        self.assertEqual(float(means["dbp"]), 70.0)
        self.assertEqual(float(means["map"]), 80.0)
        self.assertEqual(float(means["weight"]), 80.0)
        self.assertNotIn("hgb", set(hourly["var"]))
        self.assertNotIn("be", set(hourly["var"]))
        self.assertEqual(qc["missing_sentinel_removed_by_parameter"], {"HR": 1})
        self.assertEqual(qc["raw_events_outside_0_24h"], 1)
        self.assertEqual(qc["unit_conversion_count_by_variable"], {"fio2": 1, "magnesium": 1})

    def test_official_los_survival_consistency_rule(self):
        self.assertEqual(outcome_label_from_survival(-1, 5), 0)
        self.assertEqual(outcome_label_from_survival(575, 9), 0)
        self.assertEqual(outcome_label_from_survival(7, 9), 1)
        self.assertIsNone(outcome_label_from_survival(1, 5))


if __name__ == "__main__":
    unittest.main()
