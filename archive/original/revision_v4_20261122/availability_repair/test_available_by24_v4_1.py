"""Synthetic integration checks for the H24 source filters and aggregation."""
import gzip
import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

import build_available_by24_v4_1 as etl


class AvailabilitySensitivityTest(unittest.TestCase):
    def test_measurement_h24_future_and_missing_storetime_excluded(self):
        intime = pd.Timestamp('2026-01-01 08:00:00')
        measured = pd.Series([intime + pd.Timedelta(hours=h) for h in [1, 2, 23.5, 24]])
        stored = pd.Series([intime + pd.Timedelta(hours=25), pd.NaT,
                            intime + pd.Timedelta(hours=24), intime + pd.Timedelta(hours=24)])
        _, window, late, unknown, allowed = etl.measurement_availability(
            measured, stored, pd.Series([intime] * 4))
        self.assertEqual(window.tolist(), [True, True, True, False])
        self.assertEqual(late.tolist(), [True, False, False, False])
        self.assertEqual(unknown.tolist(), [False, True, False, False])
        self.assertEqual(allowed.tolist(), [False, False, True, False])

    def test_therapy_carryover_requires_documented_ongoing_overlap(self):
        intime = pd.Timestamp('2026-01-01 08:00:00')
        starts = pd.Series([intime - pd.Timedelta(hours=1),
                            intime - pd.Timedelta(hours=2),
                            intime - pd.Timedelta(hours=3),
                            intime + pd.Timedelta(hours=23.5),
                            intime + pd.Timedelta(hours=24)])
        ends = pd.Series([intime + pd.Timedelta(minutes=1),
                          intime - pd.Timedelta(minutes=1), pd.NaT,
                          intime + pd.Timedelta(hours=25), intime + pd.Timedelta(hours=25)])
        stored = pd.Series([intime + pd.Timedelta(hours=24)] * 5)
        result = etl.therapy_availability(starts, ends, stored, pd.Series([intime] * 5))
        self.assertEqual(result['allowed'].tolist(), [True, False, False, True, False])
        self.assertEqual(result['allowed_pre_icu'].tolist(), [True, False, False, False, False])
        self.assertEqual(result['hour'].tolist()[0], 0.0)
        self.assertEqual(result['ended_before_icu'].tolist(), [False, True, False, False, False])
        self.assertEqual(result['unknown_end'].tolist(), [False, False, True, False, False])

    def test_nonfinite_then_range_filter(self):
        values = pd.Series([1.0, np.inf, np.nan, -0.1, 10.0])
        _, finite, valid = etl.finite_range_mask(values, (0, 10))
        self.assertEqual(finite.tolist(), [True, False, False, True, True])
        self.assertEqual(valid.tolist(), [True, False, False, False, True])

    def test_frozen_label_check_preserves_paired_nan_and_rejects_changes(self):
        frozen = np.array([1.0, np.nan, 0.0], dtype=np.float32)
        self.assertTrue(etl.same_frozen_labels(np.array([1.0, np.nan, 0.0], dtype=np.float32), frozen))
        self.assertFalse(etl.same_frozen_labels(np.array([1.0, np.nan, 0.5], dtype=np.float32), frozen))
        self.assertFalse(etl.same_frozen_labels(np.array([1.0, 0.0, np.nan], dtype=np.float32), frozen))

    def test_gcs_requires_all_three_same_time_h24_components(self):
        t0 = pd.Timestamp('2026-01-01 09:00:00')
        t1 = pd.Timestamp('2026-01-01 10:00:00')
        components = pd.DataFrame({
            'stay_id': [1, 1, 1, 1, 1],
            'charttime': [t0, t0, t0, t1, t1],
            'itemid': [220739, 223900, 223901, 220739, 223901],
            'val': [4.0, 5.0, 6.0, 4.0, 6.0],
        })
        events, summary = etl.gcs_total_events(components, {1: pd.Timestamp('2026-01-01 08:00:00')}, (3, 15))
        self.assertEqual(len(events), 1)
        self.assertEqual(float(events.iloc[0]['val']), 15.0)
        self.assertEqual(summary['complete_component_timestamps'], 1)
        self.assertEqual(summary['incomplete_component_timestamps'], 1)

    def test_hourly_aggregate_has_only_observed_string_groups(self):
        events = pd.DataFrame({
            'stay_id': [1, 1], 'hour': [0.2, 0.8], 'var': ['hr', 'hr'], 'val': [60., 80.]
        })
        events['var'] = pd.Categorical(events['var'], categories=['hr', 'sbp', 'dbp'])
        grouped = etl.hourly_aggregate(events, {'hr': 0, 'sbp': 1, 'dbp': 2})
        self.assertEqual(len(grouped), 1)
        self.assertEqual(grouped['var'].tolist(), ['hr'])
        self.assertAlmostEqual(float(grouped.iloc[0]['val']), 70.)

    def test_one_pass_gzip_reader_hashes_synthetic_raw_file(self):
        content = b'stay_id,itemid,charttime,storetime,valuenum\n1,220045,2026-01-01 09:00:00,2026-01-01 10:00:00,60\n'
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'synthetic.csv.gz'
            with gzip.open(path, 'wb') as stream:
                stream.write(content)
            raw, gz, text, reader = etl._source_csv_chunks(
                path, ['stay_id', 'itemid', 'charttime', 'storetime', 'valuenum'],
                {'stay_id': 'Int64', 'itemid': 'Int64'})
            try:
                chunks = list(reader)
                self.assertEqual(int(chunks[0].iloc[0]['valuenum']), 60)
            finally:
                text.close()
                gz.close()
                raw.close()
            expected = hashlib.sha256(path.read_bytes()).hexdigest()
            self.assertEqual(raw.digest.hexdigest(), expected)
            self.assertEqual(raw.bytes_read, path.stat().st_size)


if __name__ == '__main__':
    unittest.main(verbosity=2)
