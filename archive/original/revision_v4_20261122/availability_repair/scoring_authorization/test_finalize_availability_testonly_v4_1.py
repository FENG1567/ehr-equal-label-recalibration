from __future__ import annotations

import sys
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import finalize_availability_testonly_v4_1 as finalizer


class TestAvailabilityTestOnlyAuthorization(unittest.TestCase):
    def setUp(self):
        n = 5
        V = np.zeros((n, 128), dtype=np.int16)
        B = np.zeros_like(V)
        H = np.zeros_like(V)
        L = np.array([1, 1, 1, 0, 1], dtype=np.int16)
        V[0, 0], B[0, 0], H[0, 0] = 1, 1, 0
        V[1, 0], B[1, 0], H[1, 0] = 2, 4, 2
        V[2, 0], B[2, 0], H[2, 0] = 3, 6, 4
        V[4, 0], B[4, 0], H[4, 0] = 5, 10, 23
        self.seq = {
            'V': V, 'B': B, 'H': H, 'L': L,
            'stay_id': np.arange(101, 106, dtype=np.int64),
            'subject': np.array(['p0', 'p1', 'p2', 'p3', 'p4']),
            'y_mort': np.array([0., 1., 1., 0., np.nan]),
        }
        self.split = np.array([0, 1, 2, 2, 2], dtype=np.int8)
        self.source = pd.DataFrame({
            'stay_id': self.seq['stay_id'],
            'subject': self.seq['subject'],
            'eligible_24h': [True, True, True, True, False],
            'temporal_verified': [True, True, True, True, False],
            'y_mort': self.seq['y_mort'],
            'endpoint': ['hospital'] * 5,
        })
        self.spec = {
            'db': 'mimic_available_by24_v4_1',
            'selection_by24_verified': True,
            'analysis_role': 'same_original_heldout_test_only_score_sensitivity',
            'input_scope': 'documented_timestamp_H24_proxy_with_new_random_sampling',
        }

    def test_scope_record_requires_test_only_and_exact_roster(self):
        record = (Path(__file__).resolve().parents[2] / 'admin' /
                  'availability_controls_scope_record_20261123.md')
        if not record.is_file():
            self.skipTest('external scope record is not bundled with this archive')
        finalizer.verify_scope_record(record.read_text(encoding='utf-8'))
        with self.assertRaises(ValueError):
            finalizer.verify_scope_record('SCOPE: TEST_ONLY')

    def test_selector_excludes_train_validation_and_empty_test(self):
        base = self.source.eligible_24h.to_numpy() & self.source.temporal_verified.to_numpy()
        got = finalizer.select_test_rows(base, self.split, self.seq['L'])
        self.assertEqual(got.tolist(), [False, False, True, False, False])

    def test_whitelist_flags_only_original_test_row_and_omits_readiness_aliases(self):
        base = self.source.eligible_24h.to_numpy() & self.source.temporal_verified.to_numpy()
        selected = finalizer.select_test_rows(base, self.split, self.seq['L'])
        table = finalizer.build_whitelist_table(self.source, self.seq, self.split, selected, self.spec)
        finalizer.validate_flags(table, selected, 'synthetic')
        self.assertTrue(table.loc[2, 'eligible_24h'])
        self.assertEqual(table.loc[[0, 1, 3, 4], 'eligible_24h'].tolist(), [False] * 4)
        self.assertEqual(table['reason'].tolist(), [
            'excluded_train_split', 'excluded_validation_split',
            'authorized_original_test_split_score_only',
            'excluded_empty_variant_sequence', 'excluded_original_source_gate'])
        self.assertFalse(finalizer.FORBIDDEN_READINESS_ALIASES & set(table.columns))
        self.assertFalse(table['clinical_input_availability_verified'].any())
        self.assertFalse(table['ready_for_future24h_clinical_claim'].any())

    def test_fixed_selection_semantics_remain_selection_unverified(self):
        base = self.source.eligible_24h.to_numpy() & self.source.temporal_verified.to_numpy()
        selected = finalizer.select_test_rows(base, self.split, self.seq['L'])
        fixed_spec = dict(self.spec, db='mimic_available_by24_fixedselection_v4_1',
                          selection_by24_verified=False,
                          input_scope='controlled_retrospective_fixed_original_key_perturbation')
        table = finalizer.build_whitelist_table(self.source, self.seq, self.split, selected, fixed_spec)
        self.assertFalse(table['selection_by24_verified'].any())
        self.assertFalse(finalizer.FORBIDDEN_READINESS_ALIASES & set(table.columns))

    def test_label_mismatch_is_rejected(self):
        base = self.source.eligible_24h.to_numpy() & self.source.temporal_verified.to_numpy()
        selected = finalizer.select_test_rows(base, self.split, self.seq['L'])
        bad_source = self.source.copy()
        bad_source.loc[2, 'y_mort'] = 0
        with self.assertRaisesRegex(ValueError, 'y_mort'):
            finalizer.build_whitelist_table(bad_source, self.seq, self.split, selected, self.spec)
        good = finalizer.build_whitelist_table(self.source, self.seq, self.split, selected, self.spec)
        good.loc[2, 'y_mort'] = 0
        with self.assertRaisesRegex(ValueError, 'actual labels'):
            finalizer.validate_flags(good, selected, 'synthetic')

    def test_split_mismatch_and_train_validation_authorization_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'split codes'):
            finalizer.assert_split_alignment(self.split, np.array([2, 1, 2, 2, 2]), 'synthetic')
        base = self.source.eligible_24h.to_numpy() & self.source.temporal_verified.to_numpy()
        selected = finalizer.select_test_rows(base, self.split, self.seq['L'])
        table = finalizer.build_whitelist_table(self.source, self.seq, self.split, selected, self.spec)
        table.loc[0, 'eligible_24h'] = True
        with self.assertRaisesRegex(ValueError, 'train/validation'):
            finalizer.validate_flags(table, selected, 'synthetic')
        table = finalizer.build_whitelist_table(self.source, self.seq, self.split, selected, self.spec)
        table.loc[2, 'split'] = 0
        with self.assertRaisesRegex(ValueError, 'non-test split'):
            finalizer.validate_flags(table, selected, 'synthetic')

    def test_subject_and_label_identity_mismatch_are_rejected(self):
        frame = pd.DataFrame({
            'stay_id': self.seq['stay_id'],
            'subject': self.seq['subject'],
            'y_mort': self.seq['y_mort'],
        })
        finalizer.assert_identity_and_labels(frame, self.seq['stay_id'], self.seq['subject'], self.seq['y_mort'], 'synthetic')
        frame.loc[1, 'subject'] = 'wrong-patient'
        with self.assertRaisesRegex(ValueError, 'subject IDs'):
            finalizer.assert_identity_and_labels(frame, self.seq['stay_id'], self.seq['subject'], self.seq['y_mort'], 'synthetic')
        frame.loc[1, 'subject'] = self.seq['subject'][1]
        frame.loc[1, 'y_mort'] = 0
        with self.assertRaisesRegex(ValueError, 'y_mort'):
            finalizer.assert_identity_and_labels(frame, self.seq['stay_id'], self.seq['subject'], self.seq['y_mort'], 'synthetic')

    def test_sequence_bin_contract_rejects_invalid_active_bin(self):
        finalizer.assert_sequence_contract(self.seq, 'synthetic')
        bad = dict(self.seq)
        bad['B'] = self.seq['B'].copy()
        bad['B'][2, 0] = 11
        with self.assertRaisesRegex(ValueError, 'frozen bin'):
            finalizer.assert_sequence_contract(bad, 'synthetic')


if __name__ == '__main__':
    unittest.main(verbosity=2)
