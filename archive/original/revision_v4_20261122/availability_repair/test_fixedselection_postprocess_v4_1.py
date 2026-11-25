"""Synthetic tests for fixed-old-token H24 postprocessing."""
import unittest

import numpy as np
import pandas as pd

import fixedselection_postprocess_v4_1 as post


def digitize(values, variable, bins):
    return np.clip(np.digitize(values, bins.get(variable, []), right=False) + 1,
                   1, post.NBINS).astype(np.int16)


def old_sequence():
    v = np.zeros((2, post.MAXLEN), dtype=np.int16)
    b = np.zeros_like(v)
    h = np.zeros_like(v)
    v[0, :3] = [1, 2, 3]
    b[0, :3] = [1, 1, 1]
    h[0, :3] = [0, 1, 2]
    return {'V': v, 'B': b, 'H': h, 'L': np.array([3, 0], dtype=np.int16),
            'stay_id': np.array([11, 12], dtype=np.int64)}


class FixedSelectionTest(unittest.TestCase):
    def setUp(self):
        self.variables = ['a', 'b', 'c']
        self.bins = {'a': [0.5], 'b': [5.0], 'c': []}

    def test_unchanged_hourly_inputs_replay_original_sequence_exactly(self):
        old = old_sequence()
        new_hourly = pd.DataFrame({
            'stay_id': [11, 11, 11], 'hb': [0, 1, 2], 'vid': [0, 1, 2],
            'val': [0.25, 2.0, 7.0],
        })
        v, b, h, length, mapping = post.build_fixed_sequence(
            old, new_hourly, self.variables, self.bins, digitize)
        np.testing.assert_array_equal(v, old['V'])
        np.testing.assert_array_equal(b, old['B'])
        np.testing.assert_array_equal(h, old['H'])
        np.testing.assert_array_equal(length, old['L'])
        self.assertEqual(mapping.status.tolist(), ['retained', 'retained', 'retained'])
        self.assertEqual(mapping.new_position.tolist(), [0, 1, 2])

    def test_unavailable_keys_removed_stably_and_new_unsampled_keys_not_added(self):
        old = old_sequence()
        new_hourly = pd.DataFrame({
            'stay_id': [11, 11, 11], 'hb': [0, 1, 3], 'vid': [0, 1, 2],
            'val': [0.25, 8.0, 9.0],
        })
        v, b, h, length, mapping = post.build_fixed_sequence(
            old, new_hourly, self.variables, self.bins, digitize)
        self.assertEqual(length.tolist(), [2, 0])
        self.assertEqual(v[0, :2].tolist(), [1, 2])
        self.assertEqual(h[0, :2].tolist(), [0, 1])
        self.assertEqual(b[0, :2].tolist(), [1, 2])
        self.assertEqual(mapping.status.tolist(), ['retained', 'retained', 'removed_unavailable_key'])
        self.assertEqual(mapping.new_position.tolist()[:2], [0, 1])
        self.assertTrue(pd.isna(mapping.new_position.iloc[2]))
        self.assertTrue(np.all(v[1] == 0) and np.all(b[1] == 0) and np.all(h[1] == 0))

    def test_empty_available_input_removes_all_tokens_and_keeps_padding(self):
        old = old_sequence()
        empty = pd.DataFrame(columns=['stay_id', 'hb', 'vid', 'val'])
        v, b, h, length, mapping = post.build_fixed_sequence(
            old, empty, self.variables, self.bins, digitize)
        self.assertEqual(length.tolist(), [0, 0])
        self.assertEqual(mapping.status.tolist(), ['removed_unavailable_key'] * 3)
        self.assertTrue(np.all(v == 0) and np.all(b == 0) and np.all(h == 0))

    def test_hourly_comparison_identifies_exact_unchanged_inputs(self):
        old = pd.DataFrame({'stay_id': [11, 11], 'hb': [0, 1], 'vid': [0, 1], 'val': [1., 2.]})
        same = old.copy()
        summary, per_stay = post.compare_hourly(old, same, np.array([11, 12]), np.array(['s1', 's2']))
        self.assertEqual(summary['stays_with_identical_hourly_keys_and_values'], 2)
        self.assertTrue(per_stay.identical_available_hourly_inputs.all())
        changed = pd.DataFrame({'stay_id': [11, 11], 'hb': [0, 1], 'vid': [0, 1], 'val': [1.5, 2.]})
        summary, per_stay = post.compare_hourly(old, changed, np.array([11, 12]), np.array(['s1', 's2']))
        self.assertEqual(summary['stays_with_identical_hourly_keys_and_values'], 1)
        self.assertFalse(bool(per_stay.loc[per_stay.stay_id.eq(11), 'identical_available_hourly_inputs'].iloc[0]))


if __name__ == '__main__':
    unittest.main(verbosity=2)
