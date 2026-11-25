"""Deterministic same-information summaries of v4 binned V/B/H/L inputs."""
from __future__ import annotations

import numpy as np


VALUE_PATTERN_NAMES = (
    "observed", "count_per_128", "mean_bin_per_10", "min_bin_per_10",
    "max_bin_per_10", "last_bin_per_10", "first_hour_per_23",
    "last_hour_per_23", "mean_hour_per_23",
)
PATTERN_NAMES = (
    "observed", "count_per_128", "first_hour_per_23",
    "last_hour_per_23", "mean_hour_per_23",
)


def summarize_inputs(ds: dict[str, np.ndarray], rows: np.ndarray, n_variables: int
                    ) -> tuple[np.ndarray, np.ndarray]:
    """Return value+pattern and pattern-only dense float32 summaries.

    The canonical variable token IDs are 1-based; token ID 0 is padding. Only
    positions before each row's ``L`` count are considered. For a repeated
    variable at the maximum hour, the last bin uses the last token in source
    sequence order as a deterministic tie break. Missing continuous summaries
    are zero and remain distinguishable from observed zero by ``observed``.
    """
    rows = np.asarray(rows, dtype=np.int64)
    n = len(rows)
    if n_variables < 1:
        raise ValueError("n_variables must be positive")
    if not n:
        return (np.zeros((0, n_variables * len(VALUE_PATTERN_NAMES)), dtype=np.float32),
                np.zeros((0, n_variables * len(PATTERN_NAMES)), dtype=np.float32))
    lengths = np.asarray(ds["L"], dtype=np.int64)[rows]
    if np.any(lengths < 0) or np.any(lengths > ds["V"].shape[1]):
        raise ValueError("sequence lengths are outside the V/B/H array width")
    total = int(lengths.sum())
    if total == 0:
        full = np.zeros((n, n_variables, len(VALUE_PATTERN_NAMES)), dtype=np.float32)
        pattern = full[:, :, [0, 1, 6, 7, 8]].reshape(n, -1).copy()
        return full.reshape(n, -1), pattern

    row_local = np.repeat(np.arange(n, dtype=np.int64), lengths)
    offsets = np.repeat(np.cumsum(lengths, dtype=np.int64) - lengths, lengths)
    positions = np.arange(total, dtype=np.int64) - offsets
    original_rows = rows[row_local]
    var_ids = np.asarray(ds["V"])[original_rows, positions].astype(np.int64)
    bins = np.asarray(ds["B"])[original_rows, positions].astype(np.float32)
    hours = np.asarray(ds["H"])[original_rows, positions].astype(np.float32)
    keep = var_ids > 0
    row_local, var_ids = row_local[keep], var_ids[keep]
    bins, hours = bins[keep], hours[keep]
    if not len(var_ids):
        full = np.zeros((n, n_variables, len(VALUE_PATTERN_NAMES)), dtype=np.float32)
        return full.reshape(n, -1), full[:, :, [0, 1, 6, 7, 8]].reshape(n, -1).copy()
    if np.any(var_ids > n_variables):
        raise ValueError(f"observed variable token exceeds canonical vocabulary width {n_variables}")
    if not np.isfinite(bins).all() or not np.isfinite(hours).all():
        raise ValueError("V/B/H token arrays contain nonfinite values")
    if np.any((hours < 0) | (hours > 23)):
        raise ValueError("observed hour token falls outside [0, 23]")

    var_index = var_ids - 1
    flat = row_local * n_variables + var_index
    n_cells = n * n_variables
    count = np.bincount(flat, minlength=n_cells).astype(np.float32)
    sum_bins = np.bincount(flat, weights=bins, minlength=n_cells).astype(np.float32)
    sum_hours = np.bincount(flat, weights=hours, minlength=n_cells).astype(np.float32)
    min_bin = np.full(n_cells, np.inf, dtype=np.float32)
    max_bin = np.full(n_cells, -np.inf, dtype=np.float32)
    min_hour = np.full(n_cells, np.inf, dtype=np.float32)
    max_hour = np.full(n_cells, -np.inf, dtype=np.float32)
    np.minimum.at(min_bin, flat, bins)
    np.maximum.at(max_bin, flat, bins)
    np.minimum.at(min_hour, flat, hours)
    np.maximum.at(max_hour, flat, hours)

    # Stable token order is the order in which valid source positions were flattened.
    token_order = np.arange(len(flat), dtype=np.int64)
    at_max_hour = hours == max_hour[flat]
    last_order = np.full(n_cells, -1, dtype=np.int64)
    np.maximum.at(last_order, flat[at_max_hour], token_order[at_max_hour])
    last_bin = np.zeros(n_cells, dtype=np.float32)
    active = last_order >= 0
    last_bin[active] = bins[last_order[active]]

    observed = (count > 0).astype(np.float32)
    safe_count = np.maximum(count, 1.0)
    mean_bin = sum_bins / safe_count
    mean_hour = sum_hours / safe_count
    min_bin[~active] = 0.0
    max_bin[~active] = 0.0
    min_hour[~active] = 0.0
    max_hour[~active] = 0.0
    full = np.stack([
        observed, count / 128.0, mean_bin / 10.0, min_bin / 10.0,
        max_bin / 10.0, last_bin / 10.0, min_hour / 23.0,
        max_hour / 23.0, mean_hour / 23.0,
    ], axis=1).reshape(n, n_variables, len(VALUE_PATTERN_NAMES))
    value_pattern = np.ascontiguousarray(full.reshape(n, -1), dtype=np.float32)
    pattern_only = np.ascontiguousarray(full[:, :, [0, 1, 6, 7, 8]].reshape(n, -1), dtype=np.float32)
    if not np.isfinite(value_pattern).all() or not np.isfinite(pattern_only).all():
        raise FloatingPointError("summary feature matrix contains nonfinite values")
    return value_pattern, pattern_only

