"""Outcome-blind token thinning and weighting helpers for the density readout."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


BASE_SEED = 20261122


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_ids(db: str, rows: np.ndarray, stay_id: np.ndarray,
               subject: np.ndarray) -> str:
    digest = hashlib.sha256(db.encode("utf-8"))
    digest.update(np.asarray(rows, dtype=np.int64).tobytes())
    for array in (np.asarray(stay_id).astype(str), np.asarray(subject).astype(str)):
        for value in array:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def derived_seed(namespace: str, stay_id: str, base_seed: int = BASE_SEED) -> int:
    """Stable per-stay/view seed; labels are deliberately not an input."""
    payload = f"density-v1\0{int(base_seed)}\0{namespace}\0{stay_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def thin_triplets(v: np.ndarray, b: np.ndarray, h: np.ndarray, length: int,
                  retention: float, seed: int) -> tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, int]:
    """Retain a deterministic rounded fraction of nonpadding V/B/H tokens."""
    if not (0 < float(retention) <= 1):
        raise ValueError("retention must be in (0, 1]")
    v = np.asarray(v)
    b = np.asarray(b)
    h = np.asarray(h)
    length = int(length)
    if v.ndim != 1 or b.shape != v.shape or h.shape != v.shape:
        raise ValueError("V, B, and H must be aligned one-dimensional arrays")
    if length < 1 or length > len(v):
        raise ValueError("length must describe at least one in-bounds sequence position")
    valid = np.flatnonzero(v[:length] != 0)
    if valid.size == 0:
        raise ValueError("sequence has no nonpadding variable token")
    keep_n = max(1, int(np.floor(valid.size * float(retention) + 0.5)))
    if keep_n == valid.size:
        chosen = valid
    else:
        chosen = np.sort(np.random.default_rng(int(seed)).choice(
            valid, size=keep_n, replace=False))
    out_v = np.zeros_like(v)
    out_b = np.zeros_like(b)
    out_h = np.zeros_like(h)
    out_v[chosen] = v[chosen]
    out_b[chosen] = b[chosen]
    out_h[chosen] = h[chosen]
    return out_v, out_b, out_h, int(keep_n)


def equal_per_stay_view_weights(n_stays: int, n_views: int) -> np.ndarray:
    """View-major weights whose sum over views is exactly one per stay."""
    n_stays, n_views = int(n_stays), int(n_views)
    if n_stays < 1 or n_views < 1:
        raise ValueError("n_stays and n_views must be positive")
    weights = np.full(n_stays * n_views, 1.0 / n_views, dtype=np.float64)
    if not np.allclose(weights.reshape(n_views, n_stays).sum(axis=0), 1.0):
        raise AssertionError("per-stay view weights do not sum to one")
    return weights


def partition_masks(split: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return exact masks for the frozen source train/tune/report split IDs."""
    split = np.asarray(split)
    if split.ndim != 1 or not np.isin(split, (0, 1, 2)).all():
        raise ValueError("split IDs must be a one-dimensional subset of 0, 1, and 2")
    masks = tuple(split == value for value in (0, 1, 2))
    if any(not mask.any() for mask in masks):
        raise ValueError("all three frozen split partitions must be present")
    if np.any(masks[0] & masks[1]) or np.any(masks[0] & masks[2]) or np.any(masks[1] & masks[2]):
        raise AssertionError("partition masks overlap")
    return masks


def fit_training_scaler(X: np.ndarray, train_mask: np.ndarray,
                        sample_weight: np.ndarray | None = None):
    """Fit StandardScaler on explicitly selected training rows only."""
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X)
    train_mask = np.asarray(train_mask, dtype=bool)
    if X.ndim != 2 or len(X) != len(train_mask) or not train_mask.any():
        raise ValueError("scaler matrix and nonempty training mask must align")
    if not np.isfinite(X[train_mask]).all():
        raise ValueError("training scaler inputs must be finite")
    weights = None
    if sample_weight is not None:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.shape != (len(X),) or np.any(sample_weight < 0):
            raise ValueError("scaler weights must be nonnegative and row-aligned")
        weights = sample_weight[train_mask]
    return StandardScaler().fit(X[train_mask], sample_weight=weights)


def select_min_validation(grid: tuple[float, ...], objectives: dict[float, float]) -> float:
    """Minimize the validation objective and prefer smaller C on exact ties."""
    if set(map(float, grid)) != set(map(float, objectives)):
        raise ValueError("validation objective keys must exactly match the C grid")
    if not all(np.isfinite(float(value)) for value in objectives.values()):
        raise ValueError("validation objectives must be finite")
    return float(min(grid, key=lambda C: (float(objectives[C]), float(C))))


def write_selection_npz(path: Path, **arrays) -> None:
    """Write source selection arrays with Unicode IDs and an atomic replace."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(arrays)
    for key in ("stay_id", "subject"):
        if key not in payload:
            raise ValueError(f"selection artifact is missing {key}")
        payload[key] = np.asarray(payload[key], dtype=str)
        if payload[key].dtype.kind != "U":
            raise ValueError(f"selection {key} IDs must serialize as Unicode")
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **payload)
    os.replace(temporary, path)
    with np.load(path, allow_pickle=False) as check:
        for key in ("stay_id", "subject"):
            if check[key].dtype.kind != "U":
                raise RuntimeError(f"serialized selection {key} IDs are not Unicode")


def load_selection_artifact(run_dir: Path) -> tuple[dict, dict[str, np.ndarray]]:
    """Strictly load a prepared selection for both encoder and readout consumers."""
    run_dir = Path(run_dir).resolve(strict=True)
    identity_path = run_dir / "input_identity.json"
    selection_path = run_dir / "selection.npz"
    identity = json.loads(identity_path.read_text())
    if Path(identity.get("run_directory", "")).resolve() != run_dir:
        raise RuntimeError("selection identity points to another run directory")
    if identity.get("external_outcomes_or_INSPIRE_scores_read") is not False:
        raise RuntimeError("selection identity does not certify source-only development")
    if sha256_file(selection_path) != identity.get("selection_npz_sha256"):
        raise RuntimeError("selection artifact hash differs from its run identity")
    needed = {"rows", "stay_id", "subject", "y", "split", "token_count", "real"}
    with np.load(selection_path, allow_pickle=False) as archive:
        if set(archive.files) != needed:
            raise RuntimeError("selection artifact keys differ from the frozen schema")
        data = {key: np.asarray(archive[key]) for key in needed}
    rows = np.asarray(data["rows"], dtype=np.int64)
    stay_id = np.asarray(data["stay_id"])
    subject = np.asarray(data["subject"])
    y = np.asarray(data["y"], dtype=np.int8)
    split = np.asarray(data["split"], dtype=np.int8)
    token_count = np.asarray(data["token_count"], dtype=np.int32)
    real = np.asarray(data["real"], dtype=np.float32)
    n = len(rows)
    if (stay_id.dtype.kind != "U" or subject.dtype.kind != "U"
            or stay_id.shape != (n,) or subject.shape != (n,)
            or y.shape != (n,) or split.shape != (n,) or token_count.shape != (n,)
            or real.ndim != 2 or len(real) != n or n == 0):
        raise RuntimeError("selection arrays have invalid types or misaligned rows")
    if (len(np.unique(rows)) != n or len(np.unique(stay_id)) != n
            or not np.isin(y, (0, 1)).all() or not np.isin(split, (0, 1, 2)).all()
            or np.any(token_count < 1) or not np.isfinite(real).all()):
        raise RuntimeError("selection contains duplicate IDs, invalid labels, or unusable rows")
    if hashlib.sha256(rows.tobytes()).hexdigest() != identity.get("selected_rows_sha256"):
        raise RuntimeError("selection row-index digest mismatch")
    if sha256_ids("mimic", rows, stay_id, subject) != identity.get("selected_identity_sha256"):
        raise RuntimeError("selection row/patient identity digest mismatch")
    data.update(rows=rows, stay_id=stay_id, subject=subject, y=y,
                split=split, token_count=token_count, real=real)
    return identity, data


def validate_run_dir(path: Path, density_root: Path) -> Path:
    """Require one explicit, dedicated child directory for a density run."""
    run_dir = Path(path).resolve()
    root = Path(density_root).resolve()
    if run_dir == root or root not in run_dir.parents:
        raise ValueError("run directory must be a dedicated child of REV/density_repair")
    return run_dir


def verify_development_code_hashes(run_dir: Path, identity: dict) -> None:
    """Reject code drift between prep, encoder, and readout stages."""
    files = {
        "density_repair_helpers_sha256": "density_repair.py",
        "fit_readout_sha256": "fit_readout.py",
        "encode_views_sha256": "encode_views.py",
        "launcher_sha256": "run_density_repair.sh",
    }
    expected = identity.get("development_code_hashes")
    if not isinstance(expected, dict) or set(expected) != set(files):
        raise RuntimeError("run identity lacks the complete development-code hash set")
    for key, filename in files.items():
        if sha256_file(Path(run_dir) / filename) != expected[key]:
            raise RuntimeError(f"development code changed between stages: {filename}")
