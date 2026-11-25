#!/usr/bin/env python3
"""Encode outcome-blind retained-token views using the frozen main20k encoder."""
from __future__ import annotations

import hashlib
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
PROC = ROOT / "data/processed"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from density_repair import (BASE_SEED, derived_seed, load_selection_artifact,
                            thin_triplets, validate_run_dir,
                            verify_development_code_hashes)  # noqa: E402


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_selection_for_encoding(run_dir: Path):
    """The exact validated selection loader the GPU path calls before model load."""
    return load_selection_artifact(run_dir)


def _verify_identity_files(run_dir: Path, identity: dict) -> None:
    file_hashes = {
        "source_npz_sha256": PROC / "seq_mimic.npz",
        "split_sha256": PROC / "split_mimic.npy",
        "checkpoint_sha256": ROOT / "models/main20k.pt",
        "fm_code_sha256": ROOT / "code/fm.py",
        "decoder_code_sha256": ROOT / "revision_v2_20261117/code/decoding_motor_v2.py",
        "bins_sha256": PROC / "bins.json",
        "variable_metadata_sha256": PROC / "meta_mimic.json",
        "eligibility_manifest_sha256": REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1/eligibility_manifest.json",
        "eligibility_table_sha256": REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1/mimic.parquet",
        "frozen_embedding_manifest_sha256": run_dir / "embedding_manifest_core14_20261123.json",
        "embedding_cache_sha256": Path(identity["embedding_cache_path"]),
    }
    cache_path = file_hashes["embedding_cache_sha256"].resolve()
    expected_cache_dir = (REV / "engine/cache").resolve()
    if cache_path.parent != expected_cache_dir:
        raise RuntimeError("real embedding cache path is outside the frozen engine cache")
    for key, path in file_hashes.items():
        if not path.is_file() or sha256_file(path) != identity.get(key):
            raise RuntimeError(f"source provenance changed before encoder load: {key}")
    verify_development_code_hashes(run_dir, identity)


def load_inputs(run_dir: Path):
    identity, selected = load_selection_for_encoding(run_dir)
    _verify_identity_files(run_dir, identity)
    rows = np.asarray(selected["rows"], dtype=np.int64)
    stays = np.asarray(selected["stay_id"]).astype(str)
    subjects = np.asarray(selected["subject"]).astype(str)
    y = np.asarray(selected["y"], dtype=np.int8)
    split = np.asarray(selected["split"], dtype=np.int8)
    real = np.asarray(selected["real"], dtype=np.float32)
    with np.load(PROC / "seq_mimic.npz", allow_pickle=True) as seq:
        source = {key: seq[key] for key in
                  ("V", "B", "H", "L", "stay_id", "subject", "y_mort")}
    split_all = np.load(PROC / "split_mimic.npy")
    if (not np.array_equal(np.asarray(source["stay_id"]).astype(str)[rows], stays)
            or not np.array_equal(np.asarray(source["subject"]).astype(str)[rows], subjects)
            or not np.array_equal(np.asarray(source["y_mort"], dtype=np.int8)[rows], y)
            or not np.array_equal(np.asarray(source["L"])[rows] > 0,
                                  np.ones(len(rows), dtype=bool))
            or not np.array_equal(split_all[rows].astype(np.int8), split)
            or real.ndim != 2 or not np.isfinite(real).all()):
        raise RuntimeError("prepared rows do not align to the original MIMIC sequence")
    source["split"] = split_all
    return identity, source, rows, stays, subjects, y, split, real


def _view_batch(source: dict[str, np.ndarray], rows: np.ndarray,
                stays: np.ndarray, retention: float, namespace: str):
    lengths = np.asarray(source["L"])[rows].astype(np.int64)
    width = max(1, int(lengths.max()))
    arrays = [np.zeros((len(rows), width), dtype=np.int64) for _ in range(3)]
    kept = np.zeros(len(rows), dtype=np.int32)
    source_arrays = [source[key] for key in ("V", "B", "H")]
    for i, (row, length, stay) in enumerate(zip(rows, lengths, stays)):
        token_arrays = [np.asarray(values[row, :width], dtype=np.int64)
                        for values in source_arrays]
        seed = derived_seed(namespace, str(stay), BASE_SEED)
        thinned = thin_triplets(*token_arrays, int(length), float(retention), seed)
        for dest, values in zip(arrays, thinned[:3]):
            dest[i] = values
        kept[i] = thinned[3]
    if np.any(np.count_nonzero(arrays[0], axis=1) < 1):
        raise RuntimeError("token thinning produced an unusable row")
    return arrays, kept


def _encode(model, torch, source, rows, stays, retention, namespace,
            batch_size: int, device: str = "cuda"):
    output = np.empty((len(rows), int(model.d)), dtype=np.float32)
    kept_all = np.empty(len(rows), dtype=np.int32)
    model.eval()
    with torch.inference_mode():
        for start in range(0, len(rows), batch_size):
            take_rows = rows[start:start + batch_size]
            take_stays = stays[start:start + batch_size]
            (V, B, H), kept = _view_batch(source, take_rows, take_stays,
                                          retention, namespace)
            Vt, Bt, Ht = (torch.from_numpy(array).to(device) for array in (V, B, H))
            embedding = model.embed(Vt, Bt, Ht, Vt == 0).detach().cpu().numpy()
            if embedding.shape != (len(take_rows), int(model.d)) or not np.isfinite(embedding).all():
                raise FloatingPointError("encoder returned malformed or nonfinite view embeddings")
            output[start:start + len(take_rows)] = embedding.astype(np.float32, copy=False)
            kept_all[start:start + len(take_rows)] = kept
    return output, kept_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    run_dir = validate_run_dir(args.run_dir, REV / "density_repair")
    import torch

    start = time.time()
    output_path = run_dir / "view_embeddings.npz"
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite {output_path}")
    identity, source, rows, stays, subjects, y, split, real = load_inputs(run_dir)
    if not torch.cuda.is_available():
        raise RuntimeError("verified GPU runtime reports CUDA unavailable")
    decoder_dir = ROOT / "revision_v2_20261117/code"
    sys.path.insert(0, str(decoder_dir))
    import decoding_motor_v2 as decoder

    model, checkpoint_hash = decoder.make_model("main20k", "cuda")
    # legacy fm sets 16 threads during import; restore the verified eight-thread runtime.
    torch.set_num_threads(8)
    if checkpoint_hash != identity["checkpoint_sha256"]:
        raise RuntimeError("loaded main20k checkpoint does not match prepared provenance")
    if torch.get_num_threads() != 8:
        raise RuntimeError("Torch thread count did not reset to eight after legacy fm import")
    train = split == 0
    valid = split == 1
    report = split == 2
    outputs = {}
    kept_outputs = {}
    for i in range(3):
        outputs[f"train_aug_{i}"] , kept_outputs[f"kept_train_{i}"] = _encode(
            model, torch, source, rows[train], stays[train], 0.5,
            f"train_aug_view_{i}", 128)
        outputs[f"val_aug_{i}"], kept_outputs[f"kept_val_{i}"] = _encode(
            model, torch, source, rows[valid], stays[valid], 0.5,
            f"validation_heldout_view_{i}", 128)
    for i, retention in enumerate((0.75, 0.5, 0.25)):
        outputs[f"test_ret_{i}"], kept_outputs[f"kept_test_{i}"] = _encode(
            model, torch, source, rows[report], stays[report], retention,
            f"report_retention_{retention:.2f}", 128)
    train_aug = np.stack([outputs[f"train_aug_{i}"] for i in range(3)])
    val_aug = np.stack([outputs[f"val_aug_{i}"] for i in range(3)])
    test_ret = np.stack([outputs[f"test_ret_{i}"] for i in range(3)])
    kept_train = np.stack([kept_outputs[f"kept_train_{i}"] for i in range(3)])
    kept_val = np.stack([kept_outputs[f"kept_val_{i}"] for i in range(3)])
    kept_test = np.stack([kept_outputs[f"kept_test_{i}"] for i in range(3)])
    runtime = {
        "python": sys.version.split()[0], "torch": str(torch.__version__),
        "torch_cuda": getattr(torch.version, "cuda", None),
        "device": torch.cuda.get_device_name(torch.cuda.current_device()),
        "device_capability": list(torch.cuda.get_device_capability()),
        "torch_num_threads": torch.get_num_threads(), "encoding_dtype": "float32",
        "max_memory_allocated_bytes": int(torch.cuda.max_memory_allocated()),
        "max_memory_reserved_bytes": int(torch.cuda.max_memory_reserved()),
        "elapsed_seconds": round(time.time() - start, 3),
        "input_rows": {"train": int(train.sum()), "validation": int(valid.sum()),
                       "report": int(report.sum())},
    }
    with output_path.open("xb") as handle:
        np.savez_compressed(handle, rows=rows, train_aug=train_aug, val_aug=val_aug,
                            test_ret=test_ret, kept_train=kept_train,
                            kept_val=kept_val, kept_test=kept_test,
                            gpu_runtime_json=np.asarray(json.dumps(runtime, sort_keys=True)))
    print("ENCODE_PASS", json.dumps({"output": str(output_path),
          "sha256": sha256_file(output_path), "runtime": runtime}, sort_keys=True),
          flush=True)


if __name__ == "__main__":
    main()
