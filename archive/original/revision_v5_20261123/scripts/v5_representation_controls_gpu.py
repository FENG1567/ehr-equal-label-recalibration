#!/usr/bin/env python3
"""Produce repaired-eICU fixed-head control embeddings without fitting a readout."""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import re
import time
from pathlib import Path

for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"): os.environ[name] = "2"
import numpy as np
import v5_cache_validator as cache_validator

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
R4 = ROOT / "revision_v4_20261122"
R5 = ROOT / "revision_v5_20261123"
SEQ = R4 / "datasets/eicu_semantic_v4_2/seq_eicu_semantic_v4_2.npz"
MANIFEST = R4 / "engine/code_snapshots/embedding_manifest_recovered_v4_2_merged41_b2eec1f7aca1.json"
EXPECTED_MANIFEST = "b2eec1f7aca1857bda440ddb5ec48e477b2a20f753d5dea28b7715f7a1a243fe"
CHECKPOINT = ROOT / "models/main20k.pt"
FM_SOURCE = ROOT / "code/fm.py"
RUN_ID = os.environ.get("V5_CONTROL_RUN_ID", "")
if RUN_ID and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,95}", RUN_ID):
    raise ValueError("V5_CONTROL_RUN_ID contains unsafe path characters")
OUT = (R5 / "controls" / f"representation_controls_{RUN_ID}" if RUN_ID else
       R5 / "controls")
EMBEDDINGS = OUT / "repaired_eicu_fixed_head_embeddings.npz"
GPU_MANIFEST = OUT / "repaired_eicu_fixed_head_gpu_manifest.json"
PARITY_REPORT = OUT / "repaired_eicu_observed_runtime_parity.json"
DB = "eicu_semantic_v4_2"
EXPECTED_SEQ = "e44507da41b045b23b2242dfb2de760b9956a961bf12522c3d55c001633da4e9"
EXPECTED_CHECKPOINT = "0719db57b91b261869aed119cee95bf8980eb14411684c2040ba8588b45460bd"
PROBE = R4 / "engine/results/source_probes/main20k_real_39321cc8e6e2ad96c18d.pkl"
EXPECTED_PROBE = "25b6c9015cf73d5e0156b76197cf1101d6f6e4649407b3ba15b48b2f90a00707"
SEED_WITHIN, SEED_DONOR = 2026112301, 2026112302
BATCH = int(os.environ.get("V5_CONTROL_BATCH_SIZE", "8"))
if BATCH not in (8, 16, 32, 64, 128, 256):
    raise ValueError("unsupported inference batch size")
OBSERVED_SAMPLE_SIZE = 128
OBSERVED_EMBEDDING_ATOL = 1e-5
OBSERVED_EMBEDDING_RTOL = 1e-5
OBSERVED_PROBABILITY_ATOL = 2e-6
OBSERVED_PROBABILITY_RTOL = 1e-5
RESOURCE_SCOPE = os.environ.get("V5_RESOURCE_SCOPE", "unspecified")


def current_cgroup_path() -> str:
    for row in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        if row.startswith("0::"):
            return row.split("::", 1)[1]
    raise RuntimeError("unified cgroup path is unavailable")


RESOURCE_LIMITS = {
    "runner_memory_max_bytes": int(os.environ.get('V5_HOST_MEMORY_MAX_BYTES', str(3*1024**3))),
    "runner_memory_swap_max_bytes": 0,
    "runner_cpu_quota_percent": 200,
    "cpu_threads": 2,
    "resource_scope": RESOURCE_SCOPE,
    "gpu_allocator_cap_bytes": 6 * 1024 ** 3,
    "global_serial_lock": "R5/admin/v5_serial_science.lock",
    "gpu_lock": "R5/admin/gpu_v5.lock",
}


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json_atomic(path: Path, payload: dict) -> None:
    temporary = path.with_name("." + path.name + ".tmp")
    if path.exists() or temporary.exists():
        raise FileExistsError(f"refusing to overwrite GPU evidence: {path}")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n",
                         encoding="utf-8")
    os.replace(temporary, path)


def identity_hash(rows: np.ndarray, stays: np.ndarray, subjects: np.ndarray) -> str:
    digest = hashlib.sha256(DB.encode("utf-8"))
    digest.update(np.asarray(rows, dtype=np.int64).tobytes())
    for values in (np.asarray(stays).astype(str), np.asarray(subjects).astype(str)):
        for value in values:
            encoded = value.encode("utf-8")
            digest.update(len(encoded).to_bytes(8, "little"))
            digest.update(encoded)
    return digest.hexdigest()


def get_manifest_entry(condition: str = "real") -> dict:
    if sha256_file(MANIFEST) != EXPECTED_MANIFEST:
        raise ValueError("frozen 41-row v4.2 embedding manifest hash differs")
    document = json.loads(MANIFEST.read_text(encoding="utf-8"))
    if document.get("mode") != "encode" or len(document.get("models", [])) != 41:
        raise ValueError("expected the frozen 41-row v4.2 embedding manifest")
    matches = [row for row in document["models"]
               if row.get("db") == DB and row.get("model_tag") == "main20k"
               and row.get("input_condition") == condition]
    if len(matches) != 1:
        raise ValueError(f"manifest lacks one repaired-eICU main20k {condition} cache")
    entry = matches[0]
    if entry.get("source_npz_sha256") != EXPECTED_SEQ:
        raise ValueError("repaired-eICU real cache source hash differs")
    if entry.get("checkpoint_sha256") != EXPECTED_CHECKPOINT:
        raise ValueError("main20k checkpoint hash differs")
    return entry


def verify_constant_cache(real_entry: dict, constant_entry: dict,
                          rows: np.ndarray, stays: np.ndarray,
                          subjects: np.ndarray,
                          cache_pair_attestation: dict) -> dict:
    for field in ("source_npz_sha256", "checkpoint_sha256", "encoder_code_sha256",
                  "split_sha256", "row_identity_sha256", "n_rows"):
        if constant_entry.get(field) != real_entry.get(field):
            raise ValueError(f"constant cache differs from verified real cache in {field}")
    if (cache_pair_attestation.get("status") != "verified_real_constant_cache_pair"
            or cache_pair_attestation.get("condition_specific_input_mask_hashes_differ") is not True):
        raise ValueError("condition-specific real/constant cache attestation did not pass")
    if constant_entry.get("model_tag") != "main20k" or constant_entry.get("input_condition") != "constant":
        raise ValueError("constant arm is not the repaired main20k constant-input cache")
    cache_path = Path(str(constant_entry["cache"]))
    cache_hash = str(cache_pair_attestation["constant"]["cache_sha256"])
    if not cache_path.is_file():
        raise ValueError("repaired-eICU main20k constant cache is missing or stale")
    with np.load(cache_path, allow_pickle=False) as cache:
        constant_rows = np.asarray(cache["rows"], dtype=np.int64)
        constant_stays = cache["stay_id"].astype(str)
        constant_subjects = cache["subject"].astype(str)
    if (identity_hash(constant_rows, constant_stays, constant_subjects)
            != constant_entry.get("row_identity_sha256")
            or not np.array_equal(constant_rows, rows)
            or not np.array_equal(constant_stays, stays)
            or not np.array_equal(constant_subjects, subjects)):
        raise ValueError("constant cache identities do not match the repaired-eICU real cache roster")
    return {
        "status": "verified_existing_cache",
        "model_tag": constant_entry["model_tag"],
        "input_condition": constant_entry["input_condition"],
        "cache_path": str(cache_path),
        "cache_sha256": cache_hash,
        "source_npz_sha256": constant_entry["source_npz_sha256"],
        "checkpoint_sha256": constant_entry["checkpoint_sha256"],
        "encoder_code_sha256": constant_entry["encoder_code_sha256"],
        "row_identity_sha256": constant_entry["row_identity_sha256"],
        "same_rows_stays_subjects_as_real_cache": True,
        "reencoded_by_this_gpu_control": False,
        "cache_pair_attestation": {
            "merge_audit_sha256": cache_pair_attestation["real"]["merge_audit_sha256"],
            "real_cache_sha256": cache_pair_attestation["real"]["cache_sha256"],
            "constant_cache_sha256": cache_pair_attestation["constant"]["cache_sha256"],
            "real_input_mask_sha256": cache_pair_attestation["real"]["input_mask_sha256"],
            "constant_input_mask_sha256": cache_pair_attestation["constant"]["input_mask_sha256"],
            "condition_specific_input_mask_hashes_differ": True,
            "shared_checkpoint_sha256": real_entry["checkpoint_sha256"],
            "shared_encoder_code_sha256": real_entry["encoder_code_sha256"],
        },
    }


def load_historical_source_probe():
    if not PROBE.is_file() or sha256_file(PROBE) != EXPECTED_PROBE:
        raise ValueError("historical main20k C=1 source probe is missing or stale")
    with PROBE.open("rb") as handle:
        artifact = pickle.load(handle)
    if not isinstance(artifact, dict) or not {"scaler", "probe"}.issubset(artifact):
        raise ValueError("historical source probe artifact schema is incomplete")
    scaler, probe = artifact["scaler"], artifact["probe"]
    metadata = {key: value for key, value in artifact.items()
                if key not in ("scaler", "probe")}
    if (metadata.get("model_tag") != "main20k"
            or metadata.get("input_condition") != "real"
            or metadata.get("source_npz_sha256") !=
            "9bd20bf1394a34ba112a4de9a8d7e8971821bef90558a9987be4b9da5ae06824"):
        raise ValueError("source probe metadata does not identify the frozen MIMIC main20k real arm")
    params = probe.get_params()
    if (float(params.get("C", float("nan"))) != 1.0
            or params.get("random_state") != 20261122
            or params.get("class_weight") is not None):
        raise ValueError("observed-arm comparison requires the historical main20k C=1 source head")
    return scaler, probe


def load_cache_embedding_sample(entry: dict, positions: np.ndarray) -> np.ndarray:
    with np.load(Path(str(entry["cache"])), allow_pickle=False) as cache:
        embedding = np.asarray(cache["embedding"][positions], dtype=np.float32)
    if embedding.ndim != 2 or not np.isfinite(embedding).all():
        raise ValueError("observed real-cache embedding sample is invalid")
    return embedding


def load_inputs(entry: dict, expected_cache_sha256: str) -> tuple[np.ndarray, ...]:
    cache_path = Path(str(entry["cache"]))
    if not cache_path.is_file() or sha256_file(cache_path) != expected_cache_sha256:
        raise ValueError("repaired-eICU real cache is missing or stale")
    if sha256_file(SEQ) != EXPECTED_SEQ:
        raise ValueError("repaired-eICU sequence hash differs from locked v4.2 data")
    with np.load(cache_path, allow_pickle=False) as cache:
        rows = np.asarray(cache["rows"], dtype=np.int64)
        stays = cache["stay_id"].astype(str)
        subjects = cache["subject"].astype(str)
    if identity_hash(rows, stays, subjects) != entry.get("row_identity_sha256"):
        raise ValueError("manifest and cache row identity digests differ")
    with np.load(SEQ, allow_pickle=False) as sequence:
        if rows.ndim != 1 or len(rows) != int(entry["n_rows"]) or rows.min() < 0 or rows.max() >= len(sequence["stay_id"]):
            raise ValueError("real cache rows are outside the repaired sequence")
        stay_all = sequence["stay_id"].astype(str)
        subject_all = sequence["subject"].astype(str)
        if not np.array_equal(stay_all[rows], stays) or not np.array_equal(subject_all[rows], subjects):
            raise ValueError("cache stay/patient identities do not match the repaired sequence")
        V = np.asarray(sequence["V"][rows], dtype=np.int64)
        B = np.asarray(sequence["B"][rows], dtype=np.int64)
        H = np.asarray(sequence["H"][rows], dtype=np.int64)
        L = np.asarray(sequence["L"][rows], dtype=np.int64)
    if V.ndim != 2 or V.shape != B.shape or V.shape != H.shape or V.shape[0] != len(L):
        raise ValueError("repaired-eICU V/B/H/L shapes do not align")
    return rows, stays, subjects, V, B, H, L
def within_shuffle(V: np.ndarray, B: np.ndarray, L: np.ndarray,
                   rng: np.random.Generator) -> tuple[np.ndarray, dict, np.ndarray]:
    out = B.copy()
    changed = 0
    active = 0
    changed_by_row = np.zeros(len(L), dtype=np.int64)
    for i, length in enumerate(L):
        k = int(length)
        variables = V[i, :k]
        for vid in np.unique(variables):
            if vid == 0:
                continue
            pos = np.flatnonzero(variables == vid)
            before = B[i, pos].copy()
            out[i, pos] = rng.permutation(before)
            if not np.array_equal(np.sort(before), np.sort(out[i, pos])):
                raise AssertionError("within-variable shuffle changed a stay-variable bin multiset")
        active += int(np.count_nonzero(variables))
        row_changed = int(np.count_nonzero(out[i, :k] != B[i, :k]))
        changed_by_row[i] = row_changed
        changed += row_changed
        if (i + 1) % 20000 == 0:
            log(f"WITHIN_SHUFFLE progress={i + 1}/{len(L)}")
    if np.any(out[V == 0] != 0):
        raise AssertionError("within-variable shuffle changed padding bins")
    return out, {
        "changed_tokens": int(changed), "active_tokens": int(active),
        "changed_token_fraction": float(changed / active) if active else None,
        "seed": SEED_WITHIN,
        "preserved": "V, H, L, padding positions, and each stay-variable B-bin multiset",
    }, changed_by_row


def donor_replace(V: np.ndarray, B: np.ndarray, L: np.ndarray, subjects: np.ndarray,
                  rng: np.random.Generator) -> tuple[np.ndarray, dict, np.ndarray]:
    donor_values: dict[tuple[str, int], list[int]] = {}
    variable_subjects: dict[int, set[str]] = {}
    subject_variables: dict[str, set[int]] = {}
    for i, subject in enumerate(subjects):
        subject = str(subject)
        k = int(L[i])
        for vid in np.unique(V[i, :k]):
            if vid == 0:
                continue
            bins = B[i, :k][V[i, :k] == vid].astype(np.int64).tolist()
            donor_values.setdefault((subject, int(vid)), []).extend(bins)
            variable_subjects.setdefault(int(vid), set()).add(subject)
            subject_variables.setdefault(subject, set()).add(int(vid))
        if (i + 1) % 20000 == 0:
            log(f"DONOR_POOL progress={i + 1}/{len(subjects)}")

    sorted_subjects = {vid: sorted(ids) for vid, ids in variable_subjects.items()}
    positions = {vid: {subject: index for index, subject in enumerate(ids)}
                 for vid, ids in sorted_subjects.items()}
    donor_map: dict[tuple[str, int], str] = {}
    candidate_counts: list[int] = []
    for group_index, (target, variables) in enumerate(subject_variables.items()):
        for vid in sorted(variables):
            ids = sorted_subjects[vid]
            n_other = len(ids) - 1
            if n_other <= 0:
                raise RuntimeError(f"variable {vid} has no distinct-patient donor")
            target_position = positions[vid][target]
            draw = int(rng.integers(0, n_other))
            if draw >= target_position:
                draw += 1
            donor_map[(target, int(vid))] = ids[draw]
            candidate_counts.append(n_other)
        if (group_index + 1) % 20000 == 0:
            log(f"DONOR_ASSIGN progress={group_index + 1}/{len(subject_variables)}")
    if any(target == donor for (target, _), donor in donor_map.items()):
        raise AssertionError("self-donor detected")

    out = B.copy()
    changed = 0
    active = 0
    changed_by_row = np.zeros(len(L), dtype=np.int64)
    for i, target in enumerate(subjects):
        target = str(target)
        k = int(L[i])
        variables = V[i, :k]
        active += int(np.count_nonzero(variables))
        for vid in np.unique(variables):
            if vid == 0:
                continue
            pos = np.flatnonzero(variables == vid)
            donor = donor_map[(target, int(vid))]
            values = np.asarray(donor_values[(donor, int(vid))], dtype=np.int64)
            if values.size == 0:
                raise AssertionError("selected donor has no variable-specific bins")
            out[i, pos] = rng.choice(values, size=len(pos), replace=True)
            if not np.isin(out[i, pos], values).all():
                raise AssertionError("donor replacement assigned a bin outside variable-specific support")
        row_changed = int(np.count_nonzero(out[i, :k] != B[i, :k]))
        changed_by_row[i] = row_changed
        changed += row_changed
        if (i + 1) % 20000 == 0:
            log(f"DONOR_REPLACE progress={i + 1}/{len(subjects)}")
    if np.any(out[V == 0] != 0):
        raise AssertionError("donor replacement changed padding bins")
    return out, {
        "changed_tokens": int(changed), "active_tokens": int(active),
        "changed_token_fraction": float(changed / active) if active else None,
        "seed": SEED_DONOR, "donor_coverage_fraction": 1.0,
        "target_subject_variable_groups": int(len(donor_map)),
        "unique_donor_subjects": int(len(set(donor_map.values()))),
        "self_donor_groups": 0, "no_other_donor_groups": 0,
        "donor_candidates_min": int(min(candidate_counts)),
        "donor_candidates_median": float(np.median(candidate_counts)),
        "donor_candidates_max": int(max(candidate_counts)),
        "donor_assignment": "one uniform distinct patient per target patient-variable; donor B bins sampled with replacement",
        "outcome_used_for_donor_assignment": False,
        "donor_status_used": False,
        "preserved": "V, H, L, padding positions, and variable-specific observed B-bin support",
        "interpretation_limit": "breaks temporal and joint physiology; bins can be marginally valid while combinations are out of distribution",
    }, changed_by_row
def main() -> None:
    parity_only = os.sys.argv[1:] == ["--observed-parity-only"]
    if os.sys.argv[1:] not in ([], ["--observed-parity-only"]):
        raise ValueError("supported argument: --observed-parity-only")
    if not parity_only and (EMBEDDINGS.exists() or GPU_MANIFEST.exists()):
        raise FileExistsError("refusing to overwrite repaired-eICU control outputs")
    OUT.mkdir(parents=True, exist_ok=True)
    if os.environ.get("V5_GPU_LOCK_HELD") != "1":
        raise RuntimeError("GPU producer must run under the shared R5/admin/gpu_v5.lock")
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ[name] = "2"
    # Torch is imported only after the launcher has acquired the shared GPU lock.
    import torch

    torch.set_num_threads(2)
    torch.set_num_interop_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("isolated GPU runtime reports no CUDA device")
    torch.cuda.set_device(0)
    props = torch.cuda.get_device_properties(0)
    memory_cap = 6 * 1024 ** 3
    memory_fraction = min(0.20, memory_cap / float(props.total_memory))
    torch.cuda.set_per_process_memory_fraction(memory_fraction, 0)
    torch.cuda.reset_peak_memory_stats(0)

    entry = get_manifest_entry()
    constant_entry = get_manifest_entry("constant")
    cache_pair_attestation = cache_validator.validate_real_constant_pair(
        MANIFEST, entry, constant_entry, DB,
    )
    rows, stays, subjects, V, B, H, L = load_inputs(
        entry, cache_pair_attestation["real"]["cache_sha256"])
    constant_cache_provenance = verify_constant_cache(
        entry, constant_entry, rows, stays, subjects, cache_pair_attestation)
    v_hash = hashlib.sha256(V.tobytes()).hexdigest()
    h_hash = hashlib.sha256(H.tobytes()).hexdigest()
    l_hash = hashlib.sha256(L.tobytes()).hexdigest()
    if sha256_file(CHECKPOINT) != EXPECTED_CHECKPOINT:
        raise ValueError("frozen main20k checkpoint hash differs")
    expected_fm_source = "9d33963a7d176f855103ac88c60f9faffebb4cb734a9de3401ba0c06bdaa39bc"
    if sha256_file(FM_SOURCE) != expected_fm_source:
        raise ValueError("original fm.py source changed from the attested v4 control runtime")

    code_path = str(ROOT / "code")
    if code_path not in os.sys.path:
        os.sys.path.insert(0, code_path)
    import fm
    torch.set_num_threads(2)
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = checkpoint["cfg"]
    nbin = int(checkpoint["state"]["bin.weight"].shape[0] - 2)
    model = fm.EHRTransformer(
        int(cfg["nvar"]), nbin=nbin, d=int(cfg["d"]),
        layers=int(cfg["layers"]), heads=int(cfg["heads"]), ff=int(cfg["ff"]),
        nhour=24, dropout=0.1,
    )
    model.load_state_dict(checkpoint["state"], strict=True)
    model.eval().to(torch.device("cuda:0"))
    frozen_state = {key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()}

    def encode(B_override: np.ndarray, label: str,
               row_positions: np.ndarray | None = None) -> np.ndarray:
        selected = (np.arange(len(rows), dtype=np.int64) if row_positions is None
                    else np.asarray(row_positions, dtype=np.int64))
        if selected.ndim != 1 or not len(selected) or selected.min() < 0 or selected.max() >= len(rows):
            raise ValueError(f"{label}: row positions are empty or outside the cache")
        result = np.empty((len(selected), int(model.d)), dtype=np.float32)
        with torch.no_grad():
            for start in range(0, len(selected), BATCH):
                end = min(start + BATCH, len(selected))
                batch_rows = selected[start:end]
                width = max(1, int(np.max(L[batch_rows])))
                v_np = np.asarray(V[batch_rows, :width], dtype=np.int64)
                b_np = np.asarray(B_override[batch_rows, :width], dtype=np.int64)
                h_np = np.asarray(H[batch_rows, :width], dtype=np.int64)
                if np.any(b_np[v_np == 0] != 0):
                    raise ValueError(f"{label}: padding bins changed")
                v_tensor = torch.from_numpy(v_np).to("cuda:0")
                b_tensor = torch.from_numpy(b_np).to("cuda:0")
                h_tensor = torch.from_numpy(h_np).to("cuda:0")
                embedding = model.embed(v_tensor, b_tensor, h_tensor, v_tensor == 0)
                result[start:end] = embedding.detach().to("cpu", dtype=torch.float32).numpy()
                if end % 10000 < BATCH or end == len(selected):
                    log(f"ENCODE {label} progress={end}/{len(selected)}")
        if not np.isfinite(result).all():
            raise FloatingPointError(f"{label}: nonfinite embeddings")
        return result

    observed_positions = np.linspace(
        0, len(rows) - 1, num=min(OBSERVED_SAMPLE_SIZE, len(rows)), dtype=np.int64)
    cached_observed_embedding = load_cache_embedding_sample(entry, observed_positions)
    observed_embedding = encode(B, "observed_runtime_reproduction", observed_positions)
    if cached_observed_embedding.shape != observed_embedding.shape:
        raise ValueError("observed runtime embedding sample shape differs from the frozen real cache")
    observed_embedding_max_abs = float(np.max(np.abs(observed_embedding - cached_observed_embedding)))
    if not np.allclose(observed_embedding, cached_observed_embedding,
                       atol=OBSERVED_EMBEDDING_ATOL, rtol=OBSERVED_EMBEDDING_RTOL):
        raise ValueError(
            "observed runtime embedding sample differs from the frozen real cache: "
            f"max_abs_difference={observed_embedding_max_abs:.10g}, "
            f"atol={OBSERVED_EMBEDDING_ATOL}, rtol={OBSERVED_EMBEDDING_RTOL}"
        )
    historical_scaler, historical_probe = load_historical_source_probe()
    cached_observed_probability = historical_probe.predict_proba(
        historical_scaler.transform(cached_observed_embedding))[:, 1]
    observed_probability = historical_probe.predict_proba(
        historical_scaler.transform(observed_embedding))[:, 1]
    observed_probability_max_abs = float(np.max(np.abs(observed_probability - cached_observed_probability)))
    if not np.allclose(observed_probability, cached_observed_probability,
                       atol=OBSERVED_PROBABILITY_ATOL, rtol=OBSERVED_PROBABILITY_RTOL):
        raise ValueError(
            "observed runtime probabilities differ from the frozen real-cache probabilities: "
            f"max_abs_difference={observed_probability_max_abs:.10g}, "
            f"atol={OBSERVED_PROBABILITY_ATOL}, rtol={OBSERVED_PROBABILITY_RTOL}"
        )
    torch.cuda.synchronize()
    parity_peak_gpu = int(torch.cuda.max_memory_allocated(0))
    if parity_peak_gpu > memory_cap:
        raise MemoryError(f"observed parity allocated {parity_peak_gpu} GPU bytes, exceeding 6 GiB")
    parity_summary = {
        "status": "passed",
        "run_id": RUN_ID or "standalone_20261123",
        "execution_resource_limits": (RESOURCE_LIMITS if RUN_ID else
                                       {"status": "standalone_limits_not_attested"}),
        "resource_scope": RESOURCE_SCOPE,
        "cgroup_path": current_cgroup_path(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort": DB,
        "embedding_manifest_sha256": sha256_file(MANIFEST),
        "real_cache_path": str(entry["cache"]),
        "real_cache_sha256": sha256_file(Path(entry["cache"])),
        "constant_input_cache_provenance": constant_cache_provenance,
        "cache_pair_attestation": cache_pair_attestation,
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "encoder_code_sha256": sha256_file(FM_SOURCE),
        "historical_source_probe_path": str(PROBE),
        "historical_source_probe_sha256": sha256_file(PROBE),
        "sample_rows": int(len(observed_positions)),
        "row_position_sha256": hashlib.sha256(observed_positions.tobytes()).hexdigest(),
        "embedding_max_abs_difference": observed_embedding_max_abs,
        "embedding_tolerance": {
            "atol": OBSERVED_EMBEDDING_ATOL, "rtol": OBSERVED_EMBEDDING_RTOL,
        },
        "probability_max_abs_difference": observed_probability_max_abs,
        "probability_tolerance": {
            "atol": OBSERVED_PROBABILITY_ATOL, "rtol": OBSERVED_PROBABILITY_RTOL,
        },
        "gpu_runtime": {
            "device": props.name, "memory_cap_bytes": memory_cap,
            "max_memory_allocated_bytes": parity_peak_gpu,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "torch_num_threads": int(torch.get_num_threads()),
        },
        "full_control_inference_started": False,
    }
    if PARITY_REPORT.exists():
        existing = json.loads(PARITY_REPORT.read_text(encoding="utf-8"))
        for field in ("status", "embedding_manifest_sha256", "real_cache_sha256",
                      "historical_source_probe_sha256", "row_position_sha256"):
            if existing.get(field) != parity_summary.get(field):
                raise ValueError(f"existing observed parity report differs in {field}")
        if existing.get("status") != "passed":
            raise ValueError("existing observed parity report does not pass the frozen tolerances")
    else:
        write_json_atomic(PARITY_REPORT, parity_summary)
    log(
        "OBSERVED_RUNTIME_PARITY_PASSED "
        f"rows={len(observed_positions)} embedding_max_abs={observed_embedding_max_abs:.10g} "
        f"probability_max_abs={observed_probability_max_abs:.10g} gpu_peak={parity_peak_gpu}"
    )
    if parity_only:
        log(f"OBSERVED_PARITY_ONLY_COMPLETE report={PARITY_REPORT}")
        return
    del cached_observed_embedding, observed_embedding
    within_bins, within_diag, within_changed_by_row = within_shuffle(
        V, B, L, np.random.default_rng(SEED_WITHIN))
    donor_bins, donor_diag, donor_changed_by_row = donor_replace(
        V, B, L, subjects, np.random.default_rng(SEED_DONOR))
    active_mask = (np.arange(V.shape[1])[None, :] < L[:, None]) & (V != 0)
    active_by_row = np.count_nonzero(active_mask, axis=1).astype(np.int64)
    constant_changed_by_row = np.count_nonzero(active_mask & (B != 1), axis=1).astype(np.int64)
    within_embedding = encode(within_bins, "within_variable_shuffle")
    del within_bins
    donor_embedding = encode(donor_bins, "distinct_patient_donor")
    del donor_bins
    torch.cuda.synchronize()
    if not all(torch.equal(frozen_state[key], value.detach().cpu())
               for key, value in model.state_dict().items()):
        raise AssertionError("pretrained encoder parameters changed during frozen inference")
    if (hashlib.sha256(V.tobytes()).hexdigest() != v_hash
            or hashlib.sha256(H.tobytes()).hexdigest() != h_hash
            or hashlib.sha256(L.tobytes()).hexdigest() != l_hash):
        raise AssertionError("V/H/L/L positions changed during inference")
    max_gpu = int(torch.cuda.max_memory_allocated(0))
    if max_gpu > memory_cap:
        raise MemoryError(f"peak allocated GPU memory {max_gpu} exceeds 6 GiB")
    temporary = EMBEDDINGS.with_suffix(".npz.tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, rows=rows, stay_id=stays, subject=subjects,
                 within_embedding=within_embedding, donor_embedding=donor_embedding,
                 active_token_counts=active_by_row,
                 constant_changed_token_counts=constant_changed_by_row,
                 within_changed_token_counts=within_changed_by_row,
                 donor_changed_token_counts=donor_changed_by_row)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, EMBEDDINGS)
    report = {
        "status": "complete",
        "run_id": RUN_ID or "standalone_20261123",
        "execution_resource_limits": (RESOURCE_LIMITS if RUN_ID else
                                       {"status": "standalone_limits_not_attested"}),
        "resource_scope": RESOURCE_SCOPE,
        "cgroup_path": current_cgroup_path(),
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "cohort": DB, "sequence_path": str(SEQ),
        "sequence_sha256": sha256_file(SEQ),
        "eligibility_identity_source": "verified repaired-eICU main20k real cache identity",
        "embedding_manifest": str(MANIFEST),
        "embedding_manifest_sha256": sha256_file(MANIFEST),
        "input_real_cache": str(entry["cache"]),
        "input_real_cache_sha256": sha256_file(Path(entry["cache"])),
        "constant_input_cache_provenance": constant_cache_provenance,
        "cache_pair_attestation": cache_pair_attestation,
        "observed_runtime_reproduction": {
            "status": "passed",
            "historical_source_probe": str(PROBE),
            "historical_source_probe_sha256": sha256_file(PROBE),
            "sample_rows": int(len(observed_positions)),
            "row_position_sha256": hashlib.sha256(observed_positions.tobytes()).hexdigest(),
            "embedding_max_abs_difference": observed_embedding_max_abs,
            "embedding_tolerance": {
                "atol": OBSERVED_EMBEDDING_ATOL, "rtol": OBSERVED_EMBEDDING_RTOL,
            },
            "probability_max_abs_difference": observed_probability_max_abs,
            "probability_tolerance": {
                "atol": OBSERVED_PROBABILITY_ATOL, "rtol": OBSERVED_PROBABILITY_RTOL,
            },
        },
        "observed_runtime_parity_report": str(PARITY_REPORT),
        "observed_runtime_parity_report_sha256": sha256_file(PARITY_REPORT),
        "row_identity_sha256": entry["row_identity_sha256"], "n": int(len(rows)),
        "checkpoint_sha256": sha256_file(CHECKPOINT),
        "encoder_code_sha256": sha256_file(FM_SOURCE),
        "transformations": {
            "within_variable_shuffle": within_diag,
            "distinct_patient_donor": donor_diag,
        },
        "outcome_labels_read_by_gpu_producer": False,
        "frozen_pretrained_parameters_verified_unchanged": True,
        "readout_fit_or_selection_performed": False,
        "gpu_runtime": {
            "device": props.name, "total_memory_bytes": int(props.total_memory),
            "memory_fraction": memory_fraction, "allocated_memory_cap_bytes": memory_cap,
            "max_memory_allocated_bytes": max_gpu, "batch_size": BATCH,
            "torch_version": str(torch.__version__),
            "torch_cuda_version": str(torch.version.cuda),
            "torch_num_threads": int(torch.get_num_threads()),
        },
        "intermediate_path": str(EMBEDDINGS),
        "intermediate_sha256": sha256_file(EMBEDDINGS),
        "intermediate_size_bytes": EMBEDDINGS.stat().st_size,
    }
    GPU_MANIFEST.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    log(f"GPU_CONTROL_COMPLETE n={len(rows)} max_cuda_memory={max_gpu} manifest={GPU_MANIFEST}")


if __name__ == "__main__":
    main()
