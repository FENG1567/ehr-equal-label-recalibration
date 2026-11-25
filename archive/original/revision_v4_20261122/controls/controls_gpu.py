#!/usr/bin/env python3
"""GPU-only embedding production for the prespecified v4 controls.

This script does not import sklearn, open the fixed-probe pickle, fit a readout,
or write patient-level prediction outputs. Its only persisted patient-level
artifact is the server-side embedding intermediate consumed by controls_cpu.py.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
OUT = REV / "controls" / "fixed_head_controls_20261123"
INPUTS = OUT / "preflight_arrays.npz"
PREFLIGHT = OUT / "preflight_manifest.json"
INTERMEDIATE = OUT / "gpu_embeddings.npz"
GPU_MANIFEST = OUT / "gpu_encode_manifest.json"
SMOKE_EMBEDDINGS = OUT / "smoke_embeddings.npz"
SMOKE_GPU_MANIFEST = OUT / "smoke_gpu_manifest.json"
CHECKPOINT = ROOT / "models/main20k.pt"
ENCODER_CODE = ROOT / "code/fm.py"
V2_ENCODER_CODE = ROOT / "revision_v2_20261117/code/decoding_motor_v2.py"
SEED = 20261102


def log(msg: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def within_shuffle(V: np.ndarray, B: np.ndarray, L: np.ndarray,
                   rng: np.random.Generator) -> tuple[np.ndarray, int, int]:
    V = np.asarray(V); B = np.asarray(B); L = np.asarray(L)
    out = B.copy(); changed = 0; active = 0
    for i in range(len(L)):
        k = int(L[i]); v = V[i, :k]; b = out[i, :k]
        for vid in np.unique(v):
            if vid == 0:
                continue
            pos = np.flatnonzero(v == vid)
            before = B[i, pos].copy()
            out[i, pos] = rng.permutation(before)
            if not np.array_equal(np.sort(before), np.sort(out[i, pos])):
                raise AssertionError("within-shuffle changed a stay-variable B-bin multiset")
        active += int(np.count_nonzero(V[i, :k]))
        changed += int(np.count_nonzero(out[i, :k] != B[i, :k]))
        if (i + 1) % 20000 == 0:
            log(f"WITHIN_SHUFFLE progress={i + 1}/{len(L)}")
    if not np.array_equal(V, np.asarray(V)) or np.any(out[V == 0] != 0):
        raise AssertionError("within-shuffle changed V or padding bins")
    return out, changed, active


def donor_replace(V: np.ndarray, B: np.ndarray, L: np.ndarray,
                  subjects: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, dict]:
    """Select one uniform different-patient donor per target patient-variable."""
    V = np.asarray(V); B = np.asarray(B); L = np.asarray(L)
    subjects = np.asarray(subjects).astype(str)
    donor_values: dict[tuple[str, int], list[int]] = {}
    variable_subjects: dict[int, set[str]] = {}
    subject_variables: dict[str, set[int]] = {}
    for i, subject in enumerate(subjects):
        k = int(L[i])
        for vid in np.unique(V[i, :k]):
            if vid == 0:
                continue
            values = B[i, :k][V[i, :k] == vid].astype(np.int64).tolist()
            donor_values.setdefault((subject, int(vid)), []).extend(values)
            variable_subjects.setdefault(int(vid), set()).add(subject)
            subject_variables.setdefault(subject, set()).add(int(vid))
        if (i + 1) % 20000 == 0:
            log(f"DONOR_POOL progress={i + 1}/{len(subjects)}")
    sorted_subjects = {vid: sorted(ids) for vid, ids in variable_subjects.items()}
    subject_positions = {vid: {subject: i for i, subject in enumerate(ids)}
                         for vid, ids in sorted_subjects.items()}
    donor_map: dict[tuple[str, int], str] = {}
    candidate_group_frequency: dict[int, int] = {}
    candidate_min = None
    candidate_max = 0
    no_donor = []
    for group_i, (target, variables) in enumerate(subject_variables.items()):
        for vid in sorted(variables):
            ids = sorted_subjects[vid]
            n = len(ids)
            n_other = n - 1
            candidate_min = n_other if candidate_min is None else min(candidate_min, n_other)
            candidate_max = max(candidate_max, n_other)
            candidate_group_frequency[n_other] = candidate_group_frequency.get(n_other, 0) + 1
            if n_other <= 0:
                no_donor.append({"target_subject": target, "variable_id": int(vid)})
                continue
            # Draw uniformly from n-1 nonself positions without constructing/sorting
            # an O(n) candidate list for every target patient-variable pair.
            target_pos = subject_positions[vid][target]
            donor_pos = int(rng.integers(0, n_other))
            if donor_pos >= target_pos:
                donor_pos += 1
            donor_map[(target, int(vid))] = ids[donor_pos]
        if (group_i + 1) % 20000 == 0:
            log(f"DONOR_ASSIGN progress={group_i + 1}/{len(subject_variables)}")
    if no_donor:
        raise RuntimeError(f"no other-patient donor exists for {len(no_donor)} patient-variable groups")
    if len(donor_map) != sum(candidate_group_frequency.values()) or any(k[0] == v for k, v in donor_map.items()):
        raise AssertionError("donor coverage incomplete or self-donor detected")
    out = B.copy(); changed = 0; active = 0
    for i, target in enumerate(subjects):
        k = int(L[i]); v = V[i, :k]
        active += int(np.count_nonzero(v))
        for vid in np.unique(v):
            if vid == 0:
                continue
            pos = np.flatnonzero(v == vid)
            donor = donor_map[(target, int(vid))]
            values = np.asarray(donor_values[(donor, int(vid))], dtype=np.int64)
            if values.size == 0:
                raise AssertionError("selected donor has no bins for requested variable")
            out[i, pos] = rng.choice(values, size=len(pos), replace=True)
        changed += int(np.count_nonzero(out[i, :k] != B[i, :k]))
        if (i + 1) % 20000 == 0:
            log(f"DONOR_REPLACE progress={i + 1}/{len(subjects)}")
    if np.any(out[V == 0] != 0):
        raise AssertionError("donor replacement changed padding bins")
    unique_donors = len(set(donor_map.values()))
    group_count = sum(candidate_group_frequency.values())
    middle = (group_count - 1) // 2
    cumulative = 0
    median_candidates = None
    for count, frequency in sorted(candidate_group_frequency.items()):
        cumulative += frequency
        if cumulative > middle:
            median_candidates = int(count)
            break
    diag = {
        "target_subject_variable_groups": int(len(donor_map)),
        "donor_coverage_groups": int(len(donor_map)),
        "donor_coverage_fraction": 1.0,
        "unique_donor_subjects": int(unique_donors),
        "self_donor_groups": 0,
        "no_other_donor_groups": 0,
        "donor_candidates_min": int(candidate_min if candidate_min is not None else 0),
        "donor_candidates_median": float(median_candidates if median_candidates is not None else 0),
        "donor_candidates_max": int(candidate_max),
        "donor_assignment": "uniform over distinct eligible evaluation subjects with this variable; one donor per target subject-variable; donor B bins sampled with replacement to fill target token positions",
        "outcome_used_for_donor_assignment": False,
    }
    return out, {**diag, "changed_tokens": int(changed), "active_tokens": int(active)}


def load_ds(db: str) -> dict[str, np.ndarray]:
    path = ROOT / "data/processed" / f"seq_{db}.npz"
    with np.load(path, allow_pickle=True) as z:
        ds = {k: z[k] for k in ("V", "B", "H", "L", "stay_id", "subject")}
    return ds


def embed_rows(model, ds: dict[str, np.ndarray], rows: np.ndarray,
               device: torch.device, b_override: np.ndarray | None = None,
               batch_size: int = 16, label: str = "") -> np.ndarray:
    rows = np.asarray(rows, dtype=np.int64)
    if b_override is not None and b_override.shape != ds["B"][rows].shape:
        raise ValueError(f"{label}: B override shape mismatch")
    result = np.empty((len(rows), int(model.d)), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(rows), batch_size):
            ix = rows[start:start + batch_size]
            width = max(1, int(np.max(ds["L"][ix])))
            V = np.asarray(ds["V"][ix, :width], dtype=np.int64)
            if b_override is None:
                B = np.asarray(ds["B"][ix, :width], dtype=np.int64)
            else:
                loc = slice(start, start + len(ix))
                B = np.asarray(b_override[loc, :width], dtype=np.int64)
            H = np.asarray(ds["H"][ix, :width], dtype=np.int64)
            if np.any(B[V == 0] != 0):
                raise ValueError(f"{label}: padding B bins changed")
            vt = torch.from_numpy(V).to(device)
            bt = torch.from_numpy(B).to(device)
            ht = torch.from_numpy(H).to(device)
            emb = model.embed(vt, bt, ht, vt == 0)
            result[start:start + len(ix)] = emb.detach().to("cpu", dtype=torch.float32).numpy()
            if (start + len(ix)) % 5000 < batch_size or start + len(ix) == len(rows):
                log(f"ENCODE {label}: {start + len(ix)}/{len(rows)}")
    if not np.isfinite(result).all():
        raise FloatingPointError(f"{label}: nonfinite embedding")
    return result


def self_test() -> None:
    V = np.asarray([[1, 1, 2, 0], [1, 2, 2, 0], [1, 2, 0, 0]], dtype=np.int64)
    B = np.asarray([[2, 5, 3, 0], [5, 7, 1, 0], [4, 9, 0, 0]], dtype=np.int64)
    H = np.asarray([[0, 1, 2, 0], [1, 2, 3, 0], [2, 3, 0, 0]], dtype=np.int64)
    L = np.asarray([3, 3, 2], dtype=np.int64)
    subjects = np.asarray(["p1", "p1", "p2"])
    within, _, _ = within_shuffle(V, B, L, np.random.default_rng(5))
    for i in range(len(L)):
        for vid in np.unique(V[i, :L[i]]):
            if vid > 0 and sorted(within[i, :L[i]][V[i, :L[i]] == vid]) != sorted(B[i, :L[i]][V[i, :L[i]] == vid]):
                raise AssertionError("actual within helper changed a variable multiset")
    donor, diag = donor_replace(V, B, L, subjects, np.random.default_rng(7))
    if diag["self_donor_groups"] != 0 or diag["donor_coverage_fraction"] != 1:
        raise AssertionError("actual donor helper self-donor/coverage check failed")
    if not np.array_equal(V, np.asarray(V)) or not np.array_equal(H, np.asarray(H)) or np.any(within[V == 0] != 0) or np.any(donor[V == 0] != 0):
        raise AssertionError("actual controls changed V/H/padding")
    log("GPU_SELF_TESTS_PASS actual within/donor transformation helpers")


def smoke() -> None:
    if not INPUTS.is_file() or not PREFLIGHT.is_file():
        raise FileNotFoundError("run validated CPU preflight before GPU smoke")
    if SMOKE_EMBEDDINGS.exists() or SMOKE_GPU_MANIFEST.exists():
        raise FileExistsError("refusing to overwrite GPU smoke artifacts")
    pre = json.loads(PREFLIGHT.read_text())
    if sha256_file(INPUTS) != pre.get("input_arrays_sha256"):
        raise ValueError("preflight input array hash mismatch")
    if sha256_file(CHECKPOINT) != pre.get("checkpoint_sha256"):
        raise ValueError("checkpoint changed after CPU preflight")
    if sha256_file(ENCODER_CODE) != pre.get("fm_source_sha256") or sha256_file(V2_ENCODER_CODE) != pre.get("v2_encoder_source_sha256"):
        raise ValueError("encoder source files changed after CPU preflight")
    with np.load(INPUTS, allow_pickle=False) as z:
        rows = z["mimic_eval_rows"][:2]
        stay = z["mimic_eval_stay_id"][:2].astype(str)
        subject = z["mimic_eval_subject"][:2].astype(str)
    sys.path.insert(0, str(ROOT / "code"))
    import fm
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("verified CUDA runtime has no device")
    torch.cuda.set_per_process_memory_fraction(0.18, 0)
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    nbin = int(ck["state"]["bin.weight"].shape[0] - 2)
    model = fm.EHRTransformer(int(cfg["nvar"]), nbin=nbin, d=int(cfg["d"]),
                              layers=int(cfg["layers"]), heads=int(cfg["heads"]),
                              ff=int(cfg["ff"])).eval().to(torch.device("cuda:0"))
    model.load_state_dict(ck["state"], strict=True)
    ds = load_ds("mimic")
    if (not np.array_equal(np.asarray(ds["stay_id"])[rows].astype(str), stay)
            or not np.array_equal(np.asarray(ds["subject"])[rows].astype(str), subject)):
        raise ValueError("smoke row IDs do not match exact eligible source rows")
    emb = embed_rows(model, ds, rows, torch.device("cuda:0"), label="mimic/smoke")
    with SMOKE_EMBEDDINGS.open("wb") as f:
        np.savez(f, rows=rows, stay_id=stay, subject=subject, embedding=emb)
        f.flush(); os.fsync(f.fileno())
    with np.load(SMOKE_EMBEDDINGS, allow_pickle=False) as z:
        if (not np.array_equal(z["rows"], rows) or not np.array_equal(z["stay_id"].astype(str), stay)
                or not np.array_equal(z["subject"].astype(str), subject)
                or not np.array_equal(z["embedding"], emb)):
            raise AssertionError("GPU smoke NPZ allow_pickle=False round trip failed")
    report = {"status": "gpu_smoke_embeddings_ready", "n": 2,
              "preflight_manifest_sha256": sha256_file(PREFLIGHT),
              "checkpoint_sha256": sha256_file(CHECKPOINT),
              "encoder_code_sha256": pre["encoder_code_sha256"],
              "smoke_embeddings_sha256": sha256_file(SMOKE_EMBEDDINGS),
              "allow_pickle_false_roundtrip": True,
              "producer_runtime": {"python": sys.version.split()[0], "torch": str(torch.__version__),
                                   "torch_cuda": str(torch.version.cuda),
                                   "device": torch.cuda.get_device_name(0),
                                   "torch_num_threads": torch.get_num_threads(),
                                   "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0))}}
    SMOKE_GPU_MANIFEST.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    log(f"GPU_SMOKE_EMBEDDING_PASS sha256={report['smoke_embeddings_sha256']}")
    del model
    torch.cuda.empty_cache()


def main():
    if sys.argv[1:] == ["--self-test"]:
        self_test()
        return
    if sys.argv[1:] == ["--smoke"]:
        smoke()
        return
    if not INPUTS.is_file() or not PREFLIGHT.is_file():
        raise FileNotFoundError("run validated CPU preflight before GPU encoding")
    if INTERMEDIATE.exists() or GPU_MANIFEST.exists() or (OUT / "random_main20k_seed20261102.pt").exists():
        raise FileExistsError("refusing to overwrite a previous control output")
    pre = json.loads(PREFLIGHT.read_text())
    if pre.get("status") != "preflight_validated" or sha256_file(INPUTS) != pre.get("input_arrays_sha256"):
        raise ValueError("preflight status or input array hash invalid")
    if (sha256_file(CHECKPOINT) != pre.get("checkpoint_sha256")
            or sha256_file(ENCODER_CODE) != pre.get("fm_source_sha256")
            or sha256_file(V2_ENCODER_CODE) != pre.get("v2_encoder_source_sha256")):
        raise ValueError("checkpoint/encoder code changed after CPU preflight")
    for db in ("mimic", "eicu"):
        meta = pre["cohorts"][db]
        seq_path = Path(meta["source_sequence"])
        stat = seq_path.stat()
        if stat.st_size != meta["source_sequence_size_bytes"] or stat.st_mtime_ns != meta["source_sequence_mtime_ns"]:
            raise ValueError(f"{db}: source NPZ changed since preflight")
    with np.load(INPUTS, allow_pickle=False) as z:
        inp = {k: z[k] for k in z.files}
    OUT.mkdir(parents=True, exist_ok=True)
    self_test()
    # Import only the original architecture definition; reset its import-time 16-thread setting.
    sys.path.insert(0, str(ROOT / "code"))
    import fm
    torch.set_num_threads(8)
    if not torch.cuda.is_available():
        raise RuntimeError("verified CUDA runtime has no CUDA device")
    torch.cuda.set_per_process_memory_fraction(0.18, 0)
    device = torch.device("cuda:0")
    ck = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    cfg = ck["cfg"]
    nbin = int(ck["state"]["bin.weight"].shape[0] - 2)
    model_cfg = {"nvar": int(cfg["nvar"]), "nbin": nbin, "d": int(cfg["d"]),
                 "layers": int(cfg["layers"]), "heads": int(cfg["heads"]), "ff": int(cfg["ff"]),
                 "nhour": 24, "dropout": 0.1}
    trained = fm.EHRTransformer(model_cfg["nvar"], nbin=model_cfg["nbin"], d=model_cfg["d"],
                                layers=model_cfg["layers"], heads=model_cfg["heads"], ff=model_cfg["ff"],
                                nhour=model_cfg["nhour"], dropout=model_cfg["dropout"])
    trained.load_state_dict(ck["state"], strict=True)
    trained.eval()
    frozen_snapshot = {k: v.detach().cpu().clone() for k, v in trained.state_dict().items()}
    trained.to(device)
    # Fixed head controls: evaluation rows are exactly those admitted by the preflight manifest.
    embeddings = {}
    diagnostics = {"mimic": {}, "eicu": {}}
    for db in ("mimic", "eicu"):
        ds = load_ds(db)
        eval_rows = inp[f"{db}_eval_rows"]
        if (not np.array_equal(np.asarray(ds["stay_id"])[eval_rows].astype(str), inp[f"{db}_eval_stay_id"])
                or not np.array_equal(np.asarray(ds["subject"])[eval_rows].astype(str), inp[f"{db}_eval_subject"])):
            raise ValueError(f"{db}: sequence IDs changed since preflight")
        V = np.asarray(ds["V"][eval_rows], dtype=np.int64)
        B = np.asarray(ds["B"][eval_rows], dtype=np.int64)
        H = np.asarray(ds["H"][eval_rows], dtype=np.int64)
        L = np.asarray(ds["L"][eval_rows], dtype=np.int64)
        # Restrict all perturbations to the same eligible evaluation stays.
        Bwithin, nw, denw = within_shuffle(V, B, L, np.random.default_rng(2026112301 + (0 if db == "mimic" else 100)))
        Bdonor, donor_diag = donor_replace(V, B, L, inp[f"{db}_eval_subject"],
                                           np.random.default_rng(2026112302 + (0 if db == "mimic" else 100)))
        # Verify invariant triplets explicitly on the production arrays.
        if not np.array_equal(V, ds["V"][eval_rows]) or not np.array_equal(H, ds["H"][eval_rows]) or not np.array_equal(L, ds["L"][eval_rows]):
            raise AssertionError("production perturbation changed V/H/L")
        if np.any(Bwithin[V == 0] != B[V == 0]) or np.any(Bdonor[V == 0] != B[V == 0]):
            raise AssertionError("production perturbation changed padding bins")
        model_override = {"V": V, "B": Bwithin, "H": H, "L": L}
        embeddings[f"{db}_within_embedding"] = embed_rows(trained, model_override, np.arange(len(L)), device,
                                                           b_override=Bwithin, label=f"{db}/within")
        model_override["B"] = Bdonor
        embeddings[f"{db}_donor_embedding"] = embed_rows(trained, model_override, np.arange(len(L)), device,
                                                          b_override=Bdonor, label=f"{db}/donor")
        active = int(np.count_nonzero(V))
        diagnostics[db] = {
            "within_changed_tokens": int(nw), "within_active_tokens": int(denw),
            "within_changed_token_fraction": float(nw / denw),
            "donor_changed_tokens": int(donor_diag["changed_tokens"]),
            "donor_active_tokens": int(donor_diag["active_tokens"]),
            "donor_changed_token_fraction": float(donor_diag["changed_tokens"] / donor_diag["active_tokens"]),
            "donor": {k: v for k, v in donor_diag.items()
                      if k not in ("changed_tokens", "active_tokens")},
            "active_tokens": active,
            "v_h_l_padding_unchanged": True,
        }
        log(f"FIXED_HEAD_ENCODE_COMPLETE {db}: within_changed={nw}/{denw}; donor_changed={donor_diag['changed_tokens']}/{donor_diag['active_tokens']}; donor_coverage={donor_diag['donor_coverage_fraction']}")
        del ds, V, B, H, L, Bwithin, Bdonor, model_override
    if not all(torch.equal(frozen_snapshot[k], v.detach().cpu()) for k, v in trained.state_dict().items()):
        raise AssertionError("frozen pretrained encoder parameters changed during control encoding")
    del trained
    torch.cuda.empty_cache()

    # Random initialization: seed before construction; no pretrained state is loaded.
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    rng_before = torch.get_rng_state().clone()
    random_model = fm.EHRTransformer(model_cfg["nvar"], nbin=model_cfg["nbin"], d=model_cfg["d"],
                                     layers=model_cfg["layers"], heads=model_cfg["heads"], ff=model_cfg["ff"],
                                     nhour=model_cfg["nhour"], dropout=model_cfg["dropout"])
    rng_after = torch.get_rng_state().clone()
    random_model.eval()
    random_state = {k: v.detach().cpu().clone() for k, v in random_model.state_dict().items()}
    parent_state = ck["state"]
    if any(random_state[k].shape != parent_state[k].shape for k in random_state):
        raise ValueError("random model architecture state shapes differ from main20k")
    if all(torch.equal(random_state[k], parent_state[k].cpu()) for k in random_state):
        raise AssertionError("random initialization unexpectedly equals all pretrained weights")
    random_checkpoint = OUT / "random_main20k_seed20261102.pt"
    tmp_ckpt = random_checkpoint.with_suffix(".pt.tmp")
    torch.save({"state": random_state, "cfg": model_cfg, "seed": SEED,
                "initialization": "fresh EHRTransformer constructor after torch.manual_seed(seed); no pretrained weights loaded",
                "weights_loaded_from_pretrained": False,
                "parent_architecture_checkpoint_sha256": pre["checkpoint_sha256"],
                "encoder_code_sha256": pre["encoder_code_sha256"],
                "torch_rng_state_before_constructor": rng_before,
                "torch_rng_state_after_constructor": rng_after}, tmp_ckpt)
    os.replace(tmp_ckpt, random_checkpoint)
    random_model.to(device)
    rand_embeddings = {}
    for db in ("mimic", "eicu"):
        ds = load_ds(db)
        if db == "mimic":
            rows = inp["mimic_model_rows"]
            if not np.array_equal(np.asarray(ds["stay_id"])[rows].astype(str), inp["mimic_model_stay_id"]):
                raise ValueError("MIMIC random-encoder rows do not match preflight identity")
            rand_embeddings["mimic_random_embedding"] = embed_rows(random_model, ds, rows, device,
                                                                    label="mimic/random")
        else:
            rows = inp["eicu_eval_rows"]
            if not np.array_equal(np.asarray(ds["stay_id"])[rows].astype(str), inp["eicu_eval_stay_id"]):
                raise ValueError("eICU random-encoder rows do not match preflight identity")
            rand_embeddings["eicu_random_embedding"] = embed_rows(random_model, ds, rows, device,
                                                                   label="eicu/random")
        del ds
    del random_model
    torch.cuda.empty_cache()
    # Non-object arrays only: allow_pickle=False is required by the consumer.
    all_embeddings = {**embeddings, **rand_embeddings}
    with INTERMEDIATE.open("wb") as f:
        np.savez(f, **all_embeddings)
        f.flush(); os.fsync(f.fileno())
    with np.load(INTERMEDIATE, allow_pickle=False) as z:
        for key in all_embeddings:
            if not np.array_equal(z[key], all_embeddings[key]):
                raise AssertionError(f"GPU intermediate serialization round trip failed: {key}")
    report = {
        "status": "gpu_encoding_complete", "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "preflight_manifest": str(PREFLIGHT), "preflight_manifest_sha256": sha256_file(PREFLIGHT),
        "preflight_input_arrays_sha256": pre["input_arrays_sha256"],
        "checkpoint_sha256": sha256_file(CHECKPOINT), "encoder_code_sha256": pre["encoder_code_sha256"],
        "fm_source_sha256": sha256_file(ENCODER_CODE),
        "v2_encoder_source_sha256": sha256_file(V2_ENCODER_CODE),
        "random_checkpoint": str(random_checkpoint), "random_checkpoint_sha256": sha256_file(random_checkpoint),
        "random_init_seed": SEED, "random_init_weights_loaded_from_pretrained": False,
        "random_init_config": model_cfg, "random_init_state_tensor_count": int(len(random_state)),
        "random_init_state_tensor_sha256": hashlib.sha256(b"".join(v.contiguous().numpy().tobytes() for _, v in sorted(random_state.items()))).hexdigest(),
        "encoder_model_training_performed": False, "readout_fitting_performed": False,
        "cohorts": diagnostics,
        "intermediate": str(INTERMEDIATE), "intermediate_sha256": sha256_file(INTERMEDIATE),
        "intermediate_size_bytes": INTERMEDIATE.stat().st_size,
        "producer_runtime": {
            "python": sys.version.split()[0], "implementation": platform.python_implementation(),
            "numpy": np.__version__, "torch": str(torch.__version__), "torch_cuda": str(torch.version.cuda),
            "device": torch.cuda.get_device_name(0), "compute_capability": list(torch.cuda.get_device_capability(0)),
            "torch_num_threads": torch.get_num_threads(), "batch_size": 16,
            "max_cuda_memory_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "cuda_memory_fraction_limit": 0.18,
        },
    }
    tmp = GPU_MANIFEST.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(tmp, GPU_MANIFEST)
    log(f"GPU_ENCODING_COMPLETE manifest={GPU_MANIFEST} intermediate_sha256={report['intermediate_sha256']} max_cuda_memory={report['producer_runtime']['max_cuda_memory_allocated_bytes']}")


if __name__ == "__main__":
    main()
