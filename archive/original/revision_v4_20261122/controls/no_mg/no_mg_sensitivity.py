#!/usr/bin/env python3
"""Symmetric main20k magnesium-omission sensitivity; server-only row artifacts."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
REV = ROOT / "revision_v4_20261122"
OUT = REV / "controls/no_mg"
MIMIC_SEQ = ROOT / "data/processed/seq_mimic.npz"
MIMIC_VOCAB = ROOT / "data/processed/meta_mimic.json"
MIMIC_ELIG = REV / "temporal/results/eligibility/status_consistent_v4_1/landmark_v4_1"
C_ELIG = REV / "temporal/results/eligibility/c2012_external_reviewed_v4_2"
REGISTRY = REV / "data_registry.json"
C_REVIEW_REGISTRY = C_ELIG / "registry_c2012_reviewed_v4_2.json"
REVIEW_MEMO = REV / "admin/c2012_scope_record_20261123.md"
CHECKPOINT = ROOT / "models/main20k.pt"
ENCODER = ROOT / "code/fm.py"
ENGINE = REV / "engine/code/v4_engine.py"
REAL_EMBED_MANIFEST = REV / "decoding/inputs/immutable/embedding_manifest_core14_20261123_a636f5f3.json"
INPUTS = OUT / "preflight_arrays.npz"
PREFLIGHT = OUT / "preflight_manifest.json"
EMBEDDINGS = OUT / "gpu_embeddings.npz"
GPU_MANIFEST = OUT / "gpu_encode_manifest.json"
PREDICTIONS = OUT / "patient_predictions.npz"
RESULTS = OUT / "no_mg_aggregate.json"
RESULTS_V4_1 = OUT / "no_mg_aggregate_v4_1.json"
RECOVERY_AUDIT = OUT / "no_mg_rank_recovery_audit_v4_1.json"
EVAL_KEYS = OUT / "c2012_full_eval_keys_server_only.json"
EXPECTED_CHECKPOINT = "0719db57b91b261869aed119cee95bf8980eb14411684c2040ba8588b45460bd"
EXPECTED_MIMIC_SEQUENCE = "9bd20bf1394a34ba112a4de9a8d7e8971821bef90558a9987be4b9da5ae06824"
EXPECTED_MIMIC_ELIGIBILITY = "ea2b55bc3cf071d48b019d4ba97b4a3dc53d5d7fb3f6ac49a9e7c26007064cb5"
EXPECTED_REAL_EMBED_MANIFEST = "a636f5f38ba7376dcb632cc62ba20592413376da366f105711ce412b32d1fdc9"
EXPECTED_REVIEW = "59d683ec8df200edc776a7628efea86d3da94d26d6ace1ae64704ae8f8661943"
C_NAMES = ("c2012_a_v4_1", "c2012_b_v4_1", "c2012_c_v4_1")
BOOTSTRAPS = 1000
SEED = 20261123
CONSTANT_BIN = 1
SCRIPT_PATH = Path(__file__).resolve()


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(payload: object) -> str:
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(raw).hexdigest()


def atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def meta_paths() -> tuple[dict, dict]:
    registry = json.loads(REGISTRY.read_text())
    reviewed = json.loads(C_REVIEW_REGISTRY.read_text())
    entries = registry["cohorts"]
    reviewed_entries = reviewed["cohorts"]
    for name in C_NAMES:
        if entries.get(name) != reviewed_entries.get(name):
            raise ValueError(f"{name}: registered sequence/split/vocab paths differ from reviewed registry")
    return registry, reviewed


def load_sequence(path: Path, split_path: Path) -> dict[str, np.ndarray]:
    # Source NPZs are registered project artifacts; some legacy IDs use object strings.
    # Derived outputs below always use Unicode and allow_pickle=False.
    with np.load(path, allow_pickle=True) as z:
        result = {k: z[k] for k in ("V", "B", "H", "L", "stay_id", "subject", "y_mort")}
    result["split"] = np.load(split_path, allow_pickle=False)
    n = len(result["L"])
    if any(len(result[k]) != n for k in ("V", "B", "H", "stay_id", "subject", "y_mort", "split")):
        raise ValueError(f"{path}: row dimensions disagree")
    if result["V"].shape != result["B"].shape or result["V"].shape != result["H"].shape:
        raise ValueError(f"{path}: V/B/H dimensions disagree")
    if np.any(result["L"] < 0) or np.any(result["L"] > result["V"].shape[1]):
        raise ValueError(f"{path}: L outside sequence width")
    return result


def sha_ids(cohort: str, ids: np.ndarray, labels: np.ndarray) -> str:
    records = sorted((str(i), int(y)) for i, y in zip(ids, labels))
    raw = "\n".join(f"{cohort}|{i}|{y}" for i, y in records).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def prepare_no_mg(V: np.ndarray, B: np.ndarray, H: np.ndarray, L: np.ndarray,
                  mg_id: int, const_bin: int = CONSTANT_BIN) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Zero Mg-token triplets; B1 sets every remaining active B to constant bin 1."""
    v = np.asarray(V).copy(); b = np.asarray(B).copy(); h = np.asarray(H).copy()
    l = np.asarray(L, dtype=np.int64)
    active = np.arange(v.shape[1])[None, :] < l[:, None]
    if np.any(v[~active] != 0):
        raise ValueError("nonpadding V tokens occur beyond original L")
    original_mg = active & (v == int(mg_id))
    v[original_mg] = 0; b[original_mg] = 0; h[original_mg] = 0
    remaining = active & (v != 0)
    excluded_empty = ~remaining.any(axis=1)
    b_const = np.zeros_like(b)
    b_const[remaining] = int(const_bin)
    if np.any(b[~remaining] != 0) or np.any(h[~remaining] != 0):
        raise AssertionError("masked/padding bins or hours must be zero")
    return v, b, h, b_const, {
        "n_rows": int(len(v)), "n_mg_tokens_omitted": int(original_mg.sum()),
        "n_active_tokens_before": int(np.count_nonzero(active & (np.asarray(V) != 0))),
        "n_active_tokens_after": int(remaining.sum()),
        "n_rows_without_non_mg_tokens": int(excluded_empty.sum()),
        "constant_bin": int(const_bin),
    }


def c2012_gate_rows(name: str, ds: dict[str, np.ndarray], gate_path: Path) -> tuple[np.ndarray, dict, list[dict]]:
    import pandas as pd
    gate = pd.read_parquet(gate_path)
    required = {"stay_id", "eligible_24h", "y_mort"}
    if not required.issubset(gate.columns):
        raise ValueError(f"{name}: reviewed gate missing columns {sorted(required-set(gate.columns))}")
    stay = np.asarray(ds["stay_id"]).astype(str)
    if len(set(stay)) != len(stay):
        raise ValueError(f"{name}: duplicate sequence stay IDs")
    pos = {sid: i for i, sid in enumerate(stay)}
    gstay = gate["stay_id"].astype(str)
    if gstay.duplicated().any():
        raise ValueError(f"{name}: duplicate reviewed gate stay IDs")
    row = gstay.map(pos)
    missing = row.isna().to_numpy()
    if missing.any():
        raise ValueError(f"{name}: {int(missing.sum())} reviewed gate IDs absent from sequence")
    rows = row.to_numpy(dtype=np.int64)
    source_y = pd.to_numeric(gate.get("source_y_mort", gate["y_mort"]), errors="coerce").to_numpy(dtype=float)
    gate_y = pd.to_numeric(gate["y_mort"], errors="coerce").to_numpy(dtype=float)
    seq_y = np.asarray(ds["y_mort"], dtype=float)[rows]
    finite_known = np.isfinite(source_y) & np.isin(source_y, [0, 1]) & np.isfinite(gate_y) & np.isin(gate_y, [0, 1])
    label_match = finite_known & (source_y == gate_y) & np.isfinite(seq_y) & (seq_y == gate_y)
    selected = gate["eligible_24h"].fillna(False).to_numpy(dtype=bool)
    length_positive = np.asarray(ds["L"], dtype=np.int64)[rows] > 0
    split = np.asarray(ds["split"])[rows]
    split_all2 = split == 2
    use = selected & finite_known & label_match & length_positive & split_all2
    counts = {
        "n_gate_rows": int(len(gate)), "n_selected_gate": int(selected.sum()),
        "n_excluded_not_selected": int((~selected).sum()),
        "n_excluded_unknown_or_nonbinary_label": int((~finite_known).sum()),
        "n_excluded_gate_sequence_label_mismatch": int((finite_known & ~label_match).sum()),
        "n_excluded_original_L_zero": int((selected & label_match & ~length_positive).sum()),
        "n_excluded_not_external_all2": int((selected & label_match & length_positive & ~split_all2).sum()),
        "n_qualified_full_input_reference": int(use.sum()),
        "deaths": int(np.sum(gate_y[use] == 1)),
    }
    if counts["n_qualified_full_input_reference"] not in (3964, 3965, 3957):
        raise ValueError(f"{name}: qualification count differs from the documented scope record: {counts}")
    ids = stay[rows[use]]
    labels = seq_y[use].astype(np.int8)
    keys = [{"stay_id": str(s), "y_mort": int(y)} for s, y in sorted(zip(ids, labels), key=lambda x: x[0])]
    return rows[use], counts, keys


def preflight() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for path in (INPUTS, PREFLIGHT, EMBEDDINGS, GPU_MANIFEST, PREDICTIONS, RESULTS, EVAL_KEYS):
        if path.exists():
            raise FileExistsError(f"refusing to overwrite existing no-Mg output: {path}")
    import pandas as pd
    sys.path.insert(0, str(REV / "engine/code"))
    import v4_engine as eng
    registry, reviewed_registry = meta_paths()
    mimic_seq_hash = sha256_file(MIMIC_SEQ)
    if mimic_seq_hash != EXPECTED_MIMIC_SEQUENCE:
        raise ValueError("MIMIC source sequence hash mismatch")
    eligibility_manifest_hash = sha256_file(MIMIC_ELIG / "eligibility_manifest.json")
    if eligibility_manifest_hash != EXPECTED_MIMIC_ELIGIBILITY:
        raise ValueError("MIMIC source eligibility manifest hash mismatch")
    c_eligibility_manifest = json.loads((C_ELIG / "eligibility_manifest.json").read_text())
    c_eligibility_manifest_hash = sha256_file(C_ELIG / "eligibility_manifest.json")
    if (c_eligibility_manifest.get("gate_version") != "c2012_external_reviewed_v4_2"
            or c_eligibility_manifest.get("scope_record_sha256") != EXPECTED_REVIEW):
        raise ValueError("C2012 eligibility manifest does not bind the reviewed external gate")
    review_hash = sha256_file(REVIEW_MEMO)
    # The reviewed eligibility manifest binds the scope-record digest;
    # this records the memo file's own content hash separately.
    review_memo_file_hash = review_hash
    checkpoint_hash = sha256_file(CHECKPOINT)
    if checkpoint_hash != EXPECTED_CHECKPOINT:
        raise ValueError(f"main20k checkpoint hash mismatch: {checkpoint_hash}")
    mimic_paths = eng.resolve_cohort_paths("mimic", verify_exists=True)
    mimic = eng.load_sequence_registered("mimic", mimic_paths)
    eligible, eligible_entry = eng.load_ready_eligibility("mimic", MIMIC_ELIG)
    matched = eng.align_eligible_rows("mimic", mimic, eligible, np.arange(len(mimic["L"])))
    split = np.asarray(mimic["split"], dtype=np.int8)[matched["rows"].to_numpy()]
    all_rows = matched["rows"].to_numpy(dtype=np.int64)
    train_rows = np.sort(all_rows[split == 0]); eval_rows = np.sort(all_rows[split == 2])
    if len(train_rows) == 0 or len(eval_rows) != 14890:
        raise ValueError(f"unexpected MIMIC train/test eligible counts: {len(train_rows)}/{len(eval_rows)}")
    train_ids = np.asarray(mimic["stay_id"])[train_rows].astype(str)
    eval_ids = np.asarray(mimic["stay_id"])[eval_rows].astype(str)
    train_subject = np.asarray(mimic["subject"]).astype(str)[train_rows]
    eval_subject = np.asarray(mimic["subject"]).astype(str)[eval_rows]
    train_y = np.asarray(mimic["y_mort"], dtype=np.int8)[train_rows]
    eval_y = np.asarray(mimic["y_mort"], dtype=np.int8)[eval_rows]
    if len(set(train_subject) & set(eval_subject)):
        raise ValueError("MIMIC split0 and split2 patients overlap")
    arrays = {
        "mimic_model_rows": np.sort(np.concatenate([train_rows, eval_rows])).astype(np.int64),
        "mimic_train_rows": train_rows, "mimic_eval_rows": eval_rows,
        "mimic_train_stay_id": train_ids, "mimic_eval_stay_id": eval_ids,
        "mimic_train_subject": train_subject, "mimic_eval_subject": eval_subject,
        "mimic_train_y": train_y, "mimic_eval_y": eval_y,
    }
    source_vocab = json.loads(MIMIC_VOCAB.read_text())
    var_names = [str(x).strip().lower() for x in source_vocab["vars"]]
    mg_ids = [i + 1 for i, name in enumerate(var_names) if name == "magnesium"]
    if len(mg_ids) != 1:
        raise ValueError(f"source magnesium vocabulary mapping is not unique: {mg_ids}")
    mg_id = int(mg_ids[0])
    c_provenance = {}
    c_keys = {}
    c_names_by_role = {"c2012_a_v4_1": "A", "c2012_b_v4_1": "B", "c2012_c_v4_1": "C"}
    for name in C_NAMES:
        entry = registry["cohorts"][name]
        seq_path = Path(entry["sequence_path"]); split_path = Path(entry["split_path"]); vocab_path = Path(entry["vocab_path"])
        ds = load_sequence(seq_path, split_path)
        c_vocab = json.loads(vocab_path.read_text())
        c_var_names = [str(x).strip().lower() for x in c_vocab["vars"]]
        c_mg_ids = [i + 1 for i, v in enumerate(c_var_names) if v == "magnesium"]
        if c_mg_ids != [mg_id] or c_var_names != var_names:
            raise ValueError(f"{name}: Mg mapping or ordered vocabulary differs from source: {c_mg_ids}")
        rows, counts, keys = c2012_gate_rows(name, ds, C_ELIG / f"{name}.parquet")
        key_hash = sha_ids(name, np.asarray([x["stay_id"] for x in keys]), np.asarray([x["y_mort"] for x in keys]))
        expected_name = c_names_by_role[name]
        if counts["n_qualified_full_input_reference"] != {"A":3964,"B":3965,"C":3957}[expected_name]:
            raise ValueError(f"{name}: exact reviewed count mismatch")
        arrays[f"{name}_rows"] = rows.astype(np.int64)
        arrays[f"{name}_stay_id"] = np.asarray(ds["stay_id"])[rows].astype(str)
        arrays[f"{name}_record_id_proxy"] = np.asarray(ds["stay_id"])[rows].astype(str)
        arrays[f"{name}_y"] = np.asarray(ds["y_mort"], dtype=np.int8)[rows]
        c_keys[name] = {"ids_and_labels": keys, "ids_labels_sha256": key_hash}
        c_provenance[name] = {
            "sequence_path": str(seq_path), "sequence_sha256": sha256_file(seq_path),
            "split_path": str(split_path), "split_sha256": sha256_file(split_path),
            "vocab_path": str(vocab_path), "vocab_sha256": sha256_file(vocab_path),
            "eligibility_table": str(C_ELIG / f"{name}.parquet"),
            "eligibility_table_sha256": sha256_file(C_ELIG / f"{name}.parquet"),
            "reviewed_qualification": counts, "full_input_eval_keys_labels_sha256": key_hash,
            "resampling_unit": "record_id_proxy",
            "n_unique_record_id_proxy": int(len(set(arrays[f"{name}_record_id_proxy"]))),
        }
        if c_provenance[name]["n_unique_record_id_proxy"] != len(rows):
            raise ValueError(f"{name}: RecordID proxy is not one unique key per qualified row")
        log(f"PREFLIGHT {name}: selected={counts['n_qualified_full_input_reference']} ids_sha={key_hash[:16]}")
    arrays["mg_variable_id"] = np.asarray([mg_id], dtype=np.int64)
    arrays["constant_bin"] = np.asarray([CONSTANT_BIN], dtype=np.int64)
    tmp = INPUTS.with_suffix(".npz.tmp")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp, INPUTS)
    with np.load(INPUTS, allow_pickle=False) as z:
        if any(z[k].dtype.kind == "O" for k in z.files):
            raise AssertionError("preflight contains object arrays")
    atomic_json(EVAL_KEYS, {"status":"paired_full_input_reference_keys_saved", "identity_notice":"C2012 RecordID is a stay/admission proxy, not verified person identity", "cohorts":c_keys})
    manifest = {
        "status":"preflight_validated", "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "root_registry_path":str(REGISTRY), "root_registry_sha256":sha256_file(REGISTRY),
        "review_registry_path":str(C_REVIEW_REGISTRY), "review_registry_sha256":sha256_file(C_REVIEW_REGISTRY),
        "scope_record_sha256_bound":c_eligibility_manifest.get("scope_record_sha256"),
        "scope_record_memo_file_sha256":review_memo_file_hash,
        "c2012_eligibility_manifest_sha256":c_eligibility_manifest_hash,
        "mimic":{"sequence_path":str(MIMIC_SEQ),"sequence_sha256":mimic_seq_hash,
                 "split_path":str(mimic_paths.split_path),"split_sha256":sha256_file(mimic_paths.split_path),
                 "vocab_path":str(MIMIC_VOCAB),"vocab_sha256":sha256_file(MIMIC_VOCAB),
                 "eligibility_manifest_sha256":eligibility_manifest_hash,
                 "eligibility_table_sha256":eligible_entry["eligibility_table_sha256"],
                 "source_train_rows_split0":int(len(train_rows)),"eval_rows_split2":int(len(eval_rows)),
                 "source_train_unique_patients":int(len(set(train_subject))),
                 "eval_unique_patients":int(len(set(eval_subject))),
                 "source_train_eval_patient_overlap":0},
        "c2012":c_provenance,
        "checkpoint_path":str(CHECKPOINT),"checkpoint_sha256":sha256_file(CHECKPOINT),
        "encoder_definition_path":str(ENCODER),"encoder_definition_sha256":sha256_file(ENCODER),
        "engine_source_path":str(ENGINE),"engine_source_sha256":sha256_file(ENGINE),
        "mg_vocabulary_name":"magnesium","mg_variable_id":mg_id,"mg_mapping_unique":True,
        "input_condition":"symmetric_mg_omitted", "constant_condition":"B1 on remaining active non-Mg tokens",
        "constant_bin":CONSTANT_BIN,"fit_scope":"eligible MIMIC split0 only; separate C1 scaler/logistic heads per condition",
        "evaluation_scope":"eligible MIMIC split2 and reviewed C2012 A/B/C all2 rows; no target fit/rescaling",
        "bootstrap_replicates":BOOTSTRAPS,"bootstrap_seed":SEED,
        "paired_full_input_key_file_server_only":str(EVAL_KEYS),"paired_full_input_key_file_sha256":sha256_file(EVAL_KEYS),
        "input_arrays_path":str(INPUTS),"input_arrays_sha256":sha256_file(INPUTS),
        "runtime":{"python":sys.version.split()[0],"executable":sys.executable,"numpy":np.__version__,"pandas":pd.__version__},
        "limitations":["retrospective measurement-time inputs; real-time clinical availability unverified","C2012 endpoint is operational proxy under review gate","C2012 RecordID is a stay/admission proxy, not verified patient identity","magnesium harmonization remains uncertain; this is a symmetric Mg-omitted sensitivity","no causal interpretation"],
    }
    atomic_json(PREFLIGHT, manifest)
    log(f"PREFLIGHT_COMPLETE inputs_sha256={sha256_file(INPUTS)} mg_id={mg_id} train0={len(train_rows)} mimic_eval2={len(eval_rows)}")


def encode() -> None:
    if not INPUTS.is_file() or not PREFLIGHT.is_file(): raise FileNotFoundError("run preflight first")
    for path in (EMBEDDINGS, GPU_MANIFEST):
        if path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
    pre=json.loads(PREFLIGHT.read_text())
    if pre.get("status")!="preflight_validated" or sha256_file(INPUTS)!=pre["input_arrays_sha256"]: raise ValueError("preflight identity/hash failed")
    if sha256_file(CHECKPOINT)!=pre["checkpoint_sha256"] or sha256_file(ENCODER)!=pre["encoder_definition_sha256"]: raise ValueError("frozen checkpoint/encoder changed")
    import torch
    if not torch.cuda.is_available(): raise RuntimeError("GPU runtime has no CUDA device")
    torch.cuda.set_per_process_memory_fraction(0.18, 0)
    sys.path.insert(0,str(ROOT/"code")); import fm
    torch.set_num_threads(8)
    if torch.get_num_threads() != 8: raise RuntimeError(f"failed to enforce 8 Torch CPU threads: {torch.get_num_threads()}")
    ck=torch.load(CHECKPOINT,map_location="cpu",weights_only=False); cfg=ck["cfg"]
    nbin=int(ck["state"]["bin.weight"].shape[0]-2)
    model=fm.EHRTransformer(int(cfg["nvar"]),nbin=nbin,d=int(cfg["d"]),layers=int(cfg["layers"]),heads=int(cfg["heads"]),ff=int(cfg["ff"]),nhour=24,dropout=0.1).eval().to("cuda:0")
    model.load_state_dict(ck["state"],strict=True)
    with np.load(INPUTS,allow_pickle=False) as z: inp={k:z[k] for k in z.files}
    mg_id=int(inp["mg_variable_id"][0]); outputs={}; diag={}
    registry=json.loads(REGISTRY.read_text())
    datasets={"mimic":(ROOT/"data/processed/seq_mimic.npz",ROOT/"data/processed/split_mimic.npy")}
    for name in C_NAMES:
        e=registry["cohorts"][name]; datasets[name]=(Path(e["sequence_path"]),Path(e["split_path"]))
    def encode_group(name, ds, rows, ids, labels, subjects, condition):
        if not np.array_equal(np.asarray(ds["stay_id"])[rows].astype(str),ids.astype(str)) or not np.array_equal(np.asarray(ds["y_mort"],dtype=np.int8)[rows],labels.astype(np.int8)):
            raise ValueError(f"{name}: row IDs/labels changed after preflight")
        V,B,H,L=ds["V"][rows],ds["B"][rows],ds["H"][rows],ds["L"][rows]
        Vn,Bn,Hn,B1,stats=prepare_no_mg(V,B,H,L,mg_id)
        # The no-Mg-empty mask is recomputed directly; drop such rows in both conditions.
        keep=(Vn!=0).any(axis=1)
        Vn,Bn,Hn,B1=Vn[keep],Bn[keep],Hn[keep],B1[keep]
        kept_rows=np.asarray(rows)[keep]
        def one(Buse, tag):
            out=np.empty((len(kept_rows),int(model.d)),dtype=np.float32)
            with torch.inference_mode():
                for start in range(0,len(kept_rows),32):
                    ix=kept_rows[start:start+32]
                    width=max(1,int(np.max(ds["L"][ix])))
                    vv=Vn[start:start+len(ix),:width]; bb=Buse[start:start+len(ix),:width]; hh=Hn[start:start+len(ix),:width]
                    if np.any(bb[vv==0]!=0): raise AssertionError("masked or padding B bins are nonzero")
                    z=model.embed(torch.from_numpy(vv.astype(np.int64,copy=False)).to("cuda:0"),torch.from_numpy(bb.astype(np.int64,copy=False)).to("cuda:0"),torch.from_numpy(hh.astype(np.int64,copy=False)).to("cuda:0"),torch.from_numpy(vv==0).to("cuda:0"))
                    out[start:start+len(ix)]=z.detach().cpu().numpy().astype(np.float32,copy=False)
                    if start==0 or start+len(ix)==len(kept_rows) or (start+len(ix))%4096==0: log(f"ENCODE {name}/{condition}/{tag}: {start+len(ix)}/{len(kept_rows)}")
            if not np.isfinite(out).all(): raise FloatingPointError(f"{name}: nonfinite embedding")
            return out
        results={"nomg":one(Bn,"nomg"),"b1":one(B1,"b1")}
        diag[name]={**stats,"n_scored_after_empty_exclusion":int(keep.sum()),"n_excluded_no_non_mg_tokens":int((~keep).sum()),"mg_id":mg_id}
        if name == "mimic":
            source_split=np.asarray(ds["split"],dtype=np.int8)[rows]
            diag[name]["n_excluded_split0_source_no_non_mg_tokens"]=int(np.sum((~keep)&(source_split==0)))
            diag[name]["n_excluded_split2_eval_no_non_mg_tokens"]=int(np.sum((~keep)&(source_split==2)))
        return results, kept_rows, keep
    # Encode eligible source split0 and split2 together, retaining explicit split labels.
    mimic_rows=inp["mimic_model_rows"]
    mimic_ds=load_sequence(*datasets["mimic"])
    # Before any full no-Mg encode, establish standalone encoder parity with the
    # exact immutable main20k/real cache on eight eligible MIMIC split2 rows.
    if sha256_file(REAL_EMBED_MANIFEST) != EXPECTED_REAL_EMBED_MANIFEST:
        raise ValueError("immutable real-embedding manifest SHA256 mismatch")
    real_manifest=json.loads(REAL_EMBED_MANIFEST.read_text())
    real_entry=next((r for r in real_manifest.get("models",[]) if r.get("db")=="mimic" and r.get("model_tag")=="main20k" and r.get("input_condition")=="real"),None)
    if not isinstance(real_entry,dict): raise ValueError("immutable manifest lacks MIMIC main20k/real cache")
    sys.path.insert(0,str(REV/"engine/code")); import v4_engine as eng
    composite_encoder_hash=eng.encoder_code_hash()
    if real_entry.get("checkpoint_sha256")!=pre["checkpoint_sha256"] or real_entry.get("encoder_code_sha256")!=composite_encoder_hash:
        raise ValueError("real cache checkpoint/encoder identity differs from this model")
    parity_rows=inp["mimic_eval_rows"][:8]
    if len(parity_rows)!=8: raise ValueError("need eight eligible MIMIC rows for encoder parity")
    cache_path=Path(real_entry["cache"])
    if cache_path.parent.resolve()!=(REV/"engine/cache").resolve() or not cache_path.is_file():
        raise ValueError("immutable real cache is absent or outside engine/cache")
    with np.load(cache_path,allow_pickle=False) as cache:
        cache_rows=cache["rows"].astype(np.int64); cache_stay=cache["stay_id"].astype(str); cache_subject=cache["subject"].astype(str)
        cache_emb=cache["embedding"].astype(np.float32,copy=False)
        cache_meta=json.loads(str(np.asarray(cache["metadata_json"]).item()))
    cache_pos={int(row):i for i,row in enumerate(cache_rows)}
    if any(int(row) not in cache_pos for row in parity_rows): raise ValueError("parity rows absent from immutable real cache")
    cache_ix=np.asarray([cache_pos[int(row)] for row in parity_rows],dtype=np.int64)
    parity_ids=np.asarray(mimic_ds["stay_id"])[parity_rows].astype(str); parity_subject=np.asarray(mimic_ds["subject"]).astype(str)[parity_rows]
    if not np.array_equal(cache_stay[cache_ix],parity_ids) or not np.array_equal(cache_subject[cache_ix],parity_subject):
        raise ValueError("parity cache IDs differ from source sequence rows")
    for k, expected in {"checkpoint_sha256":pre["checkpoint_sha256"],"encoder_code_sha256":composite_encoder_hash,"input_condition":"real","db":"mimic","model_tag":"main20k"}.items():
        if cache_meta.get(k)!=expected: raise ValueError(f"parity cache metadata differs for {k}")
    width=max(1,int(np.max(mimic_ds["L"][parity_rows])))
    pv=np.asarray(mimic_ds["V"][parity_rows,:width],dtype=np.int64); pb=np.asarray(mimic_ds["B"][parity_rows,:width],dtype=np.int64); ph=np.asarray(mimic_ds["H"][parity_rows,:width],dtype=np.int64)
    with torch.inference_mode():
        pz=model.embed(torch.from_numpy(pv).to("cuda:0"),torch.from_numpy(pb).to("cuda:0"),torch.from_numpy(ph).to("cuda:0"),torch.from_numpy(pv==0).to("cuda:0")).detach().cpu().numpy().astype(np.float32)
    parity_diff=np.abs(pz-cache_emb[cache_ix]); parity_max=float(np.max(parity_diff))
    if not np.allclose(pz,cache_emb[cache_ix],atol=1e-5,rtol=1e-5): raise ValueError(f"standalone model fails real-cache 8-row parity: max_abs={parity_max}")
    parity_result={"status":"passed","n_rows":int(len(parity_rows)),"sequence_rows":parity_rows.astype(int).tolist(),"stay_ids_sha256":hashlib.sha256("\n".join(parity_ids).encode()).hexdigest(),"cache_path":str(cache_path),"cache_sha256":sha256_file(cache_path),"cache_manifest_sha256":sha256_file(REAL_EMBED_MANIFEST),"checkpoint_sha256":pre["checkpoint_sha256"],"encoder_code_sha256":composite_encoder_hash,"max_abs_embedding_difference":parity_max,"atol":1e-5,"rtol":1e-5}
    log(f"REAL_CACHE_PARITY_PASS n=8 max_abs={parity_max:.8g}")
    mimic_ids=np.asarray(mimic_ds["stay_id"])[mimic_rows].astype(str)
    mimic_subject=np.asarray(mimic_ds["subject"]).astype(str)[mimic_rows]
    mimic_y=np.asarray(mimic_ds["y_mort"],dtype=np.int8)[mimic_rows]
    mimic_out,mrows,mkeep=encode_group("mimic",mimic_ds,mimic_rows,mimic_ids,mimic_y,mimic_subject,"source_and_eval")
    outputs.update({"mimic_rows":mrows,"mimic_stay_id":mimic_ids[mkeep],"mimic_subject":mimic_subject[mkeep],"mimic_y":mimic_y[mkeep],"mimic_embedding_nomg":mimic_out["nomg"],"mimic_embedding_b1":mimic_out["b1"]})
    mimic_split=np.asarray(mimic_ds["split"],dtype=np.int8)[mrows]
    outputs["mimic_split"]=mimic_split
    for name in C_NAMES:
        rows=inp[f"{name}_rows"]; ids=inp[f"{name}_stay_id"].astype(str); sub=inp[f"{name}_record_id_proxy"].astype(str); y=inp[f"{name}_y"].astype(np.int8)
        cohort_ds=load_sequence(*datasets[name])
        res,kept_rows,keep=encode_group(name,cohort_ds,rows,ids,y,sub,"target")
        outputs[f"{name}_rows"]=kept_rows; outputs[f"{name}_stay_id"]=ids[keep]; outputs[f"{name}_record_id_proxy"]=sub[keep]; outputs[f"{name}_y"]=y[keep]
        outputs[f"{name}_embedding_nomg"]=res["nomg"]; outputs[f"{name}_embedding_b1"]=res["b1"]
    tmp=EMBEDDINGS.with_suffix(".npz.tmp")
    with tmp.open("wb") as f: np.savez_compressed(f,**outputs); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,EMBEDDINGS)
    with np.load(EMBEDDINGS,allow_pickle=False) as z:
        if any(z[k].dtype.kind=="O" for k in z.files): raise AssertionError("GPU output contains object arrays")
        for k,v in outputs.items():
            if not np.array_equal(z[k],v): raise AssertionError(f"GPU serialization mismatch: {k}")
    report={"status":"gpu_encoding_complete","analysis_script_sha256":sha256_file(SCRIPT_PATH),"input_arrays_sha256":sha256_file(INPUTS),"preflight_sha256":sha256_file(PREFLIGHT),"checkpoint_sha256":sha256_file(CHECKPOINT),"encoder_sha256":sha256_file(ENCODER),"embedding_sha256":sha256_file(EMBEDDINGS),"embedding_size_bytes":EMBEDDINGS.stat().st_size,"mg_variable_id":mg_id,"constant_bin":CONSTANT_BIN,"real_cache_parity_check":parity_result,"diagnostics":diag,"row_identity_and_labels_verified":True,"gpu_runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"torch":str(torch.__version__),"cuda":str(torch.version.cuda),"device":torch.cuda.get_device_name(0),"compute_capability":list(torch.cuda.get_device_capability(0)),"torch_num_threads":int(torch.get_num_threads()),"max_cuda_memory_allocated_bytes":int(torch.cuda.max_memory_allocated(0))},"source_only_head_fit_performed":False,"readout_fitting_performed":False}
    atomic_json(GPU_MANIFEST,report)
    log(f"GPU_ENCODING_COMPLETE pid={os.getpid()} sha256={report['embedding_sha256']}")


def grouped_scores(y: np.ndarray, p: np.ndarray):
    # ROC-AUC and average precision accumulate positives from the highest score down.
    order=np.argsort(-np.asarray(p,dtype=np.float64),kind="mergesort"); ys=np.asarray(y,dtype=np.int8)[order]; ps=np.asarray(p,dtype=np.float64)[order]
    starts=np.r_[0,1+np.flatnonzero(ps[1:]!=ps[:-1])]
    return order,starts,ys


def weighted_metrics(y: np.ndarray, p: np.ndarray, weights: np.ndarray, groups=None) -> dict:
    y=np.asarray(y,dtype=np.int8); p=np.asarray(p,dtype=np.float64); w=np.asarray(weights,dtype=np.float64)
    if groups is None: groups=grouped_scores(y,p)
    order,starts,ys=groups; wg=w[order]
    pos=np.add.reduceat(wg*ys,starts); neg=np.add.reduceat(wg*(1-ys),starts)
    pt=float(pos.sum()); nt=float(neg.sum()); total=pt+nt
    if pt<=0 or nt<=0 or total<=0: raise ValueError("bootstrap sample lacks an outcome class")
    cumpos=np.cumsum(pos); cumneg=np.cumsum(neg)
    auc=float(np.sum(pos*(nt-cumneg+0.5*neg))/(pt*nt))
    ap=float(np.sum((pos/pt)*(cumpos/np.maximum(cumpos+cumneg,1e-300))))
    pc=np.clip(p,1e-15,1-1e-15)
    brier=float(np.sum(w*(pc-y)**2)/total)
    ll=float(-np.sum(w*(y*np.log(pc)+(1-y)*np.log1p(-pc)))/total)
    return {"auroc":auc,"average_precision":ap,"brier":brier,"logloss":ll}


def sklearn_metric_crosscheck() -> dict:
    """Check weighted rank/proper-score implementation against sklearn, including ties/zero weights."""
    from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
    tied_y=np.array([0,1,1,0,1,0],dtype=np.int8)
    tied_p=np.array([0.5,0.5,0.8,0.8,0.5,0.1],dtype=np.float64)
    cluster_y=np.array([0,1,0,0,1,1],dtype=np.int8)
    cluster_p=np.array([0.2,0.6,0.4,0.7,0.8,0.9],dtype=np.float64)
    cases={
        "perfect":(np.array([0,1,0,1]),np.array([0.1,0.9,0.2,0.8]),np.ones(4)),
        "inverse":(np.array([0,1,0,1]),np.array([0.9,0.1,0.8,0.2]),np.ones(4)),
        "ties_and_zero_weights":(tied_y,tied_p,np.array([1,2,1,0,2,1],dtype=np.float64)),
        "null_constant_score":(np.array([0,1,0,0,1]),np.full(5,0.4),np.ones(5)),
        "cluster_bootstrap_weights":(cluster_y,cluster_p,np.array([2,2,1,0,0,1],dtype=np.float64)),
    }
    max_diff=0.0
    for name,(y,p,w) in cases.items():
        actual=weighted_metrics(y,p,w)
        expected={
            "auroc":roc_auc_score(y,p,sample_weight=w),
            "average_precision":average_precision_score(y,p,sample_weight=w),
            "brier":brier_score_loss(y,p,sample_weight=w),
            "logloss":log_loss(y,np.column_stack([1-p,p]),sample_weight=w),
        }
        diff=max(abs(actual[k]-float(expected[k])) for k in expected)
        max_diff=max(max_diff,diff)
        if diff>1e-12: raise AssertionError(f"weighted metric differs from sklearn for {name}: {diff}")
        if name=="perfect" and not (np.isclose(actual["auroc"],1.0) and np.isclose(actual["average_precision"],1.0)):
            raise AssertionError("perfect-score rank metrics are not 1")
        if name=="inverse" and not np.isclose(actual["auroc"],0.0):
            raise AssertionError("inverse-score AUROC is not 0")
        if name=="null_constant_score" and not np.isclose(actual["auroc"],0.5):
            raise AssertionError("constant-score AUROC is not 0.5")
    return {"status":"passed","n_cases":len(cases),"case_names":list(cases),"max_absolute_difference_vs_sklearn":float(max_diff)}


def paired_bootstrap(y,p_nomg,p_b1,cluster,seed:int,reps:int=BOOTSTRAPS,unit:str="patient_cluster") -> dict:
    y=np.asarray(y,dtype=np.int8); cluster=np.asarray(cluster).astype(str)
    unique,inverse=np.unique(cluster,return_inverse=True)
    if len(unique)<2 or len(np.unique(y))!=2: raise ValueError("bootstrap needs multiple clusters and both labels")
    g0=grouped_scores(y,p_nomg); g1=grouped_scores(y,p_b1)
    base=weighted_metrics(y,p_nomg,np.ones(len(y)),g0); alt=weighted_metrics(y,p_b1,np.ones(len(y)),g1)
    keys=("auroc","average_precision","brier","logloss")
    vals={k:np.empty(reps,dtype=np.float64) for k in keys}; rng=np.random.default_rng(seed)
    invalid_draws=0
    for b in range(reps):
        attempts=0
        while True:
            draws=rng.integers(0,len(unique),size=len(unique)); weights=np.bincount(draws,minlength=len(unique))[inverse].astype(np.float64)
            if np.sum(weights*y)>0 and np.sum(weights*(1-y))>0: break
            invalid_draws+=1; attempts+=1
            if attempts>1000: raise RuntimeError("cannot obtain bootstrap draw containing both outcome classes")
        m0=weighted_metrics(y,p_nomg,weights,g0); m1=weighted_metrics(y,p_b1,weights,g1)
        for k in keys: vals[k][b]=m1[k]-m0[k]
        if (b+1)%100==0: log(f"BOOTSTRAP {unit}: {b+1}/{reps}")
    return {"n_rows":int(len(y)),"n_resampling_clusters":int(len(unique)),"bootstrap_replicates":reps,"invalid_single_class_draws_resampled":int(invalid_draws),"seed":seed,"resampling_unit":unit,"base_condition":"symmetric Mg omitted; remaining bins retain source/target values","alternative_condition":"symmetric Mg omitted; all remaining active bins set to B1","base_metrics":base,"alternative_metrics":alt,"change_B1_minus_nomg":{k:alt[k]-base[k] for k in keys},"paired_cluster_bootstrap_95pct_ci":{k:[float(np.quantile(v,0.025)),float(np.quantile(v,0.975))] for k,v in vals.items()},"draws_shared_between_conditions":True}


def analyze() -> None:
    if any(not p.is_file() for p in (INPUTS,PREFLIGHT,EMBEDDINGS,GPU_MANIFEST)): raise FileNotFoundError("preflight and GPU output required")
    if PREDICTIONS.exists() or RESULTS.exists(): raise FileExistsError("refusing to overwrite CPU outputs")
    pre=json.loads(PREFLIGHT.read_text()); gpu=json.loads(GPU_MANIFEST.read_text())
    if sha256_file(INPUTS)!=pre["input_arrays_sha256"] or sha256_file(EMBEDDINGS)!=gpu["embedding_sha256"]: raise ValueError("input or embedding hash mismatch")
    if gpu.get("preflight_sha256")!=sha256_file(PREFLIGHT) or gpu.get("checkpoint_sha256")!=pre["checkpoint_sha256"]: raise ValueError("GPU manifest provenance mismatch")
    if gpu.get("analysis_script_sha256")!=sha256_file(SCRIPT_PATH): raise ValueError("analysis script changed after GPU production")
    import sklearn
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    if sklearn.__version__!="1.8.0": raise RuntimeError(f"CPU consumer must use sklearn 1.8.0; found {sklearn.__version__}")
    with np.load(INPUTS,allow_pickle=False) as z: inp={k:z[k] for k in z.files}
    with np.load(EMBEDDINGS,allow_pickle=False) as z: emb={k:z[k] for k in z.files}
    mrows=emb["mimic_rows"]; split=emb["mimic_split"]; y=emb["mimic_y"]
    is_train=split==0; is_eval=split==2
    if not is_train.any() or not is_eval.any() or len(np.unique(y[is_train]))!=2: raise ValueError("MIMIC source split masks/classes invalid")
    if (not np.isin(mrows[is_train],inp["mimic_train_rows"]).all()
            or not np.isin(mrows[is_eval],inp["mimic_eval_rows"]).all()
            or len(mrows[is_train]) + gpu["diagnostics"]["mimic"]["n_excluded_split0_source_no_non_mg_tokens"] != len(inp["mimic_train_rows"])
            or len(mrows[is_eval]) + gpu["diagnostics"]["mimic"]["n_excluded_split2_eval_no_non_mg_tokens"] != len(inp["mimic_eval_rows"])):
        raise ValueError("MIMIC source fit/eval rows changed beyond reported no-Mg-empty exclusions")
    preds={}; heads={}; fitted={}
    for condition,key in (("nomg","nomg"),("b1","b1")):
        X=emb[f"mimic_embedding_{key}"]
        scaler=StandardScaler().fit(X[is_train])
        head=LogisticRegression(C=1.0,max_iter=3000,random_state=20261122).fit(scaler.transform(X[is_train]),y[is_train])
        if int(head.n_features_in_)!=128: raise AssertionError("source C1 readout dimensions unexpected")
        p=head.predict_proba(scaler.transform(X))[:,1]
        heads[condition]={"scaler_mean":scaler.mean_,"scaler_scale":scaler.scale_,"coef":head.coef_,"intercept":head.intercept_}
        fitted[condition]=(scaler,head)
        preds[f"mimic_p_{condition}"]=p[is_eval]
        log(f"CPU_HEAD_FIT {condition}: split0_rows={int(is_train.sum())} eval_rows={int(is_eval.sum())}")
    prediction_arrays={"mimic_stay_id":emb["mimic_stay_id"][is_eval],"mimic_subject":emb["mimic_subject"][is_eval],"mimic_y":y[is_eval],**preds}
    results={"status":"complete","provenance":{"analysis_script_sha256":sha256_file(SCRIPT_PATH),"preflight_sha256":sha256_file(PREFLIGHT),"input_arrays_sha256":sha256_file(INPUTS),"gpu_manifest_sha256":sha256_file(GPU_MANIFEST),"gpu_embeddings_sha256":sha256_file(EMBEDDINGS),"checkpoint_sha256":pre["checkpoint_sha256"],"encoder_definition_sha256":pre["encoder_definition_sha256"],"bootstrap_replicates":BOOTSTRAPS,"bootstrap_seed":SEED,"cpu_runtime":{"python":sys.version.split()[0],"numpy":np.__version__,"scikit_learn":sklearn.__version__,"executable":sys.executable}},"fitting":{"condition_heads_fit_separately":True,"fit_cohort":"MIMIC","fit_split":0,"fit_eligibility":"reviewed status_consistent_v4_1 eligible rows","classifier":"StandardScaler + LogisticRegression(C=1.0,max_iter=3000,random_state=20261122)","target_fitting_or_rescaling":False},"cohorts":{},"limitations":pre["limitations"],"all_exclusions":{}}
    for name in ("mimic",*C_NAMES):
        if name=="mimic":
            yy=prediction_arrays["mimic_y"]; p0=preds["mimic_p_nomg"]; p1=preds["mimic_p_b1"]; cluster=prediction_arrays["mimic_subject"]; unit="patient_cluster"; ids=prediction_arrays["mimic_stay_id"]
            counts={"n_scored":int(len(yy)),"n_excluded_no_non_mg_tokens":int(gpu["diagnostics"]["mimic"]["n_excluded_no_non_mg_tokens"]),"source_train_rows_split0":int(is_train.sum())}
        else:
            yy=emb[f"{name}_y"]; X0=emb[f"{name}_embedding_nomg"]; X1=emb[f"{name}_embedding_b1"]
            # Source-fitted heads only; no C2012 fit, adaptation or rescaling.
            p0=[];p1=[]
            for condition,key in (("nomg","nomg"),("b1","b1")):
                scaler,head=fitted[condition]
                p=scaler.transform(X0 if key=="nomg" else X1)
                (p0 if key=="nomg" else p1).extend(head.predict_proba(p)[:,1].tolist())
            p0=np.asarray(p0);p1=np.asarray(p1);cluster=emb[f"{name}_record_id_proxy"].astype(str); unit="record_id_proxy"; ids=emb[f"{name}_stay_id"]
            counts={"n_scored":int(len(yy)),"n_excluded_no_non_mg_tokens":int(gpu["diagnostics"][name]["n_excluded_no_non_mg_tokens"]),"reviewed_gate_qualification_n":pre["c2012"][name]["reviewed_qualification"]["n_qualified_full_input_reference"],"n_record_id_proxy":int(len(set(cluster))),"full_input_reference_keys_and_labels_sha256":pre["c2012"][name]["full_input_eval_keys_labels_sha256"]}
        if len(yy)!=len(p0) or len(yy)!=len(p1): raise ValueError(f"{name}: prediction/label row mismatch")
        seed=SEED+(0 if name=="mimic" else C_NAMES.index(name)+1)
        boot=paired_bootstrap(yy,p0,p1,cluster,seed,reps=BOOTSTRAPS,unit=unit)
        results["cohorts"][name]={**counts,"no_mg_vs_b1":boot}
        if name=="mimic":
            prediction_arrays.update({"mimic_stay_id":ids,"mimic_subject":cluster,"mimic_y":yy,"mimic_p_nomg":p0,"mimic_p_b1":p1})
        else:
            prediction_arrays.update({f"{name}_stay_id":ids,f"{name}_record_id_proxy":cluster,f"{name}_y":yy,f"{name}_p_nomg":p0,f"{name}_p_b1":p1})
        results["all_exclusions"][name]={k:v for k,v in counts.items() if "excluded" in k}
        if name == "mimic":
            results["all_exclusions"][name].update({
                "source_split0_no_non_mg_tokens":gpu["diagnostics"]["mimic"]["n_excluded_split0_source_no_non_mg_tokens"],
                "evaluation_split2_no_non_mg_tokens":gpu["diagnostics"]["mimic"]["n_excluded_split2_eval_no_non_mg_tokens"]})
        else:
            results["all_exclusions"][name]["reviewed_gate_exclusions"]={k:v for k,v in pre["c2012"][name]["reviewed_qualification"].items() if "excluded" in k}
        log(f"CPU_ANALYSIS_COMPLETE {name}: n={len(yy)} {unit}={len(set(cluster))}")
    results["provenance"]["source_train_scaler_and_head_hashes"]={k:{n:hashlib.sha256(np.asarray(v).tobytes()).hexdigest() for n,v in head.items()} for k,head in heads.items()}
    tmp=PREDICTIONS.with_suffix(".npz.tmp")
    with tmp.open("wb") as f: np.savez_compressed(f,**prediction_arrays); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,PREDICTIONS)
    results["provenance"]["patient_predictions_server_only_path"]=str(PREDICTIONS)
    results["provenance"]["patient_predictions_sha256"]=sha256_file(PREDICTIONS)
    atomic_json(RESULTS,results)
    log(f"CPU_ANALYSIS_COMPLETE results={RESULTS} sha256={sha256_file(RESULTS)}")


def recover_rank_metrics() -> None:
    """Correct rank metrics from the already saved predictions; never refit or re-encode."""
    import copy
    if any(not path.is_file() for path in (RESULTS,PREDICTIONS,INPUTS,PREFLIGHT,GPU_MANIFEST,EMBEDDINGS)):
        raise FileNotFoundError("original aggregate, saved predictions, and provenance are required")
    if RESULTS_V4_1.exists() or RECOVERY_AUDIT.exists():
        raise FileExistsError("refusing to overwrite corrected no-Mg release")
    invalid_sha=sha256_file(RESULTS); predictions_sha=sha256_file(PREDICTIONS)
    previous=json.loads(RESULTS.read_text())
    if previous.get("status")!="complete" or previous.get("provenance",{}).get("patient_predictions_sha256")!=predictions_sha:
        raise ValueError("saved patient predictions are not bound to the previous aggregate")
    if Path(previous["provenance"]["patient_predictions_server_only_path"]).resolve()!=PREDICTIONS.resolve():
        raise ValueError("previous aggregate references a different patient prediction artifact")
    crosscheck=sklearn_metric_crosscheck()
    with np.load(PREDICTIONS,allow_pickle=False) as archive: pred={k:archive[k] for k in archive.files}
    with np.load(INPUTS,allow_pickle=False) as archive: inp={k:archive[k] for k in archive.files}
    pre=json.loads(PREFLIGHT.read_text()); gpu=json.loads(GPU_MANIFEST.read_text())
    if (gpu.get("embedding_sha256")!=sha256_file(EMBEDDINGS)
            or gpu.get("preflight_sha256")!=sha256_file(PREFLIGHT)
            or pre.get("input_arrays_sha256")!=sha256_file(INPUTS)):
        raise ValueError("recovery input provenance hashes do not match frozen GPU/preflight artifacts")
    if predictions_sha!=previous["provenance"]["patient_predictions_sha256"]:
        raise ValueError("patient prediction hash changed during recovery")
    corrected=copy.deepcopy(previous); corrected["status"]="complete_corrected_rank_metrics"
    corrected["rank_metrics_recovery"]={
        "method":"metrics and paired bootstrap recomputed from saved patient-level predictions only",
        "no_head_refit":True,"no_gpu_reencoding":True,"source_predictions_sha256":predictions_sha,
        "invalid_previous_aggregate_sha256":invalid_sha,"weighted_metric_crosscheck_vs_sklearn":crosscheck,
        "rank_accumulation_order":"descending predicted score; exact-score ties grouped",
    }
    seed_by_cohort={}
    for name in ("mimic",*C_NAMES):
        old_entry=previous["cohorts"][name]; old_boot=old_entry["no_mg_vs_b1"]
        if name=="mimic":
            yy=pred["mimic_y"]; p0=pred["mimic_p_nomg"]; p1=pred["mimic_p_b1"]
            cluster=pred["mimic_subject"].astype(str); ids=pred["mimic_stay_id"].astype(str); unit="patient_cluster"
            expected_ids=inp["mimic_eval_stay_id"].astype(str); expected_y=inp["mimic_eval_y"].astype(np.int8)
            expected_clusters=int(old_boot["n_resampling_clusters"])
        else:
            yy=pred[f"{name}_y"]; p0=pred[f"{name}_p_nomg"]; p1=pred[f"{name}_p_b1"]
            cluster=pred[f"{name}_record_id_proxy"].astype(str); ids=pred[f"{name}_stay_id"].astype(str); unit="record_id_proxy"
            expected_ids=inp[f"{name}_stay_id"].astype(str); expected_y=inp[f"{name}_y"].astype(np.int8)
            expected_clusters=int(old_boot["n_resampling_clusters"])
        if unit!=old_boot["resampling_unit"] or len(yy)!=int(old_boot["n_rows"]) or not np.array_equal(ids,expected_ids) or not np.array_equal(yy,expected_y):
            raise ValueError(f"{name}: saved prediction identity/label rows differ from frozen preflight")
        if (not np.isfinite(p0).all() or not np.isfinite(p1).all()
                or np.any((p0<0)|(p0>1)) or np.any((p1<0)|(p1>1))):
            raise ValueError(f"{name}: saved prediction probabilities are invalid")
        if len(np.unique(cluster))!=expected_clusters:
            raise ValueError(f"{name}: saved prediction cluster count differs from prior report")
        if name!="mimic" and (unit!="record_id_proxy" or expected_clusters!=len(yy)):
            raise ValueError(f"{name}: C2012 proxy must remain one unique RecordID per row")
        if name!="mimic" and not np.array_equal(cluster,ids):
            raise ValueError(f"{name}: C2012 resampling proxy is not the saved RecordID/stay key")
        if int(old_boot["bootstrap_replicates"])!=BOOTSTRAPS:
            raise ValueError(f"{name}: prior bootstrap replicate count is not the fixed 1,000")
        seed=int(old_boot["seed"]); seed_by_cohort[name]=seed
        boot=paired_bootstrap(yy,p0,p1,cluster,seed,reps=BOOTSTRAPS,unit=unit)
        # The rank-order correction cannot alter proper scores or their paired CIs.
        for scope in ("base_metrics","alternative_metrics","change_B1_minus_nomg"):
            for key in ("brier","logloss"):
                if not np.isclose(boot[scope][key],old_boot[scope][key],atol=1e-14,rtol=0):
                    raise AssertionError(f"{name}: Brier/log-loss point estimate changed during rank recovery")
        for key in ("brier","logloss"):
            if not np.allclose(boot["paired_cluster_bootstrap_95pct_ci"][key],old_boot["paired_cluster_bootstrap_95pct_ci"][key],atol=1e-14,rtol=0):
                raise AssertionError(f"{name}: Brier/log-loss CI changed during rank recovery")
        corrected["cohorts"][name]["no_mg_vs_b1"]=boot
        log(f"RANK_RECOVERY {name}: rows={len(yy)} unit={unit} same_seed={seed}")
    corrected["provenance"].update({
        "analysis_script_sha256":sha256_file(SCRIPT_PATH),
        "rank_recovery_input_invalid_aggregate_sha256":invalid_sha,
        "rank_recovery_patient_predictions_sha256":predictions_sha,
        "rank_recovery_no_head_refit":True,"rank_recovery_no_gpu_reencoding":True,
    })
    atomic_json(RESULTS_V4_1,corrected)
    audit={
        "status":"corrected_rank_metrics_from_saved_predictions",
        "prior_aggregate_status":"invalid_rankmetrics_preserved_do_not_report",
        "prior_aggregate_path":str(RESULTS),"prior_aggregate_sha256":invalid_sha,
        "prior_aggregate_unchanged":sha256_file(RESULTS)==invalid_sha,
        "patient_predictions_server_only_path":str(PREDICTIONS),"patient_predictions_sha256":predictions_sha,
        "gpu_embeddings_sha256":sha256_file(EMBEDDINGS),"gpu_manifest_sha256":sha256_file(GPU_MANIFEST),
        "preflight_manifest_sha256":sha256_file(PREFLIGHT),"preflight_arrays_sha256":sha256_file(INPUTS),
        "analysis_script_sha256":sha256_file(SCRIPT_PATH),"no_head_refit":True,"no_gpu_reencoding":True,
        "bootstrap_replicates":BOOTSTRAPS,"bootstrap_seed_by_cohort":seed_by_cohort,
        "metric_crosscheck_vs_sklearn":crosscheck,
        "rank_order_fix":"score groups sorted descending before AUROC/AP accumulation; tied scores remain grouped",
        "brier_and_logloss_unchanged_from_preserved_prior_aggregate":True,
        "corrected_aggregate_path":str(RESULTS_V4_1),"corrected_aggregate_sha256":sha256_file(RESULTS_V4_1),
        "patient_level_rows_or_predictions_copied_local":False,
        "limits":["C2012 resampling unit is RecordID proxy only; person identity remains unverified","retrospective measurement-time inputs; real-time availability unverified","Mg harmonization remains uncertain; this is symmetric source/target omission","no causal interpretation"],
    }
    atomic_json(RECOVERY_AUDIT,audit)
    log(f"RANK_RECOVERY_COMPLETE aggregate={RESULTS_V4_1} sha256={sha256_file(RESULTS_V4_1)} audit={RECOVERY_AUDIT}")


def self_test() -> None:
    v=np.array([[1,2,2,0],[1,0,0,0]],dtype=np.int64); b=np.array([[4,3,8,0],[2,0,0,0]],dtype=np.int64); h=np.array([[1,2,3,0],[1,0,0,0]],dtype=np.int64); l=np.array([3,1])
    vn,bn,hn,b1,diag=prepare_no_mg(v,b,h,l,mg_id=2)
    assert np.array_equal(vn,[[1,0,0,0],[1,0,0,0]]) and np.array_equal(bn,[[4,0,0,0],[2,0,0,0]]) and np.array_equal(hn,[[1,0,0,0],[1,0,0,0]])
    assert np.array_equal(b1,[[1,0,0,0],[1,0,0,0]]) and diag["n_rows_without_non_mg_tokens"]==0
    y=np.array([0,1,0,1]); p=np.array([.1,.9,.2,.8]); z=np.array([.2,.8,.1,.9]); cl=np.array(["p1","p2","p1","p2"])
    out=paired_bootstrap(y,p,z,cl,seed=5,reps=10,unit="patient_cluster")
    assert out["draws_shared_between_conditions"] and out["n_resampling_clusters"]==2
    c=paired_bootstrap(y,p,z,np.array(["r1","r2","r3","r4"]),seed=5,reps=10,unit="record_id_proxy")
    assert c["resampling_unit"]=="record_id_proxy" and c["n_resampling_clusters"]==4
    with __import__("tempfile").TemporaryDirectory() as td:
        q=Path(td)/"x.npz"; np.savez(q,ids=np.array(["Rα","Rβ"]));
        with np.load(q,allow_pickle=False) as zf: assert zf["ids"].tolist()==["Rα","Rβ"] and zf["ids"].dtype.kind=="U"
    print("NO_MG_SYNTHETIC_TESTS_PASS",flush=True)


def main() -> None:
    parser=argparse.ArgumentParser(); parser.add_argument("phase",choices=("preflight","encode","analyze","recover","self-test")); args=parser.parse_args()
    if args.phase=="self-test": self_test()
    elif args.phase=="preflight": preflight()
    elif args.phase=="encode": encode()
    elif args.phase=="recover": recover_rank_metrics()
    else: analyze()


if __name__=="__main__": main()
