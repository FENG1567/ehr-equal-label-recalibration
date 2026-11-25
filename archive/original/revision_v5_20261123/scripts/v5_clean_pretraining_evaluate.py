#!/usr/bin/env python3
"""Freeze source readouts, then score all four V5 clean-pretraining arms."""
from __future__ import annotations
import argparse, hashlib, json, os, pickle, time
from pathlib import Path
for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
if os.environ.get("V5_GPU_LOCK_HELD") != "1":
    raise RuntimeError("GPU evaluator must hold R5/admin/gpu_v5.lock")
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
from scipy.special import logit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, log_loss, roc_auc_score
from sklearn.preprocessing import StandardScaler
from v5_clean_pretraining_gpu import (EHRTransformer, sha256_file, state_hash,
    current_cgroup_memory_max_bytes, current_cgroup_path,
    declared_host_memory_max_bytes, RESOURCE_SCOPE)

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
R4 = ROOT / "revision_v4_20261122"
ELIG = R4 / "temporal/results/eligibility/recovered_reviewed_v4_2"
SEEDS = (20261123, 20261124, 20261125)
ARMS = ("numeric_real", "numeric_constant", "constant_constant", "random_real")
C_GRID = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)
CONTRASTS = {
    "numeric_real_vs_constant_trained_constant": ("numeric_real", "constant_constant"),
    "numeric_real_vs_numeric_constant_input": ("numeric_real", "numeric_constant"),
    "numeric_real_vs_random_initial_real": ("numeric_real", "random_real"),
}
BOOTSTRAPS, MAXLEN, BATCH = 1000, 128, 256

def log(msg):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), msg, flush=True)

def atomic_json(path, obj, replace=False):
    path = Path(path); tmp = path.with_name("." + path.name + ".tmp")
    if tmp.exists() or (path.exists() and not replace):
        raise FileExistsError(f"refusing to overwrite {path} or stale temp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600); os.replace(tmp, path)

def atomic_pickle(path, obj):
    path = Path(path); tmp = path.with_name("." + path.name + ".tmp")
    if tmp.exists() or path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
    with tmp.open("wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL); f.flush(); os.fsync(f.fileno())
    os.chmod(tmp, 0o600); os.replace(tmp, path)

def atomic_npz(path, **arrays):
    path = Path(path); tmp = path.with_name("." + path.name + ".tmp")
    if tmp.exists() or path.exists(): raise FileExistsError(f"refusing to overwrite {path}")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.chmod(tmp, 0o600); os.replace(tmp, path)

def read_tokens(path, expected_sha):
    path = Path(path)
    if not path.is_file() or sha256_file(path) != expected_sha:
        raise RuntimeError(f"token archive missing/hash mismatch: {path}")
    with np.load(path, allow_pickle=False) as z:
        required = {"V", "B", "H", "L", "stay_id", "subject", "summary", "split", "pretrain_member"}
        if set(z.files) != required: raise RuntimeError(f"unexpected token archive fields: {path}")
        d = {k: np.asarray(z[k]) for k in required}
    d["V"] = d["V"].astype(np.int16, copy=False); d["B"] = d["B"].astype(np.int16, copy=False)
    d["H"] = d["H"].astype(np.int16, copy=False); d["L"] = d["L"].astype(np.int16, copy=False)
    d["stay_id"] = d["stay_id"].astype(np.int64, copy=False); d["subject"] = d["subject"].astype(str)
    d["split"] = d["split"].astype(np.int8, copy=False)
    if d["V"].ndim != 2 or d["V"].shape[1] != MAXLEN or d["B"].shape != d["V"].shape or d["H"].shape != d["V"].shape:
        raise RuntimeError("token tensors differ from fixed 128-position layout")
    if any(x.shape != (len(d["V"]),) for x in (d["L"], d["stay_id"], d["subject"], d["split"])):
        raise RuntimeError("token identity/length dimensions mismatch")
    if len(np.unique(d["stay_id"])) != len(d["stay_id"]): raise RuntimeError("duplicate token stay IDs")
    return d

def load_training(run_dir):
    path = run_dir / "private/training/training_manifest.json"
    if not path.is_file(): raise FileNotFoundError("complete training manifest is absent")
    digest = sha256_file(path); doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("schema") != "v5_clean_pretraining_training_v1" or doc.get("status") != "complete":
        raise RuntimeError("training manifest is not complete")
    config = doc.get("config", {})
    if config.get("seeds") != list(SEEDS) or config.get("arms") != ["numeric", "constant"] or config.get("epochs") != 2:
        raise RuntimeError("training manifest seeds, arms, or epoch count differ")
    cps = {}
    for row in doc.get("checkpoints", []):
        key = (int(row["seed"]), str(row["arm"]))
        cp = Path(row["checkpoint_path"])
        if key in cps or not cp.is_file() or sha256_file(cp) != row.get("checkpoint_sha256"):
            raise RuntimeError(f"trained checkpoint missing/hash mismatch: {cp}")
        value = torch.load(cp, map_location="cpu", weights_only=False)
        if value.get("schema") != "v5_clean_pretraining_epoch_v1" or value.get("seed") != key[0] or value.get("arm") != key[1]:
            raise RuntimeError(f"trained checkpoint seed/arm identity mismatch: {cp}")
        if value.get("epoch") != 2 or len(value.get("epoch_losses", [])) != 2:
            raise RuntimeError(f"trained checkpoint is not the frozen two-epoch result: {cp}")
        if state_hash(value["model_state"]) != value.get("model_state_sha256") or value.get("model_state_sha256") != row.get("model_state_sha256"):
            raise RuntimeError(f"trained state hash mismatch: {cp}")
        for field in ("preparation_manifest_sha256", "member_sha256", "token_sha256", "bins_sha256"):
            if value.get(field) != doc.get(field):
                raise RuntimeError(f"trained checkpoint {field} differs from its manifest: {cp}")
        cps[key] = (cp, value)
    if set(cps) != {(s, a) for s in SEEDS for a in ("numeric", "constant")}:
        raise RuntimeError("training manifest lacks one of six seed/arm checkpoints")
    initials = {}
    for row in doc.get("initial_states", []):
        seed = int(row["seed"]); path = Path(row["path"])
        if seed in initials or not path.is_file() or sha256_file(path) != row.get("checkpoint_sha256"):
            raise RuntimeError(f"initial checkpoint missing/hash mismatch: {path}")
        value = torch.load(path, map_location="cpu", weights_only=False)
        if value.get("schema") != "v5_clean_pretraining_initial_state_v1" or value.get("seed") != seed:
            raise RuntimeError(f"initial checkpoint seed identity mismatch: {path}")
        if state_hash(value["state_dict"]) != value.get("state_dict_sha256") or value.get("state_dict_sha256") != row.get("state_dict_sha256"):
            raise RuntimeError(f"initial state hash mismatch: {path}")
        initials[seed] = (path, value)
    if set(initials) != set(SEEDS): raise RuntimeError("training manifest lacks all three initial states")
    for seed in SEEDS:
        initial_sha = initials[seed][1]["state_dict_sha256"]
        for arm in ("numeric", "constant"):
            checkpoint = cps[(seed, arm)][1]
            if checkpoint.get("initial_state_sha256") != initial_sha:
                raise RuntimeError(f"trained {arm} checkpoint does not match seed {seed}'s random initialization")
    for row in doc.get("checkpoints", []):
        seed, arm = int(row["seed"]), str(row["arm"])
        if row.get("initial_state_sha256") != initials[seed][1]["state_dict_sha256"]:
            raise RuntimeError(f"training manifest {arm} row does not match seed {seed}'s random initialization")
    return doc, digest, cps, initials

def read_labels(path, id_col, subject_col):
    frame = pq.read_table(path, columns=[id_col, subject_col, "eligible_24h", "y_mort"]).to_pandas()
    frame = frame.rename(columns={id_col: "stay_id", subject_col: "subject"})
    frame["stay_id"] = pd.to_numeric(frame["stay_id"], errors="raise").astype(np.int64)
    frame["subject"] = frame["subject"].astype(str)
    frame["eligible_24h"] = frame["eligible_24h"].fillna(False).astype(bool)
    if frame["stay_id"].duplicated().any() or frame["subject"].isna().any() or (frame["subject"].str.len() == 0).any():
        raise RuntimeError("eligibility label identities are duplicate or missing")
    frame = frame.loc[frame["eligible_24h"]].copy()
    y = pd.to_numeric(frame["y_mort"], errors="coerce")
    if y.isna().any() or not y.isin([0, 1]).all(): raise RuntimeError("reviewed eligible hospital mortality is incomplete/non-binary")
    frame["y"] = y.astype(np.int8)
    return frame.sort_values("stay_id").reset_index(drop=True)

def align(tokens, frame, dataset):
    position = {int(stay): i for i, stay in enumerate(tokens["stay_id"])}
    missing = [int(s) for s in frame["stay_id"] if int(s) not in position]
    if missing: raise RuntimeError(f"{dataset}: {len(missing)} eligible H24 stays absent from frozen tokens")
    rows = np.asarray([position[int(s)] for s in frame["stay_id"]], dtype=np.int64)
    if not np.array_equal(tokens["subject"][rows], frame["subject"].to_numpy(dtype=str)):
        raise RuntimeError(f"{dataset}: token stay/patient identity mismatch")
    if len(rows) != len(tokens["stay_id"]) or not np.array_equal(frame["stay_id"], tokens["stay_id"]):
        raise RuntimeError(f"{dataset}: token roster differs from the full reviewed H24 eligible roster")
    return {"rows": rows, "stay": frame["stay_id"].to_numpy(dtype=np.int64),
            "subject": frame["subject"].to_numpy(dtype=str), "y": frame["y"].to_numpy(dtype=np.int8),
            "split": tokens["split"][rows].astype(np.int8)}

def make_model(state, device):
    model = EHRTransformer(); model.load_state_dict(state, strict=True)
    return model.eval().to(device)

@torch.no_grad()
def embed(model, tokens, rows, bin_mode, device):
    out = np.zeros((len(rows), model.d), dtype=np.float32)
    lengths = tokens["L"][rows].astype(np.int64)
    valid_out = np.flatnonzero(lengths > 0)
    for start in range(0, len(valid_out), BATCH):
        idx = valid_out[start:start + BATCH]; src = rows[idx]
        width = int(tokens["L"][src].max())
        v = tokens["V"][src, :width].astype(np.int64, copy=True)
        h = tokens["H"][src, :width].astype(np.int64, copy=True)
        if bin_mode == "constant": b = np.where(v > 0, 1, 0).astype(np.int64)
        elif bin_mode == "real": b = tokens["B"][src, :width].astype(np.int64, copy=True)
        else: raise ValueError("unknown bin input mode")
        vt, bt, ht = (torch.from_numpy(x).to(device) for x in (v, b, h))
        out[idx] = model.embed(vt, bt, ht, vt == 0).detach().cpu().numpy()
        if start == 0 or start + BATCH >= len(valid_out):
            log(f"EMBED {bin_mode} progress={min(start+BATCH,len(valid_out))}/{len(valid_out)}")
    if not np.isfinite(out).all(): raise FloatingPointError("non-finite mean pooled embedding")
    return out

def fit_readout(X, y, split, seed):
    train, valid = split == 0, split == 1
    if not train.any() or not valid.any() or len(np.unique(y[train])) != 2:
        raise RuntimeError("readout requires both classes in split0 and a nonempty split1")
    scaler = StandardScaler().fit(X[train]); xt, xv = scaler.transform(X[train]), scaler.transform(X[valid])
    scores, models = {}, {}
    for c in C_GRID:
        clf = LogisticRegression(C=float(c), solver="lbfgs", max_iter=3000, random_state=int(seed))
        clf.fit(xt, y[train]); scores[str(c)] = float(brier_score_loss(y[valid], clf.predict_proba(xv)[:, 1]))
        models[float(c)] = clf
    chosen = min(C_GRID, key=lambda c: (scores[str(c)], c))
    if not np.any(split == 2): raise RuntimeError("source split2 roster is empty")
    return scaler, models[float(chosen)], float(chosen), scores

def metric(y, p, subjects):
    y = np.asarray(y, dtype=np.int8); p = np.asarray(p, dtype=float); subjects = np.asarray(subjects).astype(str)
    if not len(y) or len(y) != len(p) or len(y) != len(subjects): raise RuntimeError("metric arrays are empty/misaligned")
    p_log = np.clip(p, 1e-6, 1 - 1e-6)
    auc = float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else None
    ap = float(average_precision_score(y, p)) if y.sum() else None
    intercept = slope = None; cal_status = "not_estimable_single_class"
    if len(np.unique(y)) == 2:
        try:
            cal = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000, tol=1e-10)
            cal.fit(logit(np.clip(p, 1e-6, 1-1e-6)).reshape(-1, 1), y)
            intercept, slope = float(cal.intercept_[0]), float(cal.coef_[0, 0]); cal_status = "fit"
        except Exception as exc:
            cal_status = "fit_failed:" + type(exc).__name__
    return {"n_stays": int(len(y)), "n_patients": int(len(np.unique(subjects))),
            "events": int(y.sum()), "prevalence": float(y.mean()), "mean_predicted": float(p.mean()),
            "auroc": auc, "average_precision": ap, "brier": float(brier_score_loss(y, p)),
            "logloss": float(log_loss(y, p_log, labels=[0,1])), "calibration_intercept": intercept,
            "calibration_slope": slope, "calibration_status": cal_status}

def prediction_file(path, dataset, seed, arm, stay, subject, y, p, split):
    atomic_npz(path, dataset=np.asarray([dataset]), seed=np.asarray([seed], dtype=np.int32),
        arm=np.asarray([arm]), stay_id=np.asarray(stay, dtype=np.int64), subject=np.asarray(subject).astype(str),
        y=np.asarray(y, dtype=np.int8), p=np.asarray(p, dtype=float), split=np.asarray(split, dtype=np.int8))
    return sha256_file(path)

def read_prediction(path, digest, dataset, seed, arm, stay, subject, y, split):
    if not Path(path).is_file() or sha256_file(path) != digest: raise RuntimeError("prediction checkpoint hash mismatch")
    with np.load(path, allow_pickle=False) as z:
        if z["dataset"].astype(str).tolist() != [dataset] or int(z["seed"][0]) != seed or z["arm"].astype(str).tolist() != [arm]:
            raise RuntimeError("prediction checkpoint metadata differs")
        if not np.array_equal(z["stay_id"], stay) or not np.array_equal(z["subject"].astype(str), subject.astype(str)):
            raise RuntimeError("prediction stay/patient roster differs")
        if not np.array_equal(z["y"], y) or not np.array_equal(z["split"], split):
            raise RuntimeError("prediction outcomes/splits differ")
        p = np.asarray(z["p"], dtype=float)
    if not np.isfinite(p).all() or ((p < 0) | (p > 1)).any(): raise RuntimeError("prediction values outside [0,1]")
    return p

def cluster_bootstrap(y, subjects, pred_by_arm, seed):
    y = np.asarray(y, dtype=np.int8); subjects = np.asarray(subjects).astype(str)
    unique, inverse = np.unique(subjects, return_inverse=True)
    arm_order = list(ARMS); n = len(unique)
    if n == 0: raise RuntimeError("patient-cluster bootstrap has no patients")
    loss_cols = []
    for arm in arm_order: loss_cols.append((pred_by_arm[arm] - y) ** 2)
    for arm in arm_order:
        p = np.clip(pred_by_arm[arm], 1e-6, 1-1e-6)
        loss_cols.append(-(y*np.log(p) + (1-y)*np.log(1-p)))
    sums = np.zeros((n, 8), dtype=float); np.add.at(sums, inverse, np.column_stack(loss_cols))
    rows_per_patient = np.bincount(inverse, minlength=n).astype(float)
    sampled = {name: {"brier": np.empty(BOOTSTRAPS), "logloss": np.empty(BOOTSTRAPS)} for name in CONTRASTS}
    rng = np.random.default_rng(int(seed))
    for b in range(BOOTSTRAPS):
        mult = np.bincount(rng.integers(0, n, size=n), minlength=n).astype(float)
        denom = float(mult @ rows_per_patient); mean = (mult @ sums) / denom
        for name, (left, right) in CONTRASTS.items():
            li, ri = arm_order.index(left), arm_order.index(right)
            sampled[name]["brier"][b] = mean[li]-mean[ri]
            sampled[name]["logloss"][b] = mean[4+li]-mean[4+ri]
    out = {}
    for name, (left, right) in CONTRASTS.items():
        out[name] = {}
        for m in ("brier", "logloss"):
            if m == "brier": point = float(np.mean((pred_by_arm[left]-y)**2-(pred_by_arm[right]-y)**2))
            else:
                lp=np.clip(pred_by_arm[left],1e-6,1-1e-6); rp=np.clip(pred_by_arm[right],1e-6,1-1e-6)
                point=float(np.mean(-(y*np.log(lp)+(1-y)*np.log(1-lp)) + y*np.log(rp)+(1-y)*np.log(1-rp)))
            vals = sampled[name][m]
            out[name][m] = {"left_minus_right": point,
                "patient_cluster_bootstrap_95ci": [float(np.quantile(vals,.025)),float(np.quantile(vals,.975))],
                "bootstrap_replicates": BOOTSTRAPS}
    return out

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root-approved", action="store_true",
        help="required after source C/readouts are frozen and root authorizes label-based scoring")
    parser.add_argument("--resume", action="store_true")
    args=parser.parse_args()
    if not args.root_approved: parser.error("root must separately authorize label-based evaluation")
    run_dir=args.run_dir.resolve(strict=True)
    declared_memory_max_bytes=declared_host_memory_max_bytes()
    host_memory_max_bytes=current_cgroup_memory_max_bytes()
    if host_memory_max_bytes != declared_memory_max_bytes: raise RuntimeError(f"declared host cgroup cap {declared_memory_max_bytes} differs from actual {host_memory_max_bytes}")
    prep_path=run_dir/"preparation_manifest.json"; training_path=run_dir/"private/training/training_manifest.json"
    if not prep_path.is_file(): raise FileNotFoundError("preparation manifest absent")
    prep_sha=sha256_file(prep_path); prep=json.loads(prep_path.read_text(encoding="utf-8"))
    train, train_sha, checkpoints, initials=load_training(run_dir)
    if train.get("preparation_manifest_sha256")!=prep_sha or train.get("run_id")!=prep.get("run_id"):
        raise RuntimeError("training/preparation provenance mismatch")
    eval_dir=run_dir/"private/evaluation"
    if eval_dir.exists() and not args.resume: raise FileExistsError("evaluation outputs exist; use --resume after audit")
    eval_dir.mkdir(parents=True,exist_ok=True,mode=0o700); os.chmod(eval_dir,0o700)
    pred_dir=eval_dir/"predictions"; readout_dir=eval_dir/"readouts"
    pred_dir.mkdir(exist_ok=True,mode=0o700); readout_dir.mkdir(exist_ok=True,mode=0o700)
    final_path=run_dir/"evaluation_results.json"
    if final_path.exists():
        if not args.resume: raise FileExistsError("evaluation results already exist")
        old=json.loads(final_path.read_text(encoding="utf-8"))
        if old.get("training_manifest_sha256")!=train_sha or old.get("preparation_manifest_sha256")!=prep_sha:
            raise RuntimeError("existing evaluation provenance differs")
        if old.get("status")=="complete": log("EVALUATION_ALREADY_COMPLETE"); return

    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    if not torch.cuda.is_available(): raise RuntimeError("isolated GPU runtime reports no CUDA device")
    torch.cuda.set_device(0); prop=torch.cuda.get_device_properties(0); cap=6*1024**3
    fraction=min(.20,cap/float(prop.total_memory)); torch.cuda.set_per_process_memory_fraction(fraction,0)
    torch.cuda.reset_peak_memory_stats(0); device=torch.device("cuda:0")

    info=prep["datasets"]
    tokens={d:read_tokens(info[d]["sequence_path"],info[d]["sequence_sha256"]) for d in ("mimic","eicu","nwicu")}
    source_frame=read_labels(ELIG/"mimic.parquet","stay_id","subject_id")
    source=align(tokens["mimic"],source_frame,"mimic"); split=source["split"]
    if not np.isin(split,[0,1,2]).all(): raise RuntimeError("source split array is outside 0/1/2")
    if int(np.count_nonzero(split == 0)) != 51714:
        raise RuntimeError("MIMIC source tokens do not contain all 51,714 eligible split0 H24 stays")
    source_manifest_path=eval_dir/"source_readouts.json"
    source_meta=None; readouts={}; preds={"mimic":{}}; metrics={"mimic":{str(s):{} for s in SEEDS}}
    if source_manifest_path.exists():
        if not args.resume: raise FileExistsError("source readouts exist; use --resume")
        source_meta=json.loads(source_manifest_path.read_text(encoding="utf-8"))
        if source_meta.get("training_manifest_sha256")!=train_sha or source_meta.get("preparation_manifest_sha256")!=prep_sha:
            raise RuntimeError("saved source readout provenance differs")
        for seed in SEEDS:
            for arm in ARMS:
                rec=source_meta["readouts"][str(seed)][arm]
                rp=Path(rec["readout_path"])
                if not rp.is_file() or sha256_file(rp)!=rec["readout_sha256"]: raise RuntimeError("readout object hash mismatch")
                with rp.open("rb") as f: readouts[(seed,arm)]=pickle.load(f)
                test=split==2; pp=Path(rec["source_prediction_path"])
                p=read_prediction(pp,rec["source_prediction_sha256"],"mimic",seed,arm,
                    source["stay"][test],source["subject"][test],source["y"][test],split[test])
                preds["mimic"].setdefault(seed,{})[arm]=p; metrics["mimic"][str(seed)][arm]=rec["source_test_metrics"]
    else:
        records={str(s):{} for s in SEEDS}
        for seed in SEEDS:
            for arm in ARMS:
                if arm in ("numeric_real","numeric_constant"):
                    model_path,payload=checkpoints[(seed,"numeric")]; state=payload["model_state"]
                    condition="real" if arm=="numeric_real" else "constant"
                elif arm=="constant_constant":
                    model_path,payload=checkpoints[(seed,"constant")]; state=payload["model_state"]; condition="constant"
                else:
                    model_path,payload=initials[seed]; state=payload["state_dict"]; condition="real"
                model=EHRTransformer(); model.load_state_dict(state,strict=True); model.eval().to(device)
                X=embed(model,tokens["mimic"],source["rows"],condition,device)
                scaler,readout,c,grid=fit_readout(X,source["y"],split,seed)
                test=split==2; p=readout.predict_proba(scaler.transform(X[test]))[:,1]
                summary=metric(source["y"][test],p,source["subject"][test])
                rp=readout_dir/f"seed_{seed}_{arm}.pkl"
                atomic_pickle(rp,{"scaler":scaler,"readout":readout,"seed":seed,"arm":arm,"C":c,
                    "encoder_checkpoint_sha256":sha256_file(model_path)})
                pp=pred_dir/f"mimic_split2_seed_{seed}_{arm}.npz"
                ph=prediction_file(pp,"mimic",seed,arm,source["stay"][test],source["subject"][test],
                                   source["y"][test],p,split[test])
                readouts[(seed,arm)]={"scaler":scaler,"readout":readout}
                preds["mimic"].setdefault(seed,{})[arm]=p; metrics["mimic"][str(seed)][arm]=summary
                records[str(seed)][arm]={"C":c,"validation_brier_by_C":grid,
                    "readout_path":str(rp),"readout_sha256":sha256_file(rp),
                    "source_prediction_path":str(pp),"source_prediction_sha256":ph,
                    "source_test_metrics":summary,"encoder_checkpoint_path":str(model_path),
                    "encoder_checkpoint_sha256":sha256_file(model_path),"bin_input_condition":condition}
                log(f"SOURCE_C_FROZEN seed={seed} arm={arm} C={c} split2_n={summary['n_stays']}")
                del model,X; torch.cuda.empty_cache()
        source_meta={"schema":"v5_clean_pretraining_source_readouts_v1","status":"source_readouts_frozen",
            "run_id":prep["run_id"],"preparation_manifest_sha256":prep_sha,"training_manifest_sha256":train_sha,
            "outcome":"reviewed hospital mortality",
            "selection_rule":"split0 scaler/logistic; C by split1 Brier; exact ties choose smaller C; no split0+split1 refit",
            "C_grid":list(C_GRID),"source_test_partition":"split2 scored once","readouts":records}
        atomic_json(source_manifest_path,source_meta)

    # Target outcomes are opened only after the source readout/C manifest is frozen.
    specs={"eicu":(ELIG/"eicu_semantic_v4_2.parquet","stay_id","subject"),
           "nwicu":(ELIG/"nwicu_discharge_v4_1.parquet","stay_id","subject")}
    labels={d:align(tokens[d],read_labels(*specs[d]),d) for d in ("eicu","nwicu")}
    for d in ("eicu", "nwicu"):
        if len(np.unique(labels[d]["y"])) != 2:
            raise RuntimeError(f"{d}: expected both binary outcome classes in the full reviewed eligible H24 cohort")
    for d in labels: metrics[d]={str(s):{} for s in SEEDS}; preds[d]={}
    progress_path=eval_dir/"evaluation_progress.json"
    progress={"schema":"v5_clean_pretraining_evaluation_progress_v1","run_id":prep["run_id"],
        "preparation_manifest_sha256":prep_sha,"training_manifest_sha256":train_sha,
        "status":"evaluation_in_progress","target_prediction_hashes":{}}
    if progress_path.exists():
        if not args.resume: raise FileExistsError("evaluation progress exists; use --resume")
        old=json.loads(progress_path.read_text(encoding="utf-8"))
        for k in ("run_id","preparation_manifest_sha256","training_manifest_sha256"):
            if old.get(k)!=progress.get(k): raise RuntimeError("evaluation progress provenance differs")
        progress=old
    for d in ("eicu","nwicu"):
        a=labels[d]
        for seed in SEEDS:
            for arm in ARMS:
                if arm in ("numeric_real","numeric_constant"): state=checkpoints[(seed,"numeric")][1]["model_state"]; condition="real" if arm=="numeric_real" else "constant"
                elif arm=="constant_constant": state=checkpoints[(seed,"constant")][1]["model_state"]; condition="constant"
                else: state=initials[seed][1]["state_dict"]; condition="real"
                model=EHRTransformer(); model.load_state_dict(state,strict=True); model.eval().to(device)
                X=embed(model,tokens[d],a["rows"],condition,device)
                ro=readouts.get((seed,arm))
                if ro is None:
                    with Path(source_meta["readouts"][str(seed)][arm]["readout_path"]).open("rb") as f: ro=pickle.load(f)
                    readouts[(seed,arm)]=ro
                p=ro["readout"].predict_proba(ro["scaler"].transform(X))[:,1]
                pp=pred_dir/f"{d}_seed_{seed}_{arm}.npz"; key=f"{d}/{seed}/{arm}"
                prior=progress["target_prediction_hashes"].get(key)
                if pp.exists():
                    if not args.resume or not prior: raise FileExistsError("target prediction exists without resume provenance")
                    p=read_prediction(pp,prior,d,seed,arm,a["stay"],a["subject"],a["y"],a["split"])
                else:
                    prior=prediction_file(pp,d,seed,arm,a["stay"],a["subject"],a["y"],p,a["split"])
                    progress["target_prediction_hashes"][key]=prior
                    atomic_json(progress_path,progress,replace=progress_path.exists())
                preds[d].setdefault(seed,{})[arm]=p
                metrics[d][str(seed)][arm]=metric(a["y"],p,a["subject"])
                log(f"TARGET_SCORED dataset={d} seed={seed} arm={arm} n={len(p)}")
                del model,X; torch.cuda.empty_cache()

    bootstrap={}
    for d in ("mimic","eicu","nwicu"):
        bootstrap[d]={}
        if d=="mimic": keep=split==2; y=source["y"][keep]; sub=source["subject"][keep]
        else: y=labels[d]["y"]; sub=labels[d]["subject"]
        for seed in SEEDS:
            bootstrap[d][str(seed)]=cluster_bootstrap(y,sub,preds[d][seed],seed+20261123+(0 if d=="mimic" else 100 if d=="eicu" else 200))
    ranges={}
    for d in bootstrap:
        ranges[d]={}
        for contrast in CONTRASTS:
            ranges[d][contrast]={}
            for measure in ("brier","logloss"):
                vals=[bootstrap[d][str(s)][contrast][measure]["left_minus_right"] for s in SEEDS]
                ranges[d][contrast][measure]={"min":float(min(vals)),"max":float(max(vals)),
                    "interpretation":"descriptive range across three seeds, not a confidence interval"}
    result={"schema":"v5_clean_pretraining_evaluation_v1","status":"complete","run_id":prep["run_id"],
        "created_utc":time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        "preparation_manifest_sha256":prep_sha,"training_manifest_sha256":train_sha,
        "source_readout_manifest_path":str(source_manifest_path),
        "source_readout_manifest_sha256":sha256_file(source_manifest_path),
        "source_C_rule":"fit scaler/logistic split0; select C by split1 Brier; ties smaller C; score split2 once; no split0+split1 refit",
        "arms":{"numeric_real":"numeric-pretrained encoder with observed bins",
            "numeric_constant":"same numeric-pretrained encoder with constant-one bins",
            "constant_constant":"matched constant-trained encoder with constant-one bins",
            "random_real":"same-seed random initial encoder with observed bins"},
        "label_isolation":"Labels were opened only after bins/tokens/training were frozen; target labels were opened only after source C/readouts were frozen.",
        "historical_scope":"New clean source-only sensitivity; no historical checkpoint membership or historical bins are certified.",
        "zero_token_readout_policy":"Zero-token stays remain in the eligible roster with a zero mean-pooled vector; they contribute no pretraining loss positions.",
        "metrics_by_dataset_seed_arm":metrics,
        "paired_patient_cluster_bootstrap":{"replicates":BOOTSTRAPS,"unit":"patient subject",
            "conditional_on":"each frozen seed/model","delta":"left minus right; negative Brier/logloss favors left",
            "contrasts":CONTRASTS,"results":bootstrap},
        "cross_seed_ranges":ranges,
        "private_predictions":{"directory":str(pred_dir),
            "files":{str(p.relative_to(eval_dir)):sha256_file(p) for p in sorted(pred_dir.glob("*.npz"))},
            "patient_keys_stay_server_private":True},
        "runtime":{"host_memory_max_bytes":host_memory_max_bytes,
            "host_memory_declared_max_bytes":declared_memory_max_bytes,
            "resource_scope":RESOURCE_SCOPE,"cgroup_path":current_cgroup_path(),
            "host_cpu_quota_percent":200,
            "torch_version":torch.__version__,"cuda_version":torch.version.cuda,
            "device_name":torch.cuda.get_device_properties(0).name,"gpu_cap_bytes":cap,
            "peak_gpu_allocated_bytes":int(torch.cuda.max_memory_allocated(0)),"threads":2},
        "seed_averaged_predictions_used":False}
    atomic_json(final_path,result); progress["status"]="complete"; atomic_json(progress_path,progress,replace=True)
    log("EVALUATION_COMPLETE path="+str(final_path)+" sha256="+sha256_file(final_path))

if __name__=="__main__": main()
