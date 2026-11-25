#!/usr/bin/env python3
"""Train locked V5 source-only numeric and matched-constant pretraining arms."""
from __future__ import annotations
import argparse, csv, hashlib, json, os, time
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[_name] = "2"
if os.environ.get("V5_GPU_LOCK_HELD") != "1":
    raise RuntimeError("GPU producer must hold R5/admin/gpu_v5.lock")
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path("/data/admin/fengshuo/ehr_fm_20261102")
R5 = ROOT / "revision_v5_20261123"
SEEDS = (20261123, 20261124, 20261125)
ARMS = ("numeric", "constant")
NBINS, MASK_ID, MAXLEN = 10, 11, 128
EPOCHS, BATCH_SIZE = 2, 256
D_MODEL, LAYERS, HEADS, FF_DIM, DROPOUT = 128, 2, 4, 512, 0.1
LR, WEIGHT_DECAY, GRAD_CLIP, MASK_FRAC = 0.001, 0.01, 1.0, 0.2
SCHEDULE_SEED_OFFSET, DROPOUT_SEED_OFFSET = 0x51A7, 0xD09F

def log(message):
    print(time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), message, flush=True)

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

RESOURCE_SCOPE = os.environ.get("V5_RESOURCE_SCOPE", "unspecified")

def declared_host_memory_max_bytes():
    value = int(os.environ.get("V5_HOST_MEMORY_MAX_BYTES", str(3 * 1024 ** 3)))
    if value not in (3 * 1024 ** 3, 12 * 1024 ** 3):
        raise RuntimeError("V5_HOST_MEMORY_MAX_BYTES must declare exactly 3 GiB or 12 GiB")
    return value

def current_cgroup_path():
    for row in Path("/proc/self/cgroup").read_text(encoding="utf-8").splitlines():
        if row.startswith("0::"):
            return row.split("::", 1)[1]
    raise RuntimeError("unified cgroup path is unavailable")

def current_cgroup_memory_max_bytes():
    relative = current_cgroup_path().lstrip("/")
    value = (Path("/sys/fs/cgroup") / relative / "memory.max").read_text(encoding="ascii").strip()
    return None if value == "max" else int(value)


def state_hash(state):
    h = hashlib.sha256()
    for name in sorted(state):
        a = state[name].detach().cpu().contiguous().numpy()
        h.update(name.encode()); h.update(str(a.dtype).encode("ascii"))
        h.update(np.asarray(a.shape, dtype="<i8").tobytes()); h.update(a.tobytes())
    return h.hexdigest()

def atomic_json(path, payload, replace=False):
    path = Path(path); tmp = path.with_name("." + path.name + ".tmp")
    if tmp.exists() or (path.exists() and not replace):
        raise FileExistsError(f"refusing to overwrite {path} or stale temporary")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.chmod(tmp, 0o600); os.replace(tmp, path)

def atomic_torch_save(path, payload):
    path = Path(path); tmp = path.with_name("." + path.name + ".tmp")
    if tmp.exists() or path.exists():
        raise FileExistsError(f"refusing to overwrite {path} or stale temporary")
    torch.save(payload, tmp); os.chmod(tmp, 0o600); os.replace(tmp, path)

def atomic_npz(path, **arrays):
    path = Path(path); tmp = path.with_name("." + path.name + ".tmp")
    if tmp.exists() or path.exists():
        raise FileExistsError(f"refusing to overwrite {path} or stale temporary")
    with tmp.open("wb") as f:
        np.savez_compressed(f, **arrays); f.flush(); os.fsync(f.fileno())
    os.chmod(tmp, 0o600); os.replace(tmp, path)

class EHRTransformer(nn.Module):
    """Local copy of the frozen small masked-EHR transformer architecture."""
    def __init__(self, nvar=48, nbin=NBINS, nhour=24, d=D_MODEL,
                 layers=LAYERS, heads=HEADS, ff=FF_DIM, dropout=DROPOUT):
        super().__init__()
        self.var = nn.Embedding(nvar + 1, d, padding_idx=0)
        self.bin = nn.Embedding(nbin + 2, d, padding_idx=0)
        self.hour = nn.Embedding(nhour, d)
        layer = nn.TransformerEncoderLayer(d, heads, ff, dropout, batch_first=True, norm_first=True)
        self.enc = nn.TransformerEncoder(layer, layers)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, nbin + 1)
        self.d = d
    def forward(self, v, b, h, pad_mask=None):
        return self.norm(self.enc(self.var(v) + self.bin(b) + self.hour(h), src_key_padding_mask=pad_mask))
    def mlm_logits(self, hidden):
        return self.out(hidden)
    def embed(self, v, b, h, pad_mask):
        hidden = self.forward(v, b, h, pad_mask)
        weights = (~pad_mask).float().unsqueeze(-1)
        return (hidden * weights).sum(1) / weights.sum(1).clamp(min=1)

def verify_preparation(run_dir):
    mp = run_dir / "preparation_manifest.json"
    if not mp.is_file():
        raise FileNotFoundError("preparation gate absent; GPU training is not permitted")
    prep_hash = sha256_file(mp); prep = json.loads(mp.read_text(encoding="utf-8"))
    if prep.get("schema") != "v5_clean_pretraining_preparation_v1" or prep.get("status") != "pretraining_inputs_built_not_yet_root_gate_approved":
        raise RuntimeError("preparation manifest is not complete and frozen")
    if "no outcomes" not in prep.get("label_isolation", ""):
        raise RuntimeError("preparation manifest lacks outcome isolation attestation")
    hashes = prep.get("input_hashes", {})
    original, snapshot = hashes.get("source_events", {}), hashes.get("source_events_ram_snapshot", {})
    if not original or not snapshot or original.get("sha256") != snapshot.get("sha256") or original.get("bytes") != snapshot.get("bytes"):
        raise RuntimeError("original source and RAM snapshot provenance mismatch")
    bi = prep.get("binning", {}); bins_path = Path(bi.get("path", ""))
    if not bins_path.is_file() or sha256_file(bins_path) != bi.get("sha256"):
        raise RuntimeError("bins missing or hash mismatch")
    bins = json.loads(bins_path.read_text(encoding="utf-8"))
    scope = "all eligible split0 H24 stays"
    source = prep["source"]
    fit_n = int(source.get("source_eligible_h24_by_split", {}).get("0", 0))
    vocab = prep.get("source_vocab", {})
    variable_order = vocab.get("variable_order")
    if (vocab.get("legacy_sequence_ids_checked") is not True
            or not isinstance(variable_order, list) or len(variable_order) != 48
            or len(set(variable_order)) != 48
            or bins.get("variable_order") != variable_order):
        raise RuntimeError("source token/bin variable order differs from the checked 48-variable vocabulary")
    if bins.get("schema") != "v5_clean_source_only_bins_v1" or int(bins.get("n_bins", -1)) != NBINS:
        raise RuntimeError("source bins do not match the frozen 10-bin schema")
    if fit_n != 51714 or bins.get("source_fit_scope") != scope or int(bins.get("source_fit_stay_n", -1)) != fit_n:
        raise RuntimeError("source bins were not fit on all 51,714 eligible split0 H24 stays")
    if int(source.get("pretraining_member_n", -1)) != 20000:
        raise RuntimeError("pretraining source manifest does not certify exactly 20,000 members")
    if (int(bins.get("pretraining_member_n", -1)) != 20000
            or int(bins.get("pretraining_member_seed", -1)) != 20261123
            or bins.get("pretraining_member_sha256") != source.get("pretraining_member_sha256")):
        raise RuntimeError("bins provenance differs from the fixed 20,000-member roster")
    if bins.get("source_event_sha256") != original.get("sha256") or bins.get("source_event_scan_sha256") != snapshot.get("sha256"):
        raise RuntimeError("bins source hashes differ from the original/snapshot hashes")
    member_path = Path(source.get("pretraining_member_path", ""))
    member_sha = source.get("pretraining_member_sha256")
    if not member_path.is_file() or sha256_file(member_path) != member_sha:
        raise RuntimeError("pretraining member file missing or hash mismatch")
    with member_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["stay_id", "subject"]:
            raise RuntimeError("unexpected member CSV schema")
        members = [(int(row["stay_id"]), str(row["subject"])) for row in reader]
    if len(members) != 20000 or len({x[0] for x in members}) != 20000:
        raise RuntimeError("pretraining member list is not exactly 20,000 unique stays")
    seq = prep.get("datasets", {}).get("mimic", {})
    token_path = Path(seq.get("sequence_path", "")); token_sha = seq.get("sequence_sha256")
    if not token_path.is_file() or sha256_file(token_path) != token_sha:
        raise RuntimeError("MIMIC token archive missing or hash mismatch")
    with np.load(token_path, allow_pickle=False) as z:
        expected = {"V", "B", "H", "L", "stay_id", "subject", "summary", "split", "pretrain_member"}
        if set(z.files) != expected:
            raise RuntimeError("token archive schema differs or includes extra fields")
        V, B, H = (np.asarray(z[k], dtype=np.int16) for k in ("V", "B", "H"))
        L, stay = np.asarray(z["L"], dtype=np.int16), np.asarray(z["stay_id"], dtype=np.int64)
        subject = np.asarray(z["subject"]).astype(str)
        split, flag = np.asarray(z["split"], dtype=np.int8), np.asarray(z["pretrain_member"], dtype=np.int8)
    if V.ndim != 2 or V.shape[1] != MAXLEN or B.shape != V.shape or H.shape != V.shape:
        raise RuntimeError("token tensor dimensions differ from fixed 128-token layout")
    if any(x.shape != (len(V),) for x in (L, stay, subject, split, flag)) or (L < 0).any() or (L > MAXLEN).any():
        raise RuntimeError("token identity/role/length dimensions are invalid")
    if len(np.unique(stay)) != len(stay) or not np.isin(split, [0, 1, 2]).all():
        raise RuntimeError("MIMIC token stay IDs or source split roles are invalid")
    split_counts = source.get("source_eligible_h24_by_split", {})
    for role in (0, 1, 2):
        expected_n = int(split_counts.get(str(role), 0))
        if int(np.count_nonzero(split == role)) != expected_n:
            raise RuntimeError(f"MIMIC token roster count differs for source split {role}")
    valid = np.arange(MAXLEN)[None, :] < L[:, None]
    if not np.array_equal(V > 0, valid):
        raise RuntimeError("padding locations differ from token lengths")
    if ((V[valid] < 1) | (V[valid] > 48)).any() or ((B[valid] < 1) | (B[valid] > NBINS)).any() or ((H[valid] < 0) | (H[valid] > 23)).any():
        raise RuntimeError("token vocabulary/bin/hour values exceed frozen limits")
    if np.any(B[~valid] != 0) or np.any(H[~valid] != 0) or not np.isin(flag, [0, 1]).all():
        raise RuntimeError("padding or membership flags are invalid")
    rows = np.flatnonzero(flag == 1)
    if len(rows) != 20000 or not np.all(split[rows] == 0):
        raise RuntimeError("token archive does not mark exactly 20,000 split0 training members")
    if not np.array_equal(stay[rows], np.asarray([x[0] for x in members], dtype=np.int64)) or not np.array_equal(subject[rows], np.asarray([x[1] for x in members], dtype=str)):
        raise RuntimeError("token member identity/order differs from member CSV")
    return {
        "prep": prep, "prep_path": str(mp), "prep_sha": prep_hash,
        "bins_path": str(bins_path), "bins_sha": bi["sha256"],
        "token_path": str(token_path), "token_sha": token_sha,
        "member_path": str(member_path), "member_sha": member_sha, "member_n": len(members),
        "member_positions": rows, "V": V[rows], "B": B[rows], "H": H[rows], "L": L[rows],
        "member_stay_sha": hashlib.sha256(stay[rows].astype("<i8", copy=False).tobytes()).hexdigest(),
        "source_fit_stay_n": fit_n, "source_event_sha": original["sha256"],
        "source_snapshot_sha": snapshot["sha256"],
    }

def schedule_hashes(schedule):
    return {k: hashlib.sha256(np.ascontiguousarray(v).tobytes()).hexdigest() for k, v in schedule.items()}

def make_schedule(lengths, seed):
    n = len(lengths); rng = np.random.default_rng(np.random.SeedSequence([int(seed), SCHEDULE_SEED_OFFSET]))
    order = np.empty((EPOCHS, n), dtype=np.int32)
    mask = np.zeros((EPOCHS, n, MAXLEN), dtype=np.uint8)
    kind = np.zeros_like(mask); random_bin = np.zeros_like(mask)
    for ep in range(EPOCHS):
        order[ep] = rng.permutation(n)
        for row, length0 in enumerate(lengths):
            length = int(length0)
            if length == 0:
                continue
            selected = rng.random(length) < MASK_FRAC
            if not selected.any():
                selected[int(rng.integers(length))] = True
            draw = rng.random(length); k = np.zeros(length, dtype=np.uint8)
            k[selected & (draw < .8)] = 1
            k[selected & (draw >= .8) & (draw < .9)] = 2
            k[selected & (draw >= .9)] = 3
            mask[ep, row, :length] = selected
            kind[ep, row, :length] = k
            random_bin[ep, row, :length] = rng.integers(1, NBINS + 1, size=length, dtype=np.uint8)
    return {"permutation": order, "mask": mask, "replacement_kind": kind, "random_bin": random_bin}

def load_schedule(seed_dir, lengths, seed, member_sha, resume):
    path, meta_path = seed_dir / "mask_schedule.npz", seed_dir / "mask_schedule.json"
    if path.exists() or meta_path.exists():
        if not resume or not path.is_file() or not meta_path.is_file():
            raise RuntimeError("partial/existing schedule requires --resume and both files")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("seed") != seed or meta.get("member_sha256") != member_sha or meta.get("n_members") != len(lengths):
            raise RuntimeError("saved schedule provenance differs")
        if sha256_file(path) != meta.get("schedule_file_sha256"):
            raise RuntimeError("saved schedule file hash differs")
        with np.load(path, allow_pickle=False) as z:
            sch = {k: np.asarray(z[k]) for k in ("permutation", "mask", "replacement_kind", "random_bin")}
        if schedule_hashes(sch) != meta.get("array_sha256"):
            raise RuntimeError("saved schedule array hashes differ")
        return sch, meta
    sch = make_schedule(lengths, seed)
    seed_dir.mkdir(parents=True, exist_ok=True, mode=0o700); os.chmod(seed_dir, 0o700)
    atomic_npz(path, **sch)
    meta = {
        "seed": int(seed), "schedule_seed_parts": [int(seed), int(SCHEDULE_SEED_OFFSET)],
        "member_sha256": member_sha, "n_members": int(len(lengths)),
        "epochs": EPOCHS, "max_len": MAXLEN, "mask_fraction": MASK_FRAC,
        "selected_replacement_probability": {"mask": .8, "random_bin": .1, "unchanged": .1},
        "array_sha256": schedule_hashes(sch), "schedule_file_sha256": sha256_file(path),
    }
    atomic_json(meta_path, meta)
    return sch, meta

def initial_state(seed, path, resume):
    if path.exists():
        if not resume:
            raise FileExistsError("initial state exists; use --resume after audit")
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("seed") != seed or state_hash(saved["state_dict"]) != saved.get("state_dict_sha256"):
            raise RuntimeError("initial state checkpoint provenance/hash mismatch")
        return saved["state_dict"], saved["state_dict_sha256"]
    torch.manual_seed(seed)
    model = EHRTransformer()
    state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    digest = state_hash(state)
    atomic_torch_save(path, {"schema": "v5_clean_pretraining_initial_state_v1",
        "seed": int(seed), "state_dict": state, "state_dict_sha256": digest,
        "model_config": {"nvar": 48, "nbin": NBINS, "d": D_MODEL, "layers": LAYERS,
                         "heads": HEADS, "ff": FF_DIM, "dropout": DROPOUT}})
    return state, digest

def train_arm(seed, arm, initial, initial_sha, gate, schedule, schedule_meta,
              arm_dir, resume, device):
    arm_dir.mkdir(parents=True, exist_ok=True, mode=0o700); os.chmod(arm_dir, 0o700)
    model = EHRTransformer().to(device); model.load_state_dict(initial, strict=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    losses, completed = [], 0
    for epoch in range(1, EPOCHS + 1):
        ckpt = arm_dir / f"epoch_{epoch:02d}.pt"
        if ckpt.exists():
            if not resume:
                raise FileExistsError(f"epoch checkpoint exists; use --resume: {ckpt}")
            saved = torch.load(ckpt, map_location="cpu", weights_only=False)
            fields = {"seed": seed, "arm": arm, "epoch": epoch,
                      "initial_state_sha256": initial_sha, "member_sha256": gate["member_sha"],
                      "token_sha256": gate["token_sha"], "bins_sha256": gate["bins_sha"],
                      "schedule_file_sha256": schedule_meta["schedule_file_sha256"]}
            if any(saved.get(k) != v for k, v in fields.items()):
                raise RuntimeError(f"epoch checkpoint provenance mismatch: {ckpt}")
            if state_hash(saved["model_state"]) != saved.get("model_state_sha256"):
                raise RuntimeError(f"epoch model state hash mismatch: {ckpt}")
            model.load_state_dict(saved["model_state"], strict=True)
            optimizer.load_state_dict(saved["optimizer_state"])
            losses, completed = list(saved["epoch_losses"]), epoch
            continue
        if completed != epoch - 1:
            raise RuntimeError("epoch checkpoints are not contiguous")
        model.train(); perm = schedule["permutation"][epoch - 1]
        masks = schedule["mask"][epoch - 1]
        kinds = schedule["replacement_kind"][epoch - 1]
        random_bins = schedule["random_bin"][epoch - 1]
        dropout_seed = int(seed + DROPOUT_SEED_OFFSET + epoch)
        torch.manual_seed(dropout_seed); torch.cuda.manual_seed_all(dropout_seed)
        loss_sum, token_n, updates = 0.0, 0, 0
        started = time.monotonic()
        for batch_no, begin in enumerate(range(0, gate["member_n"], BATCH_SIZE), start=1):
            rows = perm[begin:begin + BATCH_SIZE].astype(np.int64)
            rows = rows[gate["L"][rows] > 0]
            if not len(rows):
                continue
            width = int(gate["L"][rows].max())
            v = gate["V"][rows, :width].astype(np.int64, copy=True)
            h = gate["H"][rows, :width].astype(np.int64, copy=True)
            original = gate["B"][rows, :width].astype(np.int64, copy=True)
            base = np.ones_like(original) if arm == "constant" else original.copy()
            base[v == 0] = 0
            mask = masks[rows, :width].astype(bool)
            target = np.full(base.shape, -100, dtype=np.int64)
            target[mask] = 1 if arm == "constant" else original[mask]
            inp = base.copy(); k = kinds[rows, :width]
            inp[k == 1] = MASK_ID
            inp[k == 2] = random_bins[rows, :width][k == 2]
            if np.any(inp[v == 0] != 0) or np.any(target[v == 0] != -100):
                raise RuntimeError("schedule changed padding positions")
            masked_n = int(mask.sum())
            if not masked_n:
                continue
            vt, bt, ht, tt = (torch.from_numpy(a).to(device) for a in (v, inp, h, target))
            logits = model.mlm_logits(model(vt, bt, ht, vt == 0))
            loss = F.cross_entropy(logits.reshape(-1, NBINS + 1), tt.reshape(-1),
                                   ignore_index=-100, reduction="sum") / masked_n
            optimizer.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP); optimizer.step()
            loss_sum += float(loss.detach().cpu()) * masked_n; token_n += masked_n; updates += 1
            if batch_no % 20 == 0:
                log(f"TRAIN seed={seed} arm={arm} epoch={epoch} batch={batch_no} masked={token_n}")
        if not token_n or not updates:
            raise RuntimeError(f"no masked-bin updates for seed={seed} arm={arm} epoch={epoch}")
        epoch_loss = loss_sum / token_n; losses.append(float(epoch_loss))
        torch.cuda.synchronize(); peak = int(torch.cuda.max_memory_allocated(0))
        if peak > 6 * 1024 ** 3:
            raise MemoryError(f"GPU allocation exceeded 6 GiB: {peak}")
        state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        payload = {
            "schema": "v5_clean_pretraining_epoch_v1", "seed": seed, "arm": arm, "epoch": epoch,
            "epoch_losses": losses, "epoch_loss": float(epoch_loss), "update_n": updates,
            "masked_token_n": token_n, "initial_state_sha256": initial_sha,
            "member_sha256": gate["member_sha"], "member_stay_sha256": gate["member_stay_sha"],
            "token_sha256": gate["token_sha"], "bins_sha256": gate["bins_sha"],
            "preparation_manifest_sha256": gate["prep_sha"],
            "schedule_file_sha256": schedule_meta["schedule_file_sha256"],
            "schedule_array_sha256": schedule_meta["array_sha256"],
            "dropout_seed": dropout_seed,
            "training_config": {"epochs": EPOCHS, "batch_size": BATCH_SIZE,
                "learning_rate": LR, "weight_decay": WEIGHT_DECAY,
                "gradient_norm_cap": GRAD_CLIP, "mask_fraction": MASK_FRAC,
                "threads": 2, "cpu_affinity": "0-3 maximum"},
            "model_state": state, "model_state_sha256": state_hash(state),
            "optimizer_state": optimizer.state_dict(),
        }
        atomic_torch_save(ckpt, payload); completed = epoch
        log(f"TRAIN_EPOCH_COMPLETE seed={seed} arm={arm} epoch={epoch} loss={epoch_loss:.8f} "
            f"updates={updates} masked={token_n} peak_gpu_bytes={peak} wall_seconds={time.monotonic()-started:.1f}")
    final = arm_dir / f"epoch_{EPOCHS:02d}.pt"
    saved = torch.load(final, map_location="cpu", weights_only=False)
    if state_hash(saved["model_state"]) != saved.get("model_state_sha256"):
        raise RuntimeError("final checkpoint model state hash mismatch")
    return {"seed": seed, "arm": arm, "checkpoint_path": str(final),
            "checkpoint_sha256": sha256_file(final), "model_state_sha256": saved["model_state_sha256"],
            "epoch_losses": saved["epoch_losses"], "initial_state_sha256": initial_sha,
            "schedule_file_sha256": schedule_meta["schedule_file_sha256"],
            "schedule_array_sha256": schedule_meta["array_sha256"]}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--root-approved", action="store_true",
                        help="required only after root accepts the frozen bins/member/token hashes")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not args.root_approved:
        parser.error("root must separately approve the frozen artifacts before GPU training")
    run_dir = args.run_dir.resolve(strict=True)
    declared_memory_max_bytes = declared_host_memory_max_bytes()
    host_memory_max_bytes = current_cgroup_memory_max_bytes()
    if host_memory_max_bytes != declared_memory_max_bytes:
        raise RuntimeError(f"declared host cgroup cap {declared_memory_max_bytes} differs from actual {host_memory_max_bytes}")
    training_dir = run_dir / "private" / "training"
    if training_dir.exists() and not args.resume:
        raise FileExistsError(f"training output exists; use --resume after audit: {training_dir}")
    training_dir.mkdir(parents=True, exist_ok=True, mode=0o700); os.chmod(training_dir, 0o700)
    final_manifest_path = training_dir / "training_manifest.json"
    if final_manifest_path.exists():
        if not args.resume:
            raise FileExistsError("training manifest already exists")
        old = json.loads(final_manifest_path.read_text(encoding="utf-8"))
        if old.get("status") != "complete":
            raise RuntimeError("existing training manifest is not complete; inspect before resume")
        log("TRAINING_ALREADY_COMPLETE manifest=" + str(final_manifest_path)); return
    torch.set_num_threads(2); torch.set_num_interop_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError("isolated GPU runtime reports no CUDA device")
    torch.cuda.set_device(0); prop = torch.cuda.get_device_properties(0)
    cap = 6 * 1024 ** 3; fraction = min(.20, cap / float(prop.total_memory))
    torch.cuda.set_per_process_memory_fraction(fraction, 0); torch.cuda.reset_peak_memory_stats(0)
    device = torch.device("cuda:0")
    gate = verify_preparation(run_dir)
    log(f"PREPARATION_GATE_OK members={gate['member_n']} source_bin_fit_stays={gate['source_fit_stay_n']} token_sha={gate['token_sha']}")
    script_sha = sha256_file(Path(__file__).resolve())
    fm_sha = gate["prep"]["code_hashes"]["fm_py"]["sha256"]
    progress_path = training_dir / "training_progress.json"
    progress = {"schema": "v5_clean_pretraining_progress_v1",
        "run_id": gate["prep"]["run_id"], "training_script_sha256": script_sha,
        "preparation_manifest_sha256": gate["prep_sha"], "member_sha256": gate["member_sha"],
        "member_stay_sha256": gate["member_stay_sha"], "token_sha256": gate["token_sha"],
        "bins_sha256": gate["bins_sha"], "status": "training_in_progress",
        "completed_arms": [], "schedule_hashes": {}}
    if progress_path.exists():
        if not args.resume:
            raise FileExistsError("training progress exists; use --resume after audit")
        old = json.loads(progress_path.read_text(encoding="utf-8"))
        for k, v in progress.items():
            if k not in ("status", "completed_arms", "schedule_hashes") and old.get(k) != v:
                raise RuntimeError(f"training resume provenance mismatch in {k}")
        progress = old
    results = []
    initial_records = []
    for seed in SEEDS:
        seed_dir = training_dir / f"seed_{seed}"
        seed_dir.mkdir(parents=True, exist_ok=True, mode=0o700); os.chmod(seed_dir, 0o700)
        schedule, meta = load_schedule(seed_dir, gate["L"], seed, gate["member_sha"], args.resume)
        progress["schedule_hashes"][str(seed)] = {
            "schedule_file_sha256": meta["schedule_file_sha256"], "array_sha256": meta["array_sha256"]}
        init_path = seed_dir / "initial_state.pt"
        initial, initial_sha = initial_state(seed, init_path, args.resume)
        initial_records.append({"seed": int(seed), "path": str(init_path),
                                "checkpoint_sha256": sha256_file(init_path),
                                "state_dict_sha256": initial_sha})
        for arm in ARMS:
            result = train_arm(seed, arm, initial, initial_sha, gate, schedule, meta,
                               seed_dir / arm, args.resume, device)
            results.append(result)
            progress["completed_arms"] = [
                {"seed": r["seed"], "arm": r["arm"], "checkpoint_sha256": r["checkpoint_sha256"]}
                for r in results]
            atomic_json(progress_path, progress, replace=progress_path.exists())
            log(f"TRAIN_ARM_COMPLETE seed={seed} arm={arm} sha256={result['checkpoint_sha256']}")
            torch.cuda.empty_cache()
    required = {(s, a) for s in SEEDS for a in ARMS}
    if {(r["seed"], r["arm"]) for r in results} != required:
        raise RuntimeError("not all six seed/arm models completed")
    manifest = {
        "schema": "v5_clean_pretraining_training_v1", "status": "complete",
        "run_id": gate["prep"]["run_id"], "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "training_script_path": str(Path(__file__).resolve()), "training_script_sha256": script_sha,
        "architecture_source": "local copy of frozen code/fm.py EHRTransformer; no writable-source import",
        "architecture_source_sha256": fm_sha, "preparation_manifest_path": gate["prep_path"],
        "historical_scope": "New clean source-only sensitivity; no historical checkpoint membership or historical bins are certified.",
        "preparation_manifest_sha256": gate["prep_sha"], "token_path": gate["token_path"],
        "token_sha256": gate["token_sha"], "bins_path": gate["bins_path"], "bins_sha256": gate["bins_sha"],
        "member_path": gate["member_path"], "member_sha256": gate["member_sha"],
        "member_n": gate["member_n"], "member_stay_sha256": gate["member_stay_sha"],
        "source_fit_scope": "all eligible split0 H24 stays", "source_fit_stay_n": gate["source_fit_stay_n"],
        "source_event_sha256": gate["source_event_sha"], "source_event_snapshot_sha256": gate["source_snapshot_sha"],
        "label_isolation": "No labels are read; training consumes only the frozen token archive and outcome-free member CSV.",
        "config": {"seeds": list(SEEDS), "arms": list(ARMS), "epochs": EPOCHS,
            "batch_size": BATCH_SIZE, "d": D_MODEL, "layers": LAYERS, "heads": HEADS,
            "feed_forward": FF_DIM, "dropout": DROPOUT, "learning_rate": LR,
            "weight_decay": WEIGHT_DECAY, "gradient_norm_cap": GRAD_CLIP, "mask_fraction": MASK_FRAC,
            "selected_replacements": {"mask_token": .8, "uniform_random_bin_1_to_10": .1, "unchanged": .1},
            "constant_arm": "all observed inputs and reconstruction targets set to 1 before masking",
            "paired_schedules": "same member order, per-epoch permutation, mask, replacement kind, random-bin schedule, and seed-derived per-epoch dropout RNG for both arms",
            "zero_token_members": "retained in the 20,000-member roster and permutation; no loss positions are available for them",
            "threads": 2, "cpu_affinity": "0-3 maximum", "gpu_memory_cap_bytes": cap,
            "optimizer": "AdamW"},
        "schedule_hashes": progress["schedule_hashes"], "initial_states": initial_records,
        "dropout_rng_seed_by_seed_epoch": {
            str(seed): [int(seed + DROPOUT_SEED_OFFSET + epoch) for epoch in range(1, EPOCHS + 1)]
            for seed in SEEDS},
        "checkpoints": results,
        "runtime": {"host_memory_max_bytes": host_memory_max_bytes,
            "host_memory_declared_max_bytes": declared_memory_max_bytes,
            "resource_scope": RESOURCE_SCOPE, "cgroup_path": current_cgroup_path(),
            "host_cpu_quota_percent": 200,
            "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
            "device_name": prop.name, "peak_gpu_allocated_bytes": int(torch.cuda.max_memory_allocated(0)),
            "per_process_memory_fraction": fraction}}
    atomic_json(final_manifest_path, manifest)
    progress["status"] = "complete"; atomic_json(progress_path, progress, replace=True)
    log("TRAINING_COMPLETE manifest=" + str(final_manifest_path) + " sha256=" + sha256_file(final_manifest_path))

if __name__ == "__main__":
    main()
