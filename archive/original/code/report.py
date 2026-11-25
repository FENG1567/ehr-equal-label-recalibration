"""Consolidate result tables and draw the two headline figures.

  python report.py
Outputs: results/tables/summary_*.csv, results/figures/*.png
"""
import os, glob, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/data/admin/fengshuo/ehr_fm_20261102"
RES = f"{ROOT}/results/tables"
FIG = f"{ROOT}/results/figures"
os.makedirs(FIG, exist_ok=True)
DB_LABEL = {"mimic_test": "MIMIC-IV (test)", "mimic_val": "MIMIC-IV (val)",
            "eicu_all": "eICU-CRD", "nwicu_all": "NWICU"}
COL = {"mimic_test": "#1f77b4", "eicu_all": "#d62728", "nwicu_all": "#2ca02c"}


def load(pattern):
    fs = sorted(glob.glob(f"{RES}/{pattern}"))
    return pd.concat([pd.read_csv(f) for f in fs], ignore_index=True) if fs else pd.DataFrame()


def main():
    probe = load("probe_main20k_*.csv")
    rand = load("randprobe_*.csv")
    tab = load("baseline_tab_*.csv")
    ft = load("finetune_*.csv")
    grid = load("scaling_grid.csv")

    if len(probe):
        probe["kind"] = "pretrained+probe"
        probe.to_csv(f"{RES}/summary_probe.csv", index=False)
        print("\n=== pretrained encoder, zero target labels ===")
        print(probe[["task", "db", "n", "prev", "auroc", "slope", "intercept_fixed_slope",
                     "ici", "brier", "bss", "err_full", "err_80", "rrr_80"]]
              .round(3).to_string(index=False))

    allm = pd.concat([d for d in [probe, rand, tab, ft] if len(d)], ignore_index=True)
    if len(allm):
        allm.to_csv(f"{RES}/summary_all_models.csv", index=False)

    # ---------------- figure 1: AUROC + ICI by db x task
    if len(probe):
        tasks = [t for t in ["mort", "aki", "long"] if t in set(probe.task)]
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
        w = 0.26
        for ax, metric, title in [(axes[0], "auroc", "AUROC"), (axes[1], "ici", "ICI (lower better)")]:
            for k, db in enumerate(["mimic_test", "eicu_all", "nwicu_all"]):
                vals = []
                for t in tasks:
                    sub = probe[(probe.task == t) & (probe.db == db)]
                    vals.append(float(sub[metric].iloc[0]) if len(sub) else np.nan)
                ax.bar(np.arange(len(tasks)) + (k - 1) * w, vals, w, label=DB_LABEL[db], color=COL[db])
            ax.set_xticks(np.arange(len(tasks))); ax.set_xticklabels(tasks)
            ax.set_title(title); ax.grid(axis="y", alpha=.3)
        axes[0].axhline(0.5, ls="--", c="gray", lw=1)
        axes[0].legend(fontsize=8)
        plt.tight_layout(); plt.savefig(f"{FIG}/fig1_auroc_ici.png", dpi=160); plt.close()

    # ---------------- figure 2: calibration slope vs 1
    if len(probe):
        fig, ax = plt.subplots(figsize=(5.6, 3.4))
        for k, db in enumerate(["mimic_test", "eicu_all", "nwicu_all"]):
            sub = probe[probe.db == db]
            ax.scatter(sub["slope"], sub["ici"], s=60, label=DB_LABEL[db], color=COL[db])
            for _, r in sub.iterrows():
                ax.annotate(r["task"], (r["slope"], r["ici"]), fontsize=7, xytext=(3, 3),
                            textcoords="offset points")
        ax.axvline(1.0, ls="--", c="gray")
        ax.set_xlabel("calibration slope (ideal = 1)"); ax.set_ylabel("ICI")
        ax.set_title("Calibration shift across databases"); ax.legend(fontsize=8); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{FIG}/fig2_calibration.png", dpi=160); plt.close()

    # ---------------- figure 3: rejection curves (if saved)
    rc = load("reject_curves.csv")
    if len(rc):
        if "task" not in rc.columns:
            rc["task"] = rc["tag"].str.split("_").str[-1]
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        for (task, db), g in rc.groupby(["task", "db"]):
            ax.plot(g["coverage"], g["err"], marker="o", ms=3, label=f"{task}/{db}")
        ax.set_xlabel("coverage (fraction retained)"); ax.set_ylabel("selective error rate")
        ax.set_title("Rejection curves"); ax.legend(fontsize=6, ncol=2); ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{FIG}/fig3_rejection.png", dpi=160); plt.close()

    # ---------------- figure 4: scaling
    if len(grid):
        g1 = grid[grid["k"] == 1].groupby("n")["auroc"].mean().reset_index()
        gk = grid[(grid["n"] == 20000)].groupby("k")["auroc"].mean().reset_index()
        fig, axes = plt.subplots(1, 2, figsize=(10, 3.2))
        axes[0].plot(g1["n"], g1["auroc"], "o-")
        axes[0].set_xscale("log"); axes[0].set_xlabel("pretraining stays (log)")
        axes[0].set_ylabel("AUROC"); axes[0].set_title("Sample-size slice (K=1)")
        axes[1].plot(gk["k"], gk["auroc"], "s-", color="#d62728")
        axes[1].set_xlabel("number of source databases K"); axes[1].set_ylabel("AUROC")
        axes[1].set_title("Database-count slice (n=20k)")
        for ax in axes:
            ax.grid(alpha=.3)
        plt.tight_layout(); plt.savefig(f"{FIG}/fig4_scaling.png", dpi=160); plt.close()

    # ---------------- markdown summary
    lines = ["# 结果汇总\n"]
    if len(probe):
        lines.append("## 预训练编码器 + 源库线性探针（目标库零标签）\n")
        lines.append(probe[["task", "db", "n", "prev", "auroc", "slope",
                            "intercept_fixed_slope", "ici", "brier", "bss",
                            "err_full", "err_80", "rrr_80"]].round(3).to_string(index=False))
    if len(rand):
        lines.append("\n## 配对随机初始化对照\n")
        lines.append(rand[["task", "db", "auroc", "slope", "ici", "brier"]].round(3).to_string(index=False))
    if len(tab):
        lines.append("\n## 无预训练基线（表格逻辑回归）\n")
        lines.append(tab[["task", "db", "auroc", "slope", "ici", "brier", "rrr_80"]].round(3).to_string(index=False))
    if len(ft):
        lines.append("\n## 目标库微调（10% 标签）\n")
        lines.append(ft[["task", "db", "auroc", "slope", "ici", "brier"]].round(3).to_string(index=False))
    if len(grid):
        lines.append("\n## 规模律网格\n")
        lines.append(grid[["cell", "db", "n", "k", "auroc", "ici", "brier", "rrr_80"]].round(3).to_string(index=False))
    open(f"{ROOT}/results/summary.md", "w").write("\n".join(lines))
    print("\nwrote results/summary.md and figures")


if __name__ == "__main__":
    main()
