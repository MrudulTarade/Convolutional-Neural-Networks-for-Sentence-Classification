"""Quick check: re-print every report table from results/ — no training, pandas only.
Usage: python scripts/cr_tables.py  (from the repo root)"""
from pathlib import Path
import pandas as pd

RES = Path(__file__).resolve().parent.parent / "results"
pd.set_option("display.width", 160)

def ms(df):
    out = df.groupby("model")[["accuracy", "f1_macro"]].agg(["mean", "std"]) * 100
    out.columns = [f"{a}_{b}" for a, b in out.columns]
    return out.round(1)

print("== Baselines + CNN-rand reproduction (10-fold CV) ==")
base = pd.read_csv(RES / "cr_baseline_cv.csv")
cnn = pd.read_csv(RES / "cr_cnn_rand_cv.csv")
print(ms(pd.concat([base, cnn])))

print("\n== Ablation (mean ± sample SD over 10 folds, seeds averaged) ==")
abl = pd.read_csv(RES / "cr_ablation_summary.csv")
print(abl.set_index("model").iloc[:, :6].mul(100).round(1))

print("\n== Ablation paired tests ==")
print(pd.read_csv(RES / "cr_paired_tests.csv").round(3))

print("\n== Word2vec (CNN-static / non-static, 3 seeds x 10 folds) ==")
w = pd.read_csv(RES / "cr_word2vec_summary.csv").set_index("model")
print(w.iloc[:, :6].round(1))

print("\n== Word2vec paired tests ==")
print(pd.read_csv(RES / "cr_word2vec_paired.csv").round(4))

print("\n== Task 6 (fixed fold 0, seeds 0-4) ==")
t6 = pd.read_csv(RES / "task6_summary.csv")
print(t6.pivot(index="model", columns="metric", values=["mean", "sd"]).mul(100).round(1))
print("\nTask 6 paired by seed:")
print(pd.read_csv(RES / "task6_paired.csv").round(4))
