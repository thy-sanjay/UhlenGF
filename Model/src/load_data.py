from pathlib import Path
import sys
import tempfile
import numpy as np
import pandas as pd
from scipy import stats
import warnings

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

INPUT_EXP = DATA_DIR / "exp_fpkm.csv"
INPUT_DES = DATA_DIR / "expdes.csv"
OUTPUT_FILE = RESULTS_DIR / "exp_processed.csv"

RESULTS_DIR.mkdir(exist_ok=True, parents=True)

MISSINGNESS_CUTOFF = 0.20
LOG_TRANSFORM = True

if not INPUT_EXP.exists():
    print(f"ERROR: exp_fpkm.csv not found: {INPUT_EXP}")
    sys.exit(1)

df = pd.read_csv(INPUT_EXP, index_col=0)
print("Loaded exp_fpkm.csv:", df.shape)

if not INPUT_DES.exists():
    print(f"ERROR: expdes.csv not found: {INPUT_DES}")
    sys.exit(1)

des = pd.read_csv(INPUT_DES)
tissue_cols = des["column_name"].tolist()

print(f"Using {len(tissue_cols)} tissue numeric columns.")

meta_cols = [c for c in df.columns if c not in tissue_cols]

df_meta = df[meta_cols].copy()
df_num = df[tissue_cols].copy()

if "gene_name" not in meta_cols:
    print("WARNING: gene_name not found in metadata columns!")
else:
    print("Metadata columns:", meta_cols)

missing_frac = df_num.isna().mean(axis=1)
to_keep = missing_frac <= MISSINGNESS_CUTOFF

print(f"Dropping {(~to_keep).sum()} genes with >20% missing")

df_meta = df_meta.loc[to_keep]
df_num = df_num.loc[to_keep]

try:
    import ompute
except Exception as e:
    print("ERROR: Cannot import ompute. Install your package.")
    print(e)
    sys.exit(1)

print("Preparing for ompute.knn...")

with tempfile.TemporaryDirectory() as td:
    tmp_numeric = Path(td) / "tmp_numeric.csv"
    df_num.to_csv(tmp_numeric, index=True)

    d_arg = str(INPUT_DES)

    print("Calling ompute.knn:")
    print("  i =", tmp_numeric)
    print("  d =", d_arg)
    print("  o =", RESULTS_DIR)

    try:
        ompute.knn(i=str(tmp_numeric), d=d_arg, o=str(RESULTS_DIR))
    except Exception as e:
        print("ERROR: ompute.knn failed:", e)
        sys.exit(1)

OUT_OMPUTE = RESULTS_DIR / "output.csv"

if not OUT_OMPUTE.exists():
    print("ERROR: ompute output.csv not found.")
    sys.exit(1)

df_imp = pd.read_csv(OUT_OMPUTE, index_col=0)
print("Loaded imputed tissue matrix:", df_imp.shape)

if "gene_name" in df_meta.columns:
    final_meta = df_meta.loc[df_imp.index][["gene_name"]]
else:
    final_meta = pd.DataFrame(index=df_imp.index)

df_num_imp = df_imp.copy()

if LOG_TRANSFORM:
    if (df_num_imp < 0).any().any():
        shift = abs(df_num_imp.min().min()) + 1e-6
        df_num_imp = df_num_imp + shift
        warnings.warn(f"Shifted numeric matrix by {shift} before log1p.")

    df_log = np.log1p(df_num_imp)
else:
    df_log = df_num_imp.copy()

def row_z(row):
    arr = row.values
    if np.nanstd(arr) == 0:
        return np.zeros_like(arr)
    return np.nan_to_num(stats.zscore(arr, nan_policy="omit"))

df_z = df_log.apply(row_z, axis=1, result_type="broadcast")
df_z = pd.DataFrame(df_z.values, index=df_log.index, columns=df_log.columns)

final_df = pd.concat([final_meta, df_z], axis=1)

final_df.to_csv(OUTPUT_FILE)
print("\nSTEP 1 COMPLETE ✔")
print("Final file saved:", OUTPUT_FILE)
print("Final shape:", final_df.shape)