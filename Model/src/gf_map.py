from pathlib import Path
import sys
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

EXP_PROC_F = RESULTS_DIR / "exp_processed.csv"   # UPDATED
GF_GROUP_F = DATA_DIR / "gf_group.csv"

OUT_EXP_FAMILIES = RESULTS_DIR / "exp_families.csv"
OUT_FAMILY_EXPR = RESULTS_DIR / "gf_exp.csv"
OUT_FAMILY_MEMBERS = RESULTS_DIR / "family_members.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not EXP_PROC_F.exists():
    print(f"ERROR: processed expression not found: {EXP_PROC_F}")
    sys.exit(1)
exp_df = pd.read_csv(EXP_PROC_F, index_col=0)
print(f"Loaded processed expression: {exp_df.shape}")

tissue_cols = [c for c in exp_df.columns if c != "gene_name"]
if len(tissue_cols) == 0:
    print("ERROR: No tissue columns found in exp_processed.csv")
    sys.exit(1)
print(f"Detected {len(tissue_cols)} tissue columns.")

if not GF_GROUP_F.exists():
    print(f"ERROR: gf_group.csv not found: {GF_GROUP_F}")
    sys.exit(1)

gf = pd.read_csv(GF_GROUP_F, dtype=str)
print(f"Loaded gf_group.csv: {gf.shape}")

if "Ensembl gene ID" not in gf.columns or "Group name" not in gf.columns:
    print("ERROR: gf_group.csv must contain columns 'Ensembl gene ID' and 'Group name'.")
    print("Columns found:", gf.columns.tolist())
    sys.exit(1)

def normalize_ensg(x):
    if pd.isna(x):
        return None
    s = str(x).strip()
    if "." in s:
        s = s.split(".")[0]
    return s

gf["ensg_norm"] = gf["Ensembl gene ID"].apply(normalize_ensg)
gf_map = gf.dropna(subset=["ensg_norm"])[["ensg_norm", "Group name"]].drop_duplicates()
mapping = dict(zip(gf_map["ensg_norm"], gf_map["Group name"]))

print(f"Constructed mapping for {len(mapping)} Ensembl IDs -> family names.")

exp_df = exp_df.copy()
exp_df["_ensg_norm"] = [str(idx).split(".")[0] for idx in exp_df.index.astype(str)]

exp_df["family_name"] = exp_df["_ensg_norm"].map(mapping).fillna("Misc")

cols_order = []
if "gene_name" in exp_df.columns:
    cols_order = ["gene_name", "family_name"] + tissue_cols
else:
    cols_order = ["family_name"] + tissue_cols

exp_families_df = exp_df[cols_order].copy()

exp_families_df.to_csv(OUT_EXP_FAMILIES)
print(f"Saved per-gene family-assigned table: {OUT_EXP_FAMILIES} ({exp_families_df.shape})")

family_expr = exp_families_df.groupby("family_name")[tissue_cols].mean()
family_expr.to_csv(OUT_FAMILY_EXPR)
print(f"Saved family-level expression (mean) to: {OUT_FAMILY_EXPR} ({family_expr.shape})")