from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import fisher_exact
import warnings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

LOCALIZATION_F = DATA / "localization.csv"
FAMILY_MEMBERS_F = DATA / "family_members.csv"
TF_F = DATA / "tfgenes.csv"
DRUG_F = DATA / "drugtargets.csv"
ONCO_F = DATA / "oncogenes.csv"

MODULE_PRIORITY = [
    RESULTS / "modules_louvain.csv",
    RESULTS / "modules_leiden.csv",
    RESULTS / "modules_fallback.csv"
]

OUT_ANNOT = RESULTS / "annotated_modules.csv"
OUT_DETAILS = RESULTS / "annotated_module_details.csv"
OUT_SUM = RESULTS / "module_annotation_summary.txt"

def find_module_file():
    for p in MODULE_PRIORITY:
        if p.exists():
            return p
    return None

def read_modules(path):
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if df.shape[1] == 1:
        df = df.reset_index()
    df = df.iloc[:, :2].copy()
    df.columns = ["family_name", "module"]
    df["family_name"] = df["family_name"].astype(str)
    try:
        df["module"] = df["module"].astype(int)
    except Exception:
        df["module"] = pd.Categorical(df["module"]).codes
    return df

def read_family_members(path):
    if not path.exists():
        raise FileNotFoundError(f"family_members.csv not found at {path}")
    fm = pd.read_csv(path, dtype=str, low_memory=False)
    cols = [c.strip() for c in fm.columns]
    fm.columns = cols
    family_to_genes = {}
    if fm.shape[1] >= 2:
        second = fm.columns[1]
        try:
            has_commas = fm[second].dropna().astype(str).str.contains(",").sum() > 0
        except Exception:
            has_commas = False
        if has_commas:
            fam_col, mem_col = fm.columns[0], fm.columns[1]
            for _, r in fm.iterrows():
                fam = str(r[fam_col]).strip()
                mems = r[mem_col]
                if pd.isna(mems) or str(mems).strip() == "":
                    continue
                genes = [g.strip() for g in str(mems).split(",") if g.strip()]
                if genes:
                    family_to_genes.setdefault(fam, []).extend(genes)
            return family_to_genes
        if fm.shape[1] == 2:
            fam_col, gene_col = fm.columns[0], fm.columns[1]
            for _, r in fm.iterrows():
                fam = str(r[fam_col]).strip()
                gene = r[gene_col]
                if pd.isna(gene):
                    continue
                gene = str(gene).strip()
                if fam == "" or gene == "" or gene.lower() == "nan":
                    continue
                family_to_genes.setdefault(fam, []).append(gene)
            return family_to_genes
        for _, r in fm.iterrows():
            fam = str(r[fm.columns[0]]).strip()
            genes = []
            for c in fm.columns[1:]:
                v = r[c]
                if pd.isna(v):
                    continue
                vs = [x.strip() for x in str(v).split(",") if x.strip()]
                genes.extend(vs)
            if genes:
                family_to_genes.setdefault(fam, []).extend(genes)
        return family_to_genes
    else:
        col = fm.columns[0]
        for _, r in fm.iterrows():
            v = r[col]
            if pd.isna(v):
                continue
            parts = [p.strip() for p in str(v).replace("\t", ",").split(",") if p.strip()]
            if len(parts) >= 2:
                fam = parts[0]
                genes = parts[1:]
                family_to_genes.setdefault(fam, []).extend(genes)
        return family_to_genes

def load_onecol_set(path):
    if not path.exists():
        return set()
    df = pd.read_csv(path, dtype=str, low_memory=False)
    if df.shape[1] == 0:
        return set()
    col = df.columns[0]
    vals = df[col].dropna().astype(str).str.strip()
    vals = [v for v in vals if v != "" and v.lower() != "nan"]
    return set(vals)

def bh_fdr(pvals):
    p = np.array(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return np.array([])
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / (np.arange(1, n+1))
    q = np.minimum.accumulate(q[::-1])[::-1]
    q = np.minimum(q, 1.0)
    out = np.empty(n)
    out[order] = q
    return out

module_file = find_module_file()
if module_file is None:
    raise FileNotFoundError("No module file found in results/. Produce modules_louvain.csv or modules_leiden.csv or modules_fallback.csv first.")

modules_df = read_modules(module_file)
family_to_genes = read_family_members(FAMILY_MEMBERS_F)

gene_to_families = {}
for fam, genes in family_to_genes.items():
    for g in genes:
        gene_to_families.setdefault(g, []).append(fam)

if not LOCALIZATION_F.exists():
    raise FileNotFoundError(f"localization.csv not found at {LOCALIZATION_F}")
df_loc = pd.read_csv(LOCALIZATION_F, dtype=str, low_memory=False)
cols_low = {c.lower(): c for c in df_loc.columns}

ens_col = None
for key in ("ensg_id", "ensembl_id", "ensg", "gene_id"):
    if key in cols_low:
        ens_col = cols_low[key]
        break
if ens_col is None:
    ens_col = df_loc.columns[0]

sp_col = cols_low.get("sp", None)
tm_col = cols_low.get("tm", None)
if sp_col is None:
    for c in df_loc.columns:
        if "secret" in c.lower() or c.lower() == "sp":
            sp_col = c
            break
if tm_col is None:
    for c in df_loc.columns:
        if "transmem" in c.lower() or c.lower() == "tm":
            tm_col = c
            break

df_loc[ens_col] = df_loc[ens_col].astype(str).str.strip()

sp_genes = set()
tm_genes = set()
if sp_col and sp_col in df_loc.columns:
    sp_flags = df_loc[sp_col].astype(str).str.strip()
    sp_genes = set(df_loc.loc[sp_flags.isin(["1","True","true","Y","y","yes","Yes"]), ens_col].tolist())
if tm_col and tm_col in df_loc.columns:
    tm_flags = df_loc[tm_col].astype(str).str.strip()
    tm_genes = set(df_loc.loc[tm_flags.isin(["1","True","true","Y","y","yes","Yes"]), ens_col].tolist())

sp_families = set()
tm_families = set()
for fam, genes in family_to_genes.items():
    if any(g in sp_genes for g in genes):
        sp_families.add(fam)
    if any(g in tm_genes for g in genes):
        tm_families.add(fam)

tf_list = load_onecol_set(TF_F)
drug_list = load_onecol_set(DRUG_F)
onco_list = load_onecol_set(ONCO_F)

def map_list_to_families(lst):
    fams = set()
    for item in lst:
        if item in gene_to_families:
            fams.update(gene_to_families[item])
        elif item in family_to_genes:
            fams.add(item)
    return fams

tf_fams = map_list_to_families(tf_list)
drug_fams = map_list_to_families(drug_list)
onco_fams = map_list_to_families(onco_list)

module_groups = modules_df.groupby("module")["family_name"].apply(list).to_dict()
modules = sorted(module_groups.keys())

bg_families = set(family_to_genes.keys()).union(set(modules_df["family_name"].tolist()))
N = len(bg_families)

records = []
pvals_store = {"sp": [], "tm": [], "tf": [], "drug": [], "onco": []}

for m in modules:
    members = set(module_groups[m])
    M = len(members)
    rec = {"module": int(m), "size": M}

    def fisher_for(annot_set):
        a = len(members & annot_set)
        b = M - a
        c = len(annot_set - members)
        d = N - a - b - c
        if any(x < 0 for x in (a, b, c, d)):
            a = len((members & annot_set) & bg_families)
            b = len(members & bg_families) - a
            c = len((annot_set & bg_families) - members)
            d = N - a - b - c
        try:
            _, p = fisher_exact([[a, b], [c, d]], alternative="greater")
        except Exception:
            p = 1.0
        return a, float(p)

    a_sp, p_sp = fisher_for(sp_families)
    a_tm, p_tm = fisher_for(tm_families)
    a_tf, p_tf = fisher_for(tf_fams)
    a_drug, p_drug = fisher_for(drug_fams)
    a_onco, p_onco = fisher_for(onco_fams)

    rec.update({
        "sp_hits": int(a_sp), "sp_p": p_sp,
        "tm_hits": int(a_tm), "tm_p": p_tm,
        "tf_hits": int(a_tf), "tf_p": p_tf,
        "drug_hits": int(a_drug), "drug_p": p_drug,
        "onco_hits": int(a_onco), "onco_p": p_onco
    })

    pvals_store["sp"].append(p_sp)
    pvals_store["tm"].append(p_tm)
    pvals_store["tf"].append(p_tf)
    pvals_store["drug"].append(p_drug)
    pvals_store["onco"].append(p_onco)

    records.append(rec)

annot_df = pd.DataFrame(records).sort_values("module")

for ann in pvals_store.keys():
    pvals = pvals_store[ann]
    if len(pvals) == 0:
        annot_df[f"{ann}_fdr"] = np.nan
    else:
        annot_df[f"{ann}_fdr"] = bh_fdr(pvals)

annot_df.to_csv(OUT_ANNOT, index=False)

details = modules_df.copy()
details["is_sp"] = details["family_name"].apply(lambda x: 1 if x in sp_families else 0)
details["is_tm"] = details["family_name"].apply(lambda x: 1 if x in tm_families else 0)
details["is_tf"] = details["family_name"].apply(lambda x: 1 if x in tf_fams else 0)
details["is_drug"] = details["family_name"].apply(lambda x: 1 if x in drug_fams else 0)
details["is_onco"] = details["family_name"].apply(lambda x: 1 if x in onco_fams else 0)
details.to_csv(OUT_DETAILS, index=False)

with open(OUT_SUM, "w", encoding="utf-8") as fh:
    fh.write("Module annotation summary\n\n")
    fh.write(f"Module file used: {module_file.name}\n")
    fh.write(f"Families from family_members: {len(family_to_genes)}\n")
    fh.write(f"Background families (family_members union modules): {N}\n")
    fh.write(f"Localization -> families: SP={len(sp_families)}, TM={len(tm_families)}\n\n")
    fh.write("Modules with significant enrichments (FDR < 0.05):\n\n")
    found = False
    for _, row in annot_df.iterrows():
        m = int(row["module"])
        sigs = []
        for ann in ("sp", "tm", "tf", "drug", "onco"):
            fdr = row.get(f"{ann}_fdr", np.nan)
            if (not pd.isna(fdr)) and (fdr < 0.05):
                sigs.append(f"{ann}(hits={int(row[f'{ann}_hits'])}, fdr={fdr:.3g})")
        if sigs:
            fh.write(f"Module {m} (size={int(row['size'])}): " + "; ".join(sigs) + "\n")
            found = True
    if not found:
        fh.write("No module passed FDR < 0.05 for the tested annotations.\n")
    fh.write("\nSaved detailed files:\n")
    fh.write(f" - {OUT_ANNOT}\n - {OUT_DETAILS}\n")

print("gf_annotate.py complete.")