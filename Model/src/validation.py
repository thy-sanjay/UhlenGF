from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.metrics import silhouette_samples, silhouette_score, roc_auc_score, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.multiclass import OneVsRestClassifier
from sklearn.decomposition import PCA
import warnings
import time
import json

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

FAM_EXP_CANDIDATES = [RESULTS / "gf_exp.csv", DATA / "family_expression.csv", DATA / "gf_exp_processed.csv"]
MODULE_CAND = [RESULTS / "modules_louvain.csv", RESULTS / "modules_leiden.csv", RESULTS / "modules_fallback.csv"]
PCA_EMB = RESULTS / "family_pca_embeddings.csv"
AE_EMB = RESULTS / "family_ae_embeddings.csv"

OUT_COR = RESULTS / "validation_correlation.csv"
OUT_SIL = RESULTS / "validation_silhouette.csv"
OUT_ML = RESULTS / "ml_classification_report.csv"
OUT_SUM = RESULTS / "validation_summary.txt"

PERMUTATIONS = 500
RANDOM_STATE = 0
N_PCA_COMPONENTS = 8
CV_FOLDS = 5

def find_file(cands):
    for p in cands:
        if p.exists():
            return p
    return None

def load_family_expression():
    p = find_file(FAM_EXP_CANDIDATES)
    if p is None:
        raise FileNotFoundError("No family expression file found. Looked for: " + ",".join(str(x) for x in FAM_EXP_CANDIDATES))
    df = pd.read_csv(p, index_col=0, low_memory=False)
    numeric = df.apply(pd.to_numeric, errors='coerce')
    df_num = numeric.loc[:, numeric.dtypes != object]
    df_num = df_num.select_dtypes(include=[np.number])
    df_num.index = df.index.astype(str)
    return df_num

def load_modules():
    p = find_file(MODULE_CAND)
    if p is None:
        raise FileNotFoundError("No modules file found in results/. Run gf_graph.py first.")
    df = pd.read_csv(p, dtype=str, low_memory=False)
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

def pairwise_correlation_matrix(df):
    arr = df.values
    if np.isnan(arr).any():
        row_means = np.nanmean(arr, axis=1)
        inds = np.where(np.isnan(arr))
        arr[inds] = np.take(row_means, inds[0])
    corr = np.corrcoef(arr)
    return corr

def compute_within_between(corr_mat, families, module_groups, family_index_map, permutations=500, random_state=0):
    rng = np.random.default_rng(random_state)
    results = []
    n_fam = len(families)
    for m, members in module_groups.items():
        idxs = [family_index_map[f] for f in members if f in family_index_map]
        if len(idxs) < 2:
            within_mean = np.nan
            between_mean = np.nan
            pval = np.nan
            results.append({"module": m, "size": len(members), "within_mean": within_mean, "between_mean": between_mean, "perm_p": pval})
            continue
        sub = corr_mat[np.ix_(idxs, idxs)]
        n = len(idxs)
        if n * (n - 1) / 2 > 0:
            upper_inds = np.triu_indices(n, k=1)
            within_vals = sub[upper_inds]
            within_mean = float(np.nanmean(within_vals))
        else:
            within_mean = float(np.nanmean(sub))
        non_idxs = [i for i in range(n_fam) if i not in idxs]
        if len(non_idxs) == 0:
            between_mean = np.nan
        else:
            between_vals = corr_mat[np.ix_(idxs, non_idxs)].ravel()
            between_mean = float(np.nanmean(between_vals))
        perm_counts = 0
        perm_vals = []
        for _ in range(permutations):
            perm_idxs = rng.choice(n_fam, size=len(idxs), replace=False)
            subp = corr_mat[np.ix_(perm_idxs, perm_idxs)]
            if len(perm_idxs) >= 2:
                pv = float(np.nanmean(subp[np.triu_indices(len(perm_idxs), k=1)]))
            else:
                pv = float(np.nanmean(subp))
            perm_vals.append(pv)
            if pv >= within_mean:
                perm_counts += 1
        pval = (perm_counts + 1) / (permutations + 1)
        results.append({"module": m, "size": len(members), "within_mean": within_mean, "between_mean": between_mean, "perm_p": pval})
    return pd.DataFrame(results)

def get_embeddings(df_exp):
    if PCA_EMB.exists():
        try:
            emb = pd.read_csv(PCA_EMB, index_col=0)
            emb.index = emb.index.astype(str)
            return emb
        except Exception:
            pass
    if AE_EMB.exists():
        try:
            emb = pd.read_csv(AE_EMB, index_col=0)
            emb.index = emb.index.astype(str)
            return emb
        except Exception:
            pass
    pca = PCA(n_components=min(N_PCA_COMPONENTS, min(df_exp.shape)-1), random_state=RANDOM_STATE)
    X = df_exp.values
    if np.isnan(X).any():
        row_means = np.nanmean(X, axis=1)
        inds = np.where(np.isnan(X))
        X[inds] = np.take(row_means, inds[0])
    comp = pca.fit_transform(X)
    emb = pd.DataFrame(comp, index=df_exp.index, columns=[f"PC{i+1}" for i in range(comp.shape[1])])
    return emb

def compute_silhouettes(emb, modules_df):
    df = modules_df.copy()
    df = df[df["family_name"].isin(emb.index)]
    if df.shape[0] < 2:
        return None, None
    X = emb.loc[df["family_name"]].values
    labels = df["module"].values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    try:
        global_sil = float(silhouette_score(Xs, labels))
        sample_sil = silhouette_samples(Xs, labels)
    except Exception:
        global_sil = np.nan
        sample_sil = np.array([np.nan] * len(labels))
    per_module = []
    df["sil"] = sample_sil
    for m, group in df.groupby("module"):
        per_module.append({"module": int(m), "size": len(group), "mean_silhouette": float(np.nanmean(group["sil"]))})
    per_module_df = pd.DataFrame(per_module)
    return global_sil, per_module_df

def ml_classification_eval(emb, modules_df):
    df = modules_df.copy()
    df = df[df["family_name"].isin(emb.index)].reset_index(drop=True)
    if df.shape[0] < 10:
        return None
    X = emb.loc[df["family_name"]].values
    y = df["module"].values
    le = LabelEncoder()
    y_enc = le.fit_transform(y)
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    skf = StratifiedKFold(n_splits=min(CV_FOLDS, len(np.unique(y_enc))), shuffle=True, random_state=RANDOM_STATE)
    fold_metrics = []
    base = LogisticRegression(multi_class="multinomial", solver="lbfgs", max_iter=2000, random_state=RANDOM_STATE)
    ovr = OneVsRestClassifier(base)
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(Xs, y_enc), start=1):
        X_tr, X_te = Xs[train_idx], Xs[test_idx]
        y_tr, y_te = y_enc[train_idx], y_enc[test_idx]
        ovr.fit(X_tr, y_tr)
        y_pred = ovr.predict(X_te)
        try:
            y_score = ovr.predict_proba(X_te)
        except Exception:
            try:
                y_score = ovr.decision_function(X_te)
            except Exception:
                y_score = None
        acc = accuracy_score(y_te, y_pred)
        mf1 = f1_score(y_te, y_pred, average="macro")
        auroc = np.nan
        if y_score is not None:
            try:
                from sklearn.preprocessing import label_binarize
                y_te_bin = label_binarize(y_te, classes=np.arange(len(le.classes_)))
                aurocs = []
                for c in range(y_te_bin.shape[1]):
                    if y_te_bin[:, c].sum() <= 1:
                        continue
                    try:
                        aurocs.append(roc_auc_score(y_te_bin[:, c], y_score[:, c]))
                    except Exception:
                        pass
                if aurocs:
                    auroc = float(np.mean(aurocs))
            except Exception:
                auroc = np.nan
        fold_metrics.append({"fold": fold_idx, "accuracy": float(acc), "macro_f1": float(mf1), "macro_auroc": float(auroc)})
    df_folds = pd.DataFrame(fold_metrics)
    summary = {"accuracy_mean": float(df_folds["accuracy"].mean()), "macro_f1_mean": float(df_folds["macro_f1"].mean()), "macro_auroc_mean": float(df_folds["macro_auroc"].mean())}
    return df_folds, summary, le

def main():
    t0 = time.time()
    df_exp = load_family_expression()
    modules_df = load_modules()
    corr_mat = pairwise_correlation_matrix(df_exp)
    families = df_exp.index.astype(str).tolist()
    family_index_map = {fam: i for i, fam in enumerate(families)}
    module_groups = modules_df.groupby("module")["family_name"].apply(list).to_dict()
    corr_df = compute_within_between(corr_mat, families, module_groups, family_index_map, permutations=PERMUTATIONS, random_state=RANDOM_STATE)
    corr_df.to_csv(OUT_COR, index=False)
    print("Saved:", OUT_COR)
    emb = get_embeddings(df_exp)
    global_sil, per_module_sil = compute_silhouettes(emb, modules_df)
    if per_module_sil is None:
        print("Silhouette could not be computed; too few samples or error.")
        sil_df = pd.DataFrame([{"module": None, "size": None, "mean_silhouette": None}])
    else:
        sil_df = per_module_sil
        sil_df["global_silhouette"] = float(global_sil) if not pd.isna(global_sil) else np.nan
    sil_df.to_csv(OUT_SIL, index=False)
    print("Saved:", OUT_SIL)
    print("Running classification (embedding -> module) with cross-validation ...")
    ml_res = ml_classification_eval(emb, modules_df)
    if ml_res is None:
        print("Not enough samples for ML classification or embedding missing. Skipping ML classification.")
        pd.DataFrame([{"note": "skipped"}]).to_csv(OUT_ML, index=False)
        ml_summary = {"accuracy_mean": np.nan, "macro_f1_mean": np.nan, "macro_auroc_mean": np.nan}
    else:
        df_folds, summary, label_encoder = ml_res
        df_folds.to_csv(OUT_ML, index=False)
        ml_summary = summary
        print("Saved ML classification folds to:", OUT_ML)
    with open(OUT_SUM, "w", encoding="utf-8") as fh:
        fh.write("Validation summary\n\n")
        fh.write(f"Timestamp: {time.asctime()}\n")
        fh.write(f"Family expression used: {find_file(FAM_EXP_CANDIDATES)}\n")
        fh.write(f"Modules file used: {find_file(MODULE_CAND)}\n\n")
        fh.write("Correlation coherence\n")
        fh.write(f"  Number of modules tested: {corr_df.shape[0]}\n")
        fh.write(f"  Permutations used for p-value: {PERMUTATIONS}\n\n")
        fh.write("ML classification summary (cross-validated)\n")
        fh.write(f"  Accuracy mean: {ml_summary.get('accuracy_mean', np.nan):.4f}\n")
        fh.write(f"  Macro F1 mean: {ml_summary.get('macro_f1_mean', np.nan):.4f}\n")
        fh.write(f"  Macro AUROC mean: {ml_summary.get('macro_auroc_mean', np.nan):.4f}\n\n")
        fh.write("Files produced:\n")
        fh.write(f" - {OUT_COR}\n - {OUT_SIL}\n - {OUT_ML}\n\n")
        fh.write("Interpretation tips:\n")
        fh.write(" - Within > between correlation with small p-value suggests coherent modules.\n")
        fh.write(" - High silhouette indicates embeddings separate modules well.\n")
        fh.write(" - Good ML accuracy / AUROC suggests modules are separable by their expression patterns.\n")
    print("Saved validation summary to:", OUT_SUM)
    print("Total time (s):", time.time() - t0)

if __name__ == "__main__":
    main()