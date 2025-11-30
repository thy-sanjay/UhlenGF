# src/panel_discovery.py
"""
Panel discovery script (fixed greedy selection: always build up to MAX_PANEL_SIZE)

Usage:
    python src/panel_discovery.py

Outputs -> results/panels/
 - top_panels.csv
 - panel_classification_metrics.csv
 - panel_annotations.csv
 - heatmaps and network highlight PNGs
"""
from pathlib import Path
import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings, re

# ---------- CONFIG ----------
ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
OUT = RESULTS / "panels"
OUT.mkdir(parents=True, exist_ok=True)

FAM_EXP_FILES = [RESULTS / "gf_exp.csv", DATA / "family_expression.csv"]
EMB_FILES = [RESULTS / "family_ae_embeddings.csv", RESULTS / "family_pca_embeddings.csv"]
MODULE_FILES = [RESULTS / "modules_louvain.csv", RESULTS / "modules_fallback.csv", RESULTS / "modules_leiden.csv"]
GRAPH_GML = RESULTS / "family_graph.gml"
GRAPH_EDGES = RESULTS / "family_graph_edges.csv"
COMPARISONS = DATA / "comparisions.csv"

LOCALIZATION = DATA / "localization.csv"
TFGENES = DATA / "tfgenes.csv"
DRUGTARGET = DATA / "drugtarget.csv"
ONCOGENES = DATA / "oncogenes.csv"
FAM_MEMBERS = DATA / "family_members.csv"

TOP_K_CENTRAL = 8
TOP_K_REPR = 8
MAX_PANEL_SIZE = 100            # maximum panel size (single integer)
CV_FOLDS = 5
N_PERMUTATIONS = 200
RANDOM_STATE = 0

# ---------- helpers ----------
def find_first(paths):
    for p in paths:
        if p.exists(): return p
    return None

def read_family_expression():
    p = find_first(FAM_EXP_FILES)
    if p is None:
        raise FileNotFoundError("No family expression file found (gf_exp.csv or family_expression.csv).")
    df = pd.read_csv(p, index_col=0)
    df.index = df.index.astype(str)
    df.columns = [str(c).strip() for c in df.columns]
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def read_modules():
    p = find_first(MODULE_FILES)
    if p is None:
        raise FileNotFoundError("Module assignments not found in results/")
    m = pd.read_csv(p, dtype=str)
    if m.shape[1] == 1:
        m = m.reset_index()
    m = m.iloc[:, :2]
    m.columns = ["family_name","module"]
    m["family_name"] = m["family_name"].astype(str)
    try:
        m["module"] = m["module"].astype(int)
    except:
        m["module"] = pd.Categorical(m["module"]).codes
    return m

def read_graph():
    if GRAPH_GML.exists():
        try:
            G = nx.read_gml(str(GRAPH_GML))
            return nx.relabel_nodes(G, lambda x: str(x))
        except Exception:
            warnings.warn("Failed to read GML; falling back to edges CSV.")
    if GRAPH_EDGES.exists():
        ed = pd.read_csv(GRAPH_EDGES)
        src,tgt = ed.columns[:2]
        G = nx.Graph()
        for _,r in ed.iterrows():
            u,v = str(r[src]), str(r[tgt])
            w = float(r["weight"]) if "weight" in ed.columns else 1.0
            G.add_edge(u,v, weight=w)
        return G
    return None

def read_embeddings(families):
    p = find_first(EMB_FILES)
    if p is None:
        return None
    emb = pd.read_csv(p, index_col=0)
    emb.index = emb.index.astype(str)
    emb = emb.loc[emb.index.intersection(families)]
    for c in emb.columns:
        emb[c] = pd.to_numeric(emb[c], errors="coerce")
    return emb

def parse_aggregate_groups(comparisons_csv, available_tissues):
    if not comparisons_csv.exists():
        raise FileNotFoundError("comparisions.csv not found in data/")
    df = pd.read_csv(comparisons_csv, dtype=str, header=0, low_memory=False)
    if df.shape[1] < 2: raise ValueError("comparisions.csv must have at least two columns")
    col1 = df.iloc[:,0].dropna().astype(str).tolist()
    col2 = df.iloc[:,1].dropna().astype(str).tolist()
    def explode(lst):
        out=[]
        for s in lst:
            parts = [p.strip() for p in re.split(r"[;,]", s) if p.strip()]
            out.extend(parts)
        return sorted(set(out))
    g1 = explode(col1); g2 = explode(col2)
    def map_names(lst):
        mapped=[]
        for name in lst:
            nl = name.strip().lower()
            exact = [t for t in available_tissues if t.lower()==nl]
            if exact:
                mapped.append(exact[0]); continue
            subs = [t for t in available_tissues if nl in t.lower()]
            if subs:
                mapped.append(subs[0]); continue
            raise ValueError(f"Tissue '{name}' not found among expression columns.")
        return sorted(set(mapped))
    return map_names(g1), map_names(g2)

def cohen_d(a,b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if len(a)<2 and len(b)<2:
        return float(np.nanmean(a)-np.nanmean(b)) if (len(a)>0 or len(b)>0) else 0.0
    ma,mb = np.nanmean(a), np.nanmean(b)
    sa = np.nanstd(a, ddof=1) if len(a)>1 else 0.0
    sb = np.nanstd(b, ddof=1) if len(b)>1 else 0.0
    denom = np.sqrt(((len(a)-1)*(sa**2) + (len(b)-1)*(sb**2)) / max(1, (len(a)+len(b)-2)))
    if denom==0:
        return float(ma-mb)
    return float((ma-mb)/denom)

# ---------- candidate selection ----------
def select_candidates_per_module(df_fam, modules_df, G=None, emb=None, top_k_central=TOP_K_CENTRAL, top_k_repr=TOP_K_REPR):
    families = df_fam.index.tolist()
    modules = modules_df.groupby("module")["family_name"].apply(list).to_dict()
    # centrality
    central = {}
    if G is not None:
        deg = dict(G.degree())
        try:
            bet = nx.betweenness_centrality(G, normalized=True, weight="weight")
        except Exception:
            bet = nx.betweenness_centrality(G, normalized=True)
        for f in families:
            central[f] = {"deg": float(deg.get(f,0)), "bet": float(bet.get(f,0))}
    else:
        corr = df_fam.T.corr().fillna(0)
        for f in families:
            central[f] = {"deg": float((corr.loc[f].abs() > 0.5).sum() - 1), "bet": 0.0}
    central_df = pd.DataFrame.from_dict(central, orient="index").fillna(0)
    # representativeness (embedding)
    if emb is None:
        X = df_fam.fillna(df_fam.mean(axis=1), axis=0)
        ncomp = min(8, max(2, min(X.shape)-1))
        pca = PCA(n_components=ncomp, random_state=RANDOM_STATE)
        emb_arr = pca.fit_transform(X.values)
        emb_df = pd.DataFrame(emb_arr, index=df_fam.index)
    else:
        emb_df = emb.copy()
        miss = [f for f in df_fam.index if f not in emb_df.index]
        if miss:
            add = pd.DataFrame(index=miss, columns=emb_df.columns, dtype=float)
            emb_df = pd.concat([emb_df, add])
        emb_df = emb_df.loc[df_fam.index]
    repr_scores = {}
    for mod, fams in modules.items():
        fams_present = [f for f in fams if f in emb_df.index]
        if not fams_present: continue
        centroid = emb_df.loc[fams_present].mean(axis=0).values
        for f in fams_present:
            dist = float(np.linalg.norm(emb_df.loc[f].values.astype(float) - centroid))
            repr_scores[f] = dist
    repr_df = pd.DataFrame(index=df_fam.index)
    repr_df["dist"] = [repr_scores.get(f, np.nan) for f in df_fam.index]
    repr_df["repr_rank"] = repr_df["dist"].rank(ascending=True)
    candidates = {}
    detail_rows = []
    for mod, fams in modules.items():
        fams_present = [f for f in fams if f in df_fam.index]
        if not fams_present:
            candidates[mod] = []
            continue
        cent_sub = central_df.loc[fams_present].copy()
        cent_sub["cent_score"] = cent_sub["deg"] + cent_sub["bet"]
        top_cent = cent_sub.sort_values("cent_score", ascending=False).head(top_k_central).index.tolist()
        repr_sub = repr_df.loc[fams_present].copy().sort_values("repr_rank", ascending=True)
        top_repr = repr_sub.head(top_k_repr).index.tolist()
        inter = list(set(top_cent).intersection(top_repr))
        cand = inter if inter else list(dict.fromkeys(top_cent + top_repr))
        candidates[mod] = cand
        for f in fams_present:
            detail_rows.append({
                "module": mod, "family": f,
                "deg": float(central_df.at[f,"deg"]) if f in central_df.index else 0.0,
                "bet": float(central_df.at[f,"bet"]) if f in central_df.index else 0.0,
                "repr_dist": float(repr_df.at[f,"dist"]) if f in repr_df.index else np.nan,
                "is_candidate": int(f in cand)
            })
    details = pd.DataFrame(detail_rows)
    return candidates, details

# ---------- greedy selection (fixed: always fills up to max_size if possible) ----------
def greedy_panel_and_eval(df_fam, labels_series, candidate_list, max_size):
    """
    Greedy selection that always accumulates up to max_size:
      - compute score for each remaining candidate each iteration
      - choose the candidate with the highest score (CV mean AUC if possible else |Cohen d|)
      - continue until max_size or no remaining candidates
    """
    tissues = labels_series.index.tolist()
    candidate_list_filtered = [f for f in candidate_list if f in df_fam.index]
    if len(candidate_list_filtered) == 0:
        return {"panel": [], "auc": None, "accuracy": None, "f1_macro": None, "fpr_tpr": (None,None)}
    # build sample x features (samples = tissues)
    X_full = df_fam.loc[candidate_list_filtered, tissues].T
    imputer = SimpleImputer(strategy="median")
    X_full = pd.DataFrame(imputer.fit_transform(X_full), index=X_full.index, columns=X_full.columns)
    y = labels_series.loc[X_full.index].astype(int).values
    class_counts = np.bincount(y)
    can_cv = len(class_counts) > 1 and all(class_counts >= 2)
    selected = []
    remaining = list(X_full.columns)
    # main greedy loop
    while len(selected) < max_size and remaining:
        # compute a score for each remaining candidate
        scores = {}
        for fam in remaining:
            feats = selected + [fam]
            Xf = X_full[feats].values
            scaler = StandardScaler()
            try:
                Xs = scaler.fit_transform(Xf)
            except Exception:
                Xs = np.nan_to_num(Xf)
            if can_cv:
                n_splits = min(CV_FOLDS, max(2, min(class_counts)))
                skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
                aucs=[]
                for tr,te in skf.split(Xs, y):
                    clf = LogisticRegression(max_iter=2000)
                    clf.fit(Xs[tr], y[tr])
                    probs = clf.predict_proba(Xs[te])[:,1]
                    try:
                        aucs.append(roc_auc_score(y[te], probs))
                    except Exception:
                        # if ROC AUC fails for tiny test folds, fallback to accuracy on folds
                        preds = clf.predict(Xs[te])
                        aucs.append(accuracy_score(y[te], preds))
                scores[fam] = float(np.mean(aucs))
            else:
                fam_vals = X_full[[fam]]
                grp1 = fam_vals.loc[labels_series[labels_series==1].index].values.flatten()
                grp0 = fam_vals.loc[labels_series[labels_series==0].index].values.flatten()
                scores[fam] = abs(cohen_d(grp1, grp0))
        # pick the candidate with maximum score (even if equal or less than previous best)
        best = max(scores.items(), key=lambda x: (x[1], x[0]))[0]
        selected.append(best)
        remaining.remove(best)
    # Evaluate final selected panel
    if len(selected) == 0:
        return {"panel": [], "auc": None, "accuracy": None, "f1_macro": None, "fpr_tpr": (None,None)}
    Xsel = X_full[selected].values
    scaler = StandardScaler(); Xs = scaler.fit_transform(Xsel)
    class_counts = np.bincount(y)
    can_cv = len(class_counts) > 1 and all(class_counts >= 2)
    if can_cv:
        n_splits = min(CV_FOLDS, max(2, min(class_counts)))
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        aucs=[]; accs=[]; f1s=[]
        for tr,te in skf.split(Xs, y):
            clf = LogisticRegression(max_iter=2000)
            clf.fit(Xs[tr], y[tr])
            probs = clf.predict_proba(Xs[te])[:,1]; preds = clf.predict(Xs[te])
            try:
                aucs.append(roc_auc_score(y[te], probs))
            except Exception:
                aucs.append(accuracy_score(y[te], preds))
            accs.append(accuracy_score(y[te], preds))
            f1s.append(f1_score(y[te], preds, average="macro"))
        return {"panel": selected, "auc": float(np.mean(aucs)), "accuracy": float(np.mean(accs)), "f1_macro": float(np.mean(f1s)), "fpr_tpr": (None,None)}
    else:
        effs=[]
        for fam in selected:
            vals = df_fam.loc[fam, labels_series.index]
            grp1 = vals[labels_series==1]; grp0 = vals[labels_series==0]
            effs.append(cohen_d(grp1.values, grp0.values))
        return {"panel": selected, "auc": float(np.mean([abs(x) for x in effs])), "accuracy": None, "f1_macro": None, "fpr_tpr": (None,None)}

def metric_mean_abs_effect(labels_series, panel_families, df_fam_local):
    effs=[]
    for fam in panel_families:
        vals = df_fam_local.loc[fam, labels_series.index]
        grp1 = vals[labels_series==1].values; grp0 = vals[labels_series==0].values
        effs.append(abs(cohen_d(grp1, grp0)))
    return float(np.mean(effs)) if effs else 0.0

def permutation_pvalue(panel_fams, df_fam_local, labels_series, n_perm=N_PERMUTATIONS):
    baseline = metric_mean_abs_effect(labels_series, panel_fams, df_fam_local)
    count=0
    for i in range(n_perm):
        perm = labels_series.sample(frac=1.0, replace=False, random_state=i).reset_index(drop=True)
        perm.index = labels_series.index
        val = metric_mean_abs_effect(perm, panel_fams, df_fam_local)
        if val >= baseline: count += 1
    pval = (count + 1) / (n_perm + 1)
    return baseline, pval

# ---------- main ----------
if __name__ == "__main__":
    print("Starting panel discovery (fixed greedy selection)...")
    df_fam = read_family_expression()
    print("Family expression loaded:", df_fam.shape)
    tissues = df_fam.columns.tolist()

    g1, g2 = parse_aggregate_groups(COMPARISONS, tissues)
    print("Group1 tissues:", g1)
    print("Group2 tissues:", g2)

    tissues_union = sorted(set(g1 + g2))
    labels = pd.Series({t:(1 if t in g1 else 0) for t in tissues_union})
    print("Using tissues:", tissues_union, "label counts:", labels.value_counts().to_dict())

    modules_df = read_modules()
    print("Loaded modules. Count:", modules_df["module"].nunique())

    G = read_graph()
    if G is None:
        print("Graph not available; centrality estimated from correlation.")

    emb = read_embeddings(df_fam.index.tolist())
    if emb is None:
        print("Embeddings not found; PCA fallback will be used.")

    candidates, details = select_candidates_per_module(df_fam, modules_df, G=G, emb=emb)
    details.to_csv(OUT / "module_candidate_details.csv", index=False)

    panel_records=[]; metrics_records=[]
    for mod, cand in candidates.items():
        if not cand: continue
        cand_filtered = [f for f in cand if f in df_fam.index]
        if len(cand_filtered)==0: continue
        result = greedy_panel_and_eval(df_fam, labels, cand_filtered, MAX_PANEL_SIZE)
        panel = result.get("panel", [])
        if not panel: continue
        auc = result.get("auc")
        baseline, pval = permutation_pvalue(panel, df_fam, labels, n_perm=min(200, N_PERMUTATIONS))
        panel_records.append({"module":mod,"panel_size":len(panel),"families":";".join(panel),"auc":auc,"perm_metric":baseline,"perm_p":pval})
        metrics_records.append({"module":mod,"panel_size":len(panel),"auc":auc})
        # Heatmap: better sizing and saving
        sub = df_fam.loc[panel, tissues_union]
        subz = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0,1), axis=0)
        v = np.nanmax(np.abs(subz.values)) if subz.size>0 else 1.0
        fig_w = max(6, len(tissues_union)*0.5)
        fig_h = max(3, len(panel)*0.35)
        plt.figure(figsize=(fig_w, fig_h))
        ax = sns.heatmap(subz, cmap="vlag", center=0, vmin=-v, vmax=v, cbar_kws={"shrink":0.6})
        ax.set_ylabel("family_name", fontsize=10)
        ax.set_xlabel("")
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.title(f"Module {mod} combined panel (zscore)")
        plt.tight_layout()
        plt.savefig(OUT / f"heat_module{mod}_panel.png", dpi=200, bbox_inches='tight')
        plt.close()
        if G is not None:
            H = G.copy()
            node_color = ["red" if n in panel else "lightgray" for n in H.nodes()]
            plt.figure(figsize=(8,8))
            pos = nx.spring_layout(H, seed=RANDOM_STATE)
            nx.draw_networkx_nodes(H, pos, node_size=20, node_color=node_color)
            nx.draw_networkx_edges(H, pos, alpha=0.08)
            nx.draw_networkx_labels(H, pos, labels={n:n for n in panel}, font_size=6)
            plt.title(f"Graph highlight: module {mod}")
            plt.axis("off"); plt.tight_layout()
            plt.savefig(OUT / f"network_module{mod}_panel.png", dpi=200, bbox_inches='tight')
            plt.close()

    # Combined candidates across modules
    all_candidates = sorted({f for c in candidates.values() for f in c if f in df_fam.index})
    result = greedy_panel_and_eval(df_fam, labels, all_candidates, MAX_PANEL_SIZE)
    panel = result.get("panel", [])
    if panel:
        auc = result.get("auc")
        baseline, pval = permutation_pvalue(panel, df_fam, labels, n_perm=min(200, N_PERMUTATIONS))
        panel_records.append({"module":"combined","panel_size":len(panel),"families":";".join(panel),"auc":auc,"perm_metric":baseline,"perm_p":pval})
        metrics_records.append({"module":"combined","panel_size":len(panel),"auc":auc})
        sub = df_fam.loc[panel, tissues_union]
        subz = sub.sub(sub.mean(axis=1), axis=0).div(sub.std(axis=1).replace(0,1), axis=0)
        v = np.nanmax(np.abs(subz.values)) if subz.size>0 else 1.0
        fig_w = max(6, len(tissues_union)*0.5)
        fig_h = max(3, len(panel)*0.35)
        plt.figure(figsize=(fig_w, fig_h))
        ax = sns.heatmap(subz, cmap="vlag", center=0, vmin=-v, vmax=v, cbar_kws={"shrink":0.6})
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right", fontsize=8)
        plt.title("Combined panel (zscore)")
        plt.tight_layout()
        plt.savefig(OUT / f"heat_combined_panel.png", dpi=200, bbox_inches='tight')
        plt.close()
        if G is not None:
            H = G.copy()
            node_color = ["red" if n in panel else "lightgray" for n in H.nodes()]
            plt.figure(figsize=(8,8))
            pos = nx.spring_layout(H, seed=RANDOM_STATE)
            nx.draw_networkx_nodes(H, pos, node_size=20, node_color=node_color)
            nx.draw_networkx_edges(H, pos, alpha=0.08)
            nx.draw_networkx_labels(H, pos, labels={n:n for n in panel}, font_size=6)
            plt.title("Combined network highlight")
            plt.axis("off"); plt.tight_layout()
            plt.savefig(OUT / f"network_combined_panel.png", dpi=200, bbox_inches='tight')
            plt.close()

    pd.DataFrame(panel_records).to_csv(OUT / "top_panels.csv", index=False)
    pd.DataFrame(metrics_records).to_csv(OUT / "panel_classification_metrics.csv", index=False)
    details.to_csv(OUT / "module_candidate_details.csv", index=False)

    # annotation summary (best-effort)
    fam_to_flags = {}
    if LOCALIZATION.exists():
        loc = pd.read_csv(LOCALIZATION)
        lc = [c.lower().strip() for c in loc.columns]; loc.columns = lc
        if "family" in loc.columns:
            loc = loc.set_index("family")
            for fam in loc.index.unique():
                row = loc.loc[fam]
                sp = int(row.get("sp", 0)) if not pd.isna(row.get("sp",0)) else 0
                tm = int(row.get("tm", 0)) if not pd.isna(row.get("tm",0)) else 0
                fam_to_flags[fam] = {"sp":sp,"tm":tm}
    family_members = {}
    if FAM_MEMBERS.exists():
        fm = pd.read_csv(FAM_MEMBERS, dtype=str)
        cols = [c.lower().strip() for c in fm.columns]; fm.columns = cols
        if "family_name" in fm.columns and "members" in fm.columns:
            for _,r in fm.iterrows():
                fam = r["family_name"].strip()
                mems = r["members"]
                genes = [g.strip() for g in re.split(r"[;,]", str(mems)) if g.strip()]
                family_members[fam] = genes
    def load_gene_set(p):
        if not p.exists(): return set()
        df = pd.read_csv(p, header=None, dtype=str)
        return set([str(x).strip() for x in df.iloc[:,0].tolist() if str(x).strip()!=""])
    tf_set = load_gene_set(TFGENES); drug_set = load_gene_set(DRUGTARGET); onco_set = load_gene_set(ONCOGENES)
    top_panels_df = pd.read_csv(OUT / "top_panels.csv")
    ann_rows=[]
    for _,r in top_panels_df.iterrows():
        fams = [f.strip() for f in str(r["families"]).split(";") if f.strip()]
        spc=tmc=tfc=drugc=oncoc=0
        for f in fams:
            flags = fam_to_flags.get(f, {})
            spc += int(flags.get("sp",0)); tmc += int(flags.get("tm",0))
            mems = family_members.get(f, [])
            for g in mems:
                if g in tf_set: tfc+=1
                if g in drug_set: drugc+=1
                if g in onco_set: oncoc+=1
        ann_rows.append({"module":r.get("module",None),"panel_size":r["panel_size"],"families":r["families"],"sp_count":spc,"tm_count":tmc,"tf_count":tfc,"drug_count":drugc,"onco_count":oncoc,"auc":r.get("auc")})
    pd.DataFrame(ann_rows).to_csv(OUT / "panel_annotations.csv", index=False)

    print("Panel discovery complete. Results in:", OUT)
