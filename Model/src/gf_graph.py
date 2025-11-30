from pathlib import Path
import sys
import numpy as np
import pandas as pd
import networkx as nx
import warnings
import csv

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"

EXP_F = RESULTS / "gf_exp.csv"     # UPDATED
MEMBER_F = DATA / "family_members.csv"

OUT_COR = RESULTS / "family_cor_matrix.csv"
OUT_EDGES = RESULTS / "family_graph_edges.csv"
OUT_GML = RESULTS / "family_graph.gml"
OUT_STATS = RESULTS / "family_graph_stats.csv"
OUT_MOD_STATS = RESULTS / "module_correlation_stats.csv"

OUT_LEIDEN = RESULTS / "modules_leiden.csv"
OUT_LOUVAIN = RESULTS / "modules_louvain.csv"
OUT_FALLBACK = RESULTS / "modules_fallback.csv"

RESULTS.mkdir(exist_ok=True, parents=True)

ABS_COR_THRESHOLD = 0.60
K_NEIGHBORS = 5

if not EXP_F.exists():
    print("ERROR: gf_exp.csv missing")
    sys.exit(1)

df = pd.read_csv(EXP_F, index_col=0)
families = df.index.tolist()
X = df.values
n = len(families)

print(f"Loaded {n} families × {df.shape[1]} tissues")

print("Computing correlation matrix.")
corr = np.corrcoef(X)
corr = np.nan_to_num(corr)
np.fill_diagonal(corr, 1)

df_corr = pd.DataFrame(corr, index=families, columns=families)
df_corr.to_csv(OUT_COR)
print("Saved:", OUT_COR)

print("Selecting graph edges.")

edges = {}

for i in range(n):
    for j in range(i+1, n):
        w = corr[i, j]
        if abs(w) >= ABS_COR_THRESHOLD:
            edges[(i, j)] = w

for i in range(n):
    row = corr[i].copy()
    row[i] = -np.inf
    topk = np.argpartition(-row, K_NEIGHBORS)[:K_NEIGHBORS]
    for j in topk:
        if i == j:
            continue
        w = corr[i, j]
        key = tuple(sorted([i, j]))
        if key not in edges or abs(w) > abs(edges[key]):
            edges[key] = w

print("Total selected edges:", len(edges))

G = nx.Graph()

family_size_map = {}
if MEMBER_F.exists():
    fm = pd.read_csv(MEMBER_F)
    for _, r in fm.iterrows():
        fam = str(r["family_name"])
        members = str(r["members"])
        if pd.isna(members) or members.strip() == "":
            family_size_map[fam] = 0
        else:
            family_size_map[fam] = len([x for x in members.split(",") if x.strip()])

for idx, fam in enumerate(families):
    G.add_node(fam, family_name=fam, family_size=family_size_map.get(fam, 1))

for (i, j), w in edges.items():
    a = families[i]
    b = families[j]
    G.add_edge(a, b, weight=float(w), abs_weight=abs(float(w)))

degrees = [d for _, d in G.degree()]
stats = {
    "n_nodes": G.number_of_nodes(),
    "n_edges": G.number_of_edges(),
    "avg_degree": float(np.mean(degrees)),
    "density": nx.density(G)
}

pd.DataFrame([stats]).to_csv(OUT_STATS, index=False)
print("Saved:", OUT_STATS)

df_edges = pd.DataFrame(
    [(u, v, d["weight"], d["abs_weight"]) for u, v, d in G.edges(data=True)],
    columns=["source", "target", "weight", "abs_weight"]
)
df_edges.to_csv(OUT_EDGES, index=False)
print("Saved:", OUT_EDGES)

try:
    nx.write_gml(G, OUT_GML)
    print("Saved:", OUT_GML)
except Exception as e:
    warnings.warn(f"GML write failed: {e}")

leiden_success = False
try:
    import igraph as ig
    import leidenalg

    print("Attempting Leiden community detection.")

    ig_edges = []
    for u, v, d in G.edges(data=True):
        w = float(d["abs_weight"])
        ig_edges.append((u, v, w))

    ig_g = ig.Graph.TupleList(((u, v, {"abs_weight": w}) for u, v, w in ig_edges),
                              directed=False, edge_attrs=["abs_weight"])

    ig_g.es["abs_weight"] = [float(x) for x in ig_g.es["abs_weight"]]

    partition = leidenalg.find_partition(
        ig_g,
        leidenalg.RBConfigurationVertexPartition,
        weights=ig_g.es["abs_weight"]
    )

    mod_map = {}
    for cid, cluster in enumerate(partition):
        for vid in cluster:
            node = ig_g.vs[vid]["name"]
            mod_map[node] = cid

    pd.DataFrame(
        [(k, v) for k, v in mod_map.items()],
        columns=["family_name", "module"]
    ).to_csv(OUT_LEIDEN, index=False)

    print("Saved Leiden modules:", OUT_LEIDEN)
    leiden_success = True

except Exception as e:
    print("Leiden failed:", e)

louvain_success = False
try:
    import community as community_louvain

    print("Attempting Louvain.")

    for u, v, d in G.edges(data=True):
        d["abs_weight"] = float(d["abs_weight"])

    part = community_louvain.best_partition(G, weight="abs_weight")

    pd.DataFrame(
        [(k, v) for k, v in part.items()],
        columns=["family_name", "module"]
    ).to_csv(OUT_LOUVAIN, index=False)

    print("Saved Louvain modules:", OUT_LOUVAIN)
    louvain_success = True

except Exception as e:
    print("Louvain failed:", e)

if not leiden_success and not louvain_success:
    print("Both Leiden and Louvain failed. Using fallback.")

    from networkx.algorithms.community import greedy_modularity_communities

    comms = list(greedy_modularity_communities(G, weight="abs_weight"))
    mod_map = {}
    for cid, comm in enumerate(comms):
        for fam in comm:
            mod_map[fam] = cid

    pd.DataFrame(
        [(k, v) for k, v in mod_map.items()],
        columns=["family_name", "module"]
    ).to_csv(OUT_FALLBACK, index=False)

    print("Saved fallback modules:", OUT_FALLBACK)

print("Computing module correlation statistics.")

if OUT_LEIDEN.exists():
    df_mod = pd.read_csv(OUT_LEIDEN)
elif OUT_LOUVAIN.exists():
    df_mod = pd.read_csv(OUT_LOUVAIN)
elif OUT_FALLBACK.exists():
    df_mod = pd.read_csv(OUT_FALLBACK)
else:
    df_mod = None

if df_mod is None or df_mod.shape[0] == 0:
    print("No module file; skipping.")
else:
    df_mod = df_mod.iloc[:, :2].copy()
    df_mod.columns = ["family_name", "module"]
    df_mod["family_name"] = df_mod["family_name"].astype(str)
    try:
        df_mod["module"] = df_mod["module"].astype(int)
    except Exception:
        df_mod["module"] = pd.Categorical(df_mod["module"]).codes

    module_groups = df_mod.groupby("module")["family_name"].apply(list).to_dict()
    family_index_map = {fam: i for i, fam in enumerate(families)}
    corr_mat = corr

    records = []
    for m, members in module_groups.items():
        idxs = [family_index_map[f] for f in members if f in family_index_map]
        if len(idxs) < 2:
            records.append({"module": int(m), "size": len(members), "within_mean": np.nan, "between_mean": np.nan, "perm_p": np.nan})
            continue
        sub = corr_mat[np.ix_(idxs, idxs)]
        nmem = len(idxs)
        if nmem*(nmem-1)/2 > 0:
            upper_inds = np.triu_indices(nmem, k=1)
            within_vals = sub[upper_inds]
            within_mean = float(np.nanmean(within_vals))
        else:
            within_mean = float(np.nanmean(sub))
        non_idxs = [i for i in range(len(families)) if i not in idxs]
        if len(non_idxs) == 0:
            between_mean = np.nan
        else:
            between_vals = corr_mat[np.ix_(idxs, non_idxs)].ravel()
            between_mean = float(np.nanmean(between_vals))
        rng = np.random.default_rng(0)
        perm_counts = 0
        for _ in range(500):
            perm_idxs = rng.choice(len(families), size=len(idxs), replace=False)
            subp = corr_mat[np.ix_(perm_idxs, perm_idxs)]
            if len(perm_idxs) >= 2:
                pv = float(np.nanmean(subp[np.triu_indices(len(perm_idxs), k=1)]))
            else:
                pv = float(np.nanmean(subp))
            if pv >= within_mean:
                perm_counts += 1
        pval = (perm_counts + 1) / (500 + 1)
        records.append({
            "module": int(m),
            "size": len(members),
            "within_mean": within_mean,
            "between_mean": between_mean,
            "perm_p": pval
        })

    pd.DataFrame(records).to_csv(OUT_MOD_STATS, index=False)
    print("Saved:", OUT_MOD_STATS)

print("gf_graph.py complete.")