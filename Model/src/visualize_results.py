from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as mpdf
import networkx as nx
import matplotlib.colors as mcolors
import warnings

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGDIR = RESULTS / "figures"
FIGDIR.mkdir(parents=True, exist_ok=True)

PCA_EMB = RESULTS / "family_pca_embeddings.csv"
AE_EMB = RESULTS / "family_ae_embeddings.csv"
UMAP_EMB = RESULTS / "family_umap_coords.csv"
MODULES = RESULTS / "modules_louvain.csv"
ANNOT_DETAILS = RESULTS / "annotated_module_details.csv"
CORR_MATRIX = RESULTS / "family_cor_matrix.csv"
GRAPH_GML = RESULTS / "family_graph.gml"
ANNOT_SUM = RESULTS / "annotated_modules.csv"

OUT_PCA = FIGDIR / "pca_scatter.png"
OUT_UMAP = FIGDIR / "umap_scatter.png"
OUT_AE = FIGDIR / "ae_scatter.png"
OUT_HEAT = FIGDIR / "correlation_heatmap.png"
OUT_NET = FIGDIR / "network_plot.png"
OUT_BAR = FIGDIR / "annotation_bars.png"
OUT_PDF = FIGDIR / "figure_panel.pdf"

def load_modules(path):
    if not path.exists():
        return None
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

def load_embedding_df(path):
    if not path.exists():
        return None
    try:
        emb = pd.read_csv(path, index_col=0, low_memory=False)
    except Exception:
        return None
    emb.index = emb.index.astype(str)
    emb_reset = emb.reset_index()
    idxcol = emb_reset.columns[0]
    emb_reset = emb_reset.rename(columns={idxcol: "family_name"})
    dim_cols = [c for c in emb_reset.columns if c != "family_name"]
    if len(dim_cols) == 0:
        return None
    new_names = ["family_name"] + [f"dim{i+1}" for i in range(len(dim_cols))]
    col_map = dict(zip(emb_reset.columns, new_names))
    emb_reset = emb_reset.rename(columns=col_map)
    for c in new_names[1:]:
        emb_reset[c] = pd.to_numeric(emb_reset[c], errors="coerce")
    return emb_reset

def palette_hex(n):
    base = plt.get_cmap("tab20")
    if n <= 20:
        cols = [mcolors.to_hex(base(i)) for i in range(n)]
    else:
        cmap = plt.get_cmap("hsv")
        cols = [mcolors.to_hex(cmap(i / max(1, n - 1))) for i in range(n)]
    return cols

modules_df = load_modules(MODULES)
if modules_df is None:
    raise FileNotFoundError(f"Module file not found: {MODULES}")

pca_df = load_embedding_df(PCA_EMB)
ae_df = load_embedding_df(AE_EMB)
umap_df = load_embedding_df(UMAP_EMB)

annot_details = None
if ANNOT_DETAILS.exists():
    try:
        annot_details = pd.read_csv(ANNOT_DETAILS, dtype=str, low_memory=False)
    except Exception:
        annot_details = None

module_ids = sorted(modules_df["module"].unique().tolist())
colors = palette_hex(len(module_ids))
color_map = {m: colors[i % len(colors)] for i, m in enumerate(module_ids)}

pdf = mpdf.PdfPages(str(OUT_PDF))

def plot_embedding(emb_df, title, out_png):
    if emb_df is None or emb_df.shape[0] == 0:
        warnings.warn(f"{title}: embedding not available or empty.")
        return
    merged = modules_df.merge(emb_df, on="family_name", how="inner")
    if merged.shape[0] == 0:
        warnings.warn(f"{title}: no overlap between modules and embedding.")
        return
    xcol, ycol = "dim1", "dim2"
    if xcol not in merged.columns or ycol not in merged.columns:
        warnings.warn(f"{title}: embedding has fewer than 2 dims.")
        return
    plt.figure(figsize=(7,6))
    for m in module_ids:
        sub = merged[merged["module"].astype(int) == int(m)]
        if sub.shape[0] == 0:
            continue
        x = pd.to_numeric(sub[xcol], errors="coerce").values
        y = pd.to_numeric(sub[ycol], errors="coerce").values
        mask = ~np.isnan(x) & ~np.isnan(y)
        if mask.sum() == 0:
            continue
        plt.scatter(x[mask], y[mask], label=f"M{m} (n={mask.sum()})", s=24, alpha=0.85, c=color_map[m])
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.legend(bbox_to_anchor=(1.05,1), loc="upper left", fontsize="small")
    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    pdf.savefig()
    plt.close()

plot_embedding(pca_df, "PCA scatter (dim1 vs dim2) colored by module", OUT_PCA)
plot_embedding(umap_df, "UMAP scatter (dim1 vs dim2) colored by module", OUT_UMAP)
plot_embedding(ae_df, "Autoencoder embedding scatter (dim1 vs dim2) colored by module", OUT_AE)

if CORR_MATRIX.exists():
    try:
        corr = pd.read_csv(CORR_MATRIX, index_col=0)
        corr.index = corr.index.astype(str)
        corr.columns = corr.columns.astype(str)
        order_df = modules_df[modules_df["family_name"].isin(corr.index)].copy()
        if order_df.shape[0] > 0:
            order_df = order_df.sort_values(["module", "family_name"])
            ordered = order_df["family_name"].tolist()
            corr_ord = corr.loc[ordered, ordered]
            plt.figure(figsize=(8,8))
            im = plt.imshow(corr_ord.values, aspect="auto", interpolation="nearest", vmin=-1, vmax=1, cmap="RdBu_r")
            plt.colorbar(im, fraction=0.046, pad=0.04)
            plt.title("Family correlation matrix (ordered by module)")
            plt.xticks([], [])
            plt.yticks([], [])
            boundaries = []
            prev = None
            for i, fam in enumerate(ordered):
                mod = int(order_df.loc[order_df["family_name"] == fam, "module"].values[0])
                if prev is None:
                    prev = mod
                    continue
                if mod != prev:
                    boundaries.append(i)
                    prev = mod
            for b in boundaries:
                plt.axhline(b - 0.5, color="k", linewidth=0.5, alpha=0.5)
                plt.axvline(b - 0.5, color="k", linewidth=0.5, alpha=0.5)
            plt.tight_layout()
            plt.savefig(OUT_HEAT, dpi=200)
            pdf.savefig()
            plt.close()
        else:
            warnings.warn("Correlation heatmap: no overlapping families.")
    except Exception as e:
        warnings.warn(f"Failed correlation heatmap: {e}")
else:
    warnings.warn("Correlation matrix not found.")

if GRAPH_GML.exists():
    try:
        G = nx.read_gml(str(GRAPH_GML))
        G = nx.relabel_nodes(G, lambda x: str(x))
        node_colors = []
        node_sizes = []
        size_map = {}
        if annot_details is not None and "family_name" in annot_details.columns:
            if "family_size" in annot_details.columns:
                for _, r in annot_details.iterrows():
                    try:
                        size_map[str(r["family_name"])] = int(float(r["family_size"]))
                    except Exception:
                        size_map[str(r["family_name"])] = 1
            else:
                for _, r in annot_details.iterrows():
                    flags = 0
                    for c in ["is_sp","is_tm","is_tf","is_drug","is_onco"]:
                        if c in annot_details.columns and pd.notna(r.get(c)):
                            try:
                                flags += int(float(r.get(c)))
                            except Exception:
                                pass
                    size_map[str(r["family_name"])] = max(1, flags)
        for n in G.nodes():
            fam = str(n)
            row = modules_df[modules_df["family_name"] == fam]
            if row.shape[0] > 0:
                mod = int(row["module"].iloc[0])
                node_colors.append(color_map.get(mod, "#999999"))
            else:
                node_colors.append("#999999")
            node_sizes.append(30 + 8 * int(size_map.get(fam, 1)))
        plt.figure(figsize=(9,9))
        pos = nx.spring_layout(G, k=0.12, iterations=80, seed=0)
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color=node_colors, linewidths=0.2)
        nx.draw_networkx_edges(G, pos, alpha=0.15, width=0.3)
        try:
            sizes_arr = np.array(node_sizes)
            top_idx = np.argsort(sizes_arr)[-10:]
            labels = {list(G.nodes())[i]: list(G.nodes())[i] for i in top_idx}
            nx.draw_networkx_labels(G, pos, labels=labels, font_size=6)
        except Exception:
            pass
        plt.title("Family co-expression network")
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(OUT_NET, dpi=200)
        pdf.savefig()
        plt.close()
    except Exception as e:
        warnings.warn(f"Failed network plot: {e}")
else:
    warnings.warn("Graph GML not found.")

if ANNOT_SUM.exists():
    try:
        ann = pd.read_csv(ANNOT_SUM)
        if "module" in ann.columns:
            cols = [c for c in ann.columns if c.endswith("_hits")]
            if len(cols) == 0 and annot_details is not None and "module" in annot_details.columns:
                agg = annot_details.groupby("module")[["is_sp","is_tm","is_tf","is_drug","is_onco"]].sum()
                agg = agg.astype(int)
                agg.plot(kind="bar", figsize=(9,6))
                plt.title("Annotation counts per module")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(OUT_BAR, dpi=200)
                pdf.savefig()
                plt.close()
            elif len(cols) > 0:
                plot_df = ann[["module"] + cols].set_index("module").sort_index()
                plot_df.columns = [c.replace("_hits","") for c in plot_df.columns]
                plot_df.plot(kind="bar", figsize=(9,6))
                plt.title("Annotation hits per module")
                plt.ylabel("count")
                plt.tight_layout()
                plt.savefig(OUT_BAR, dpi=200)
                pdf.savefig()
                plt.close()
    except Exception as e:
        warnings.warn(f"Failed annotation barplots: {e}")
else:
    warnings.warn("annotated_modules.csv not found.")

pdf.close()
print("Figures saved to:", FIGDIR)