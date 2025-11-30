from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

RANDOM_STATE = 42

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

PCA_F = RESULTS_DIR / "family_pca_embeddings.csv"
AE_F = RESULTS_DIR / "family_ae_embeddings.csv"

OUT_ASSIGN = RESULTS_DIR / "family_cluster_assignments.csv"
OUT_METRICS = RESULTS_DIR / "cluster_metrics.json"
OUT_PCA_ELBOW = RESULTS_DIR / "pca_kmeans_elbow.csv"
OUT_AE_ELBOW = RESULTS_DIR / "ae_kmeans_elbow.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not PCA_F.exists():
    raise FileNotFoundError(f"Missing PCA embeddings: {PCA_F}")
if not AE_F.exists():
    raise FileNotFoundError(f"Missing AE embeddings: {AE_F}")

df_pca = pd.read_csv(PCA_F, index_col=0)
df_ae = pd.read_csv(AE_F, index_col=0)
df_ae = df_ae.loc[:, ~df_ae.columns.str.contains("#")]

print("Loaded PCA embeddings:", df_pca.shape)
print("Loaded AE embeddings:", df_ae.shape)

families = df_pca.index.tolist()

def evaluate_clusters(X, labels):
    if len(set(labels)) <= 1:
        return {"silhouette": None, "calinski_harabasz": None, "davies_bouldin": None}
    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
        "davies_bouldin": float(davies_bouldin_score(X, labels))
    }

def run_kmeans(X, k_range, elbow_out_path):
    sse = []
    all_labels = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        km.fit(X)
        sse.append({"k": k, "sse": km.inertia_})
        all_labels[k] = km.labels_
    pd.DataFrame(sse).to_csv(elbow_out_path, index=False)
    return all_labels

cluster_results = {}
metrics_results = {}

K_RANGE = list(range(3, 11))

X_pca = df_pca.values
pca_kmeans_all = run_kmeans(X_pca, K_RANGE, OUT_PCA_ELBOW)

best_k_pca = None
best_sil = -999
best_labels_pca = None
for k in K_RANGE:
    labels = pca_kmeans_all[k]
    score = silhouette_score(X_pca, labels)
    if score > best_sil:
        best_sil = score
        best_k_pca = k
        best_labels_pca = labels

pca_gmm = GaussianMixture(n_components=best_k_pca, random_state=RANDOM_STATE)
pca_gmm_labels = pca_gmm.fit_predict(X_pca)

pca_spec = SpectralClustering(
    n_clusters=best_k_pca, random_state=RANDOM_STATE, affinity='nearest_neighbors'
)
pca_spec_labels = pca_spec.fit_predict(X_pca)

pca_agg = AgglomerativeClustering(n_clusters=best_k_pca)
pca_agg_labels = pca_agg.fit_predict(X_pca)

metrics_results["PCA"] = {
    "best_k": best_k_pca,
    "KMeans": evaluate_clusters(X_pca, best_labels_pca),
    "GMM": evaluate_clusters(X_pca, pca_gmm_labels),
    "Spectral": evaluate_clusters(X_pca, pca_spec_labels),
    "Agglomerative": evaluate_clusters(X_pca, pca_agg_labels)
}

X_ae = df_ae.values
ae_kmeans_all = run_kmeans(X_ae, K_RANGE, OUT_AE_ELBOW)

best_k_ae = None
best_sil_ae = -999
best_labels_ae = None
for k in K_RANGE:
    labels = ae_kmeans_all[k]
    score = silhouette_score(X_ae, labels)
    if score > best_sil_ae:
        best_sil_ae = score
        best_k_ae = k
        best_labels_ae = labels

ae_gmm = GaussianMixture(n_components=best_k_ae, random_state=RANDOM_STATE)
ae_gmm_labels = ae_gmm.fit_predict(X_ae)

ae_spec = SpectralClustering(
    n_clusters=best_k_ae, random_state=RANDOM_STATE, affinity='nearest_neighbors'
)
ae_spec_labels = ae_spec.fit_predict(X_ae)

ae_agg = AgglomerativeClustering(n_clusters=best_k_ae)
ae_agg_labels = ae_agg.fit_predict(X_ae)

metrics_results["AE"] = {
    "best_k": best_k_ae,
    "KMeans": evaluate_clusters(X_ae, best_labels_ae),
    "GMM": evaluate_clusters(X_ae, ae_gmm_labels),
    "Spectral": evaluate_clusters(X_ae, ae_spec_labels),
    "Agglomerative": evaluate_clusters(X_ae, ae_agg_labels)
}

assignments = []
for fam, lab in zip(families, best_labels_pca):
    assignments.append([fam, int(lab)])
pd.DataFrame(assignments, columns=["family_name", "cluster"]).to_csv(OUT_ASSIGN, index=False)

with open(OUT_METRICS, "w") as f:
    json.dump(metrics_results, f, indent=4)

print("Clustering complete.")