import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
from pathlib import Path
import sys
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import umap
import warnings

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

INPUT_F = RESULTS_DIR / "gf_exp.csv"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)

if not INPUT_F.exists():
    print(f"ERROR: file not found: {INPUT_F}")
    sys.exit(1)

df = pd.read_csv(INPUT_F, index_col=0)
families = df.index.tolist()
X = df.values.astype(float)
print("Loaded gf_exp.csv:", X.shape)

scaler = StandardScaler(with_mean=True, with_std=False)
X_centered = scaler.fit_transform(X)

N_PCA = 8
pca = PCA(n_components=N_PCA, random_state=RANDOM_STATE)
X_pca = pca.fit_transform(X_centered)

pca_cols = [f"PC{i+1}" for i in range(N_PCA)]
df_pca = pd.DataFrame(X_pca, index=families, columns=pca_cols)
df_pca.to_csv(RESULTS_DIR / "family_pca_embeddings.csv")

exp_var = pca.explained_variance_ratio_
cum = np.cumsum(exp_var)
df_var = pd.DataFrame({
    "component": list(range(1, N_PCA+1)),
    "explained_variance_ratio": exp_var,
    "cumulative_variance": cum
})
df_var.to_csv(RESULTS_DIR / "family_variance_explained.csv", index=False)

print("PCA complete.")

reducer = umap.UMAP(
    n_neighbors=15,
    min_dist=0.1,
    metric="euclidean",
    random_state=RANDOM_STATE
)

pca20 = PCA(n_components=min(20, X_centered.shape[1]), random_state=RANDOM_STATE)
X_pca20 = pca20.fit_transform(X_centered)
coords = reducer.fit_transform(X_pca20)

df_umap = pd.DataFrame(coords, index=families, columns=["UMAP1", "UMAP2"])
df_umap.to_csv(RESULTS_DIR / "family_umap_coords.csv")

print("UMAP complete.")

ae_used = False
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, callbacks
    ae_used = True
except Exception:
    ae_used = False
    warnings.warn("TensorFlow not found. Autoencoder will be skipped.")

ae_output = RESULTS_DIR / "family_ae_embeddings.csv"

if ae_used:
    tf.random.set_seed(RANDOM_STATE)
    X_ae = X_centered.astype("float32")
    n_features = X_ae.shape[1]

    bottleneck = 8
    hid = 16

    inp = keras.Input(shape=(n_features,))
    x = layers.Dense(hid, activation="relu")(inp)
    bott = layers.Dense(bottleneck, activation="linear")(x)
    x = layers.Dense(hid, activation="relu")(bott)
    out = layers.Dense(n_features, activation="linear")(x)

    autoencoder = keras.Model(inp, out)
    encoder = keras.Model(inp, bott)

    autoencoder.compile(optimizer="adam", loss="mse")

    es = callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,
        restore_best_weights=True,
        verbose=1
    )

    autoencoder.fit(
        X_ae, X_ae,
        epochs=40,
        batch_size=16,
        validation_split=0.1,
        callbacks=[es],
        verbose=2
    )

    ae_emb = encoder.predict(X_ae)
    df_ae = pd.DataFrame(ae_emb, index=families, columns=[f"AE{i+1}" for i in range(bottleneck)])
    df_ae.to_csv(ae_output)

    print("Autoencoder complete.")
else:
    with open(ae_output, "w") as fh:
        fh.write("")
    print("Autoencoder skipped.")