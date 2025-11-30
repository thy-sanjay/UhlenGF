# UhlenGF

### **ML based Uhlén Human Tissue Proteome Gene Family Classifier and Discovery Panel**

---

## **Overview**
UhlenGF is a machine learning and network systems framework designed to analyse the Uhlén Human Tissue Proteome dataset at the Gene Family level rather than at the individual gene level. 
Instead of treating each gene as an isolated feature, UhlenGF aggregates protein coding genes into their HGNC defined gene families, generating a higher order representation of tissue biology that is more stable and interpretable. 
The framework learns how these gene families behave, co-express, and organize into latent embedding clusters and graph derived functional modules across human tissues. 

A key component of UhlenGF is comparative gene family discovery panel, where the user defines two custom tissue groups (e.g., Brain vs Non-Brain or High-Metabolic vs Low-Metabolic). UhlenGF then identifies the distinctive gene families that best discriminate the two tissue groups using a hybrid ML engine combining correlation graph centrality and representation learning signals. 

Overall, UhlenGF provides a gene family centred approach for systematic study of human tissue protein coding genes expression patterns. 
It produces biologically interpretable modules, tissue specific family signatures and discovery panel that can support downstream molecular and computational research.

---
**3. Dataset and Inputs**

UhlenGF operates on the following processed and derived files:

### **Expression Data**

* `exp_fpkm.csv` — Raw 32‑tissue FPKM matrix (gene × tissue).
* `exp_processed.csv` — Log1p + Z‑scored expression (cleaned).

### **Gene‑Family Mapping**

* `gf_map.csv` — Mapping of Ensembl gene IDs to HGNC gene families.
* `family_members.csv` — Gene members per family.
* `gf_exp.csv` — Collapsed family‑level expression matrix.

### **Embedding Features**

* `family_pca_embeddings.csv`
* `family_ae_embeddings.csv`
* `family_umap_coords.csv`
* `family_variance_explained.csv`

### **Graph Objects**

* `family_cor_matrix.csv` — Pearson correlation matrix.
* `family_graph.gml` — GraphML representation.
* `family_graph_edges.csv` — Weighted adjacency list.
* `modules_louvain.csv` — Family‑module assignments.

### **Annotation Files**

* `localization.csv` — SP/TM annotation.
* `tfgenes.csv` — Transcription factor gene list.
* `drugtarget.csv` — Druggable gene list.
* `oncogenes.csv` — Cancer gene list.

### **Tissue Group Definitions**

* `comparisions.csv` — Defines **Group1 vs Group2** tissue sets.

**Note: The required Uhlen and other datasets for running UhlenGF model are located in the `data/` directory.**

---
## **Methodology**

The UhlenGF workflow is structured into eight deterministic ML‑systems steps.

---

### **Step 1: Expression Preprocessing**

* Drop genes with >20% missing values.
* Log1p transform and row wise Z‑score.
* Preserve gene_name metadata.
* Output: `exp_processed.csv`.

This step ensures stable ML behaviour and removes scale biases across tissues.

---

### **Step 2: Gene Family Mapping & Family Expression Matrix**

* Map Ensembl genes to HGNC gene families.
* Aggregate expression: mean FPKM per family.
* Output: `gf_exp.csv`.

This reduces noise, enhances interpretability, and provides a compact feature space.

---

### **Step 3: Representation Learning (Embedding Based ML)**

Three independent embedding methods capture different aspects of the latent structure:

1. **PCA** - linear variance structure.
2. **UMAP** - non‑linear manifold geometry.
3. **Autoencoder** - learned deep latent representation.

Outputs:

* `family_pca_embeddings.csv`
* `family_ae_embeddings.csv`
* `family_umap_coords.csv`
* `family_variance_explained.csv`

These embeddings serve as inputs for clustering and candidate selection.

---

### **Step 4: Embedding Based Clustering**

* K‑Means clustering on PCA and AE embeddings.
* Evaluate clusters via silhouette, Davies–Bouldin.
* Output: `family_cluster_assignments.csv`.

This identifies expression‑space modules independent of graph structure.

---

### **Step 5: Graph Construction & Module Detection (Graph Based ML)**

* Compute Pearson correlation between families.
* Build weighted undirected graph.
* Apply Louvain for community detection.
* Compute centrality metrics: degree, betweenness.
* Output: `modules_louvain.csv`, `family_graph.gml`, `family_graph_edges.csv`.

Graph modules provide topology‑informed patterns complementing embeddings.

---

### **Step 6: Biological Annotation**

Each family is annotated with:

* SP (secretory protein)
* TM (transmembrane)
* TF (transcription factor)
* Drug‑target status
* Cancer associations

Outputs:

* `annotated_modules.csv`
* `annotated_module_details.csv`

This enables functional interpretation of discovered modules.

---

### **Step 7: Validation**

Validation includes:

* Within module vs between module correlation statistics.
* Embedding silhouette validation.
* Small ML tasks to verify separability.

Output:

* `validation_summary.txt`.

---

### **Step 8: Module Based Gene Family Discovery Panel Discovery**

This is the core deliverable of UhlenGF.

#### **8.1 Candidate Family Selection**

Selection uses both ML engines:

**Engine 1 (Embedding Space Influence):**

* Families closest to module centroid in PCA/AE space.

**Engine 2 (Graph Space Influence):**

* High‑degree and high‑betweenness families.

Unified candidate sets are created for each module.

#### **8.2 Groupwise Classification**

`comparisions.csv` defines:

* Group1 tissues (e.g., high metabolic tissues)
* Group2 tissues (e.g., low metabolic tissues)

Classification is **group vs group**, not tissue vs tissue.

#### **8.3 Greedy Panel Construction**

For each module:

* Greedy forward selection.
* Logistic regression with CV.
* Effect size fallback for small groups.
* Permutation testing.

Outputs:

* `top_panels.csv`
* `panel_classification_metrics.csv`
* `panel_annotations.csv`
* Module specific heatmaps and network highlight plots.

These panels represent small, interpretable sets of families that capture tissue group identity.

---
# UhlenGF Execution Steps

```cmd
cd "Project Directory"
python -m venv .venv
.\.venv\Scripts\activate

pip install --upgrade pip
pip install -r requirements.txt

python src/load_data.py
python src/gf_map.py
python src/rep_learn.py
python src/gf_cluster.py
python src/gf_graph.py
python src/gf_annotate.py
python src/validate.py
python src/visualize_results.py
python src/panel_discovery.py
```

**Gene Family Discovery Panel - What the Insights Say:** 

The gene families identified by the panel (in top_panels.csv) are the families whose expression behaviour most uniquely distinguishes one user defined tissue group from another. Think of each tissue group as a city, and each gene family as a type of industry such as transport, communication or power generation. All cities may contain all industries, but each group develops its own characteristic pattern: some industries become highly active, some remain quiet, and some coordinate with one another in ways that are unique to that group. The machine learning system identifies the industries whose activity levels and interconnections make one group’s economic profile stand out relative to the other. These families form the Top Panel, a compact fingerprint capturing what makes Group1 different from Group2, not because the families exist only in one group, but because they express in a distinctive and reproducible pattern across the tissues of that group.

The gene families in the Top Panels therefore act like a signature accent of the selected tissue group. They may appear across many tissues, but the rhythm and structure of their expression is characteristic of one group and not replicated by the other, making them reliable family-level discriminators.

---
