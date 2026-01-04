# Wholesale-Client-Segmentation-with-Unsupervised-Learning-
**K-Means · Hierarchical Clustering (Ward) · DBSCAN · Cluster Validation · Outlier Detection · Customer Analytics**

---

## Overview
This project builds a **client segmentation and anomaly-aware clustering pipeline** for a wholesale/retail distribution context. Using purchase behavior features (e.g., *Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen*), the notebook applies and compares multiple **unsupervised learning paradigms** to produce **actionable customer segments** for targeting, pricing, assortment planning, and operational strategy.

The workflow is designed to be **methodologically rigorous**: it includes **feature scaling**, **cluster model selection**, **quantitative validation (silhouette)**, **hierarchical structure interpretation (dendrograms)**, and **density-based discovery (DBSCAN)** to handle **non-spherical clusters and outliers**.

---

## Application & Business Case (What this solves)
Wholesale businesses often serve heterogeneous client types (e.g., restaurants, grocers, institutional buyers) whose purchasing profiles differ significantly across product groups. A single “average customer” strategy leads to:
- inefficient marketing and promotions,
- poor assortment and bundling decisions,
- misallocated sales resources,
- inability to detect unusual spending patterns (fraud, one-off bulk purchases, data issues).

This project solves the **client segmentation problem** by learning **data-driven customer archetypes** from transactional spending patterns and then using the learned segments to:
- recommend segment-specific strategies,
- identify outliers/extreme spenders,
- assign a new incoming client to a segment for downstream decisioning.

---

## Dataset
The notebook loads:

- `wholesale_clients.csv`

### Features (behavioral spending + context)
- **Spending**: `Fresh`, `Milk`, `Grocery`, `Frozen`, `Detergents_Paper`, `Delicassen`
- **Metadata**: `Channel`, `Region`

> Note: The clustering is performed on standardized numerical features to ensure distance-based methods behave correctly.

---

## Methodology (End-to-End Pipeline)

### 1) Data Preparation
- Load `wholesale_clients.csv`
- Basic quality checks
- Feature selection for clustering
- **Standardization** via `StandardScaler`  
  (critical for KNN-distance geometry in K-Means / Ward / DBSCAN)

---

### 2) K-Means Clustering (Centroid-Based Segmentation)
- Fit K-Means across candidate values of **K**
- Evaluate using:
  - **Silhouette score** (primary validation metric)
  - Cluster size balance (practical interpretability)
  - Cluster centroid profiling (spend archetypes)

The notebook explicitly discusses the trade-off between:
- high silhouette vs.
- operational usefulness / balanced segmentation

and proceeds with a **practically meaningful K** for interpretable segments.

---

### 3) Hierarchical Clustering (Ward Linkage)
- Fit **Agglomerative / Ward-style** hierarchical clustering
- Produce and interpret **dendrograms** to reveal:
  - which product categories co-move (e.g., tight coupling between grocery-related categories),
  - multi-scale structure in the customer base,
  - the presence of micro-clusters / extreme profiles.

This provides a **model-agnostic structural view** of customer similarity.

---

### 4) DBSCAN (Density-Based Clustering + Outlier Detection)
- Perform DBSCAN with parameter exploration over:
  - `eps` (neighborhood radius)
  - `min_samples` (density threshold)
- Select best configuration using **silhouette score** on non-noise points
- Fit final DBSCAN using the selected `(eps, min_samples)` and interpret:
  - cluster assignments,
  - **noise points (-1)** as candidate anomalies/outliers.

This stage is particularly valuable because DBSCAN:
- does not require pre-specifying K,
- can detect irregular cluster shapes,
- explicitly identifies outliers.

---

### 5) Comparative Model Assessment
The notebook compares the three clustering families:

| Method | Strength | When it wins |
|---|---|---|
| **K-Means** | Fast, interpretable centroids | When clusters are roughly spherical + you want segment “profiles” |
| **Hierarchical (Ward)** | Reveals multi-scale structure | When you want structure discovery + dendrogram insight |
| **DBSCAN** | Finds dense regions + outliers | When outliers matter or shapes are non-spherical |

---

### 6) Segment Recommendation (Business Translation)
Clusters are converted into **client archetypes** using centroid / cluster profile interpretation, e.g.:
- typical buyers vs.
- high-volume niche profiles vs.
- extreme spend outliers (rare but impactful).

This supports downstream actions such as:
- targeted promotions,
- sales prioritization,
- assortment/bundling strategy,
- anomaly monitoring.

---

### 7) Predict Segment for a New Client (Operationalization)
The notebook includes a practical operational step:
- take a new client’s spending vector,
- apply the **same scaler** (training-time transform),
- assign the client to the appropriate learned cluster (e.g., via trained K-Means).

This mirrors how segmentation is used in production:
**new entity → standardized features → segment label → business rule/action**.

---

## Technical Stack
- **Python (Jupyter Notebook)**
- Core libraries:
  - `pandas`, `numpy`
  - `scikit-learn` (`StandardScaler`, `KMeans`, `DBSCAN`, `silhouette_score`)
  - `scipy` (hierarchical clustering + dendrograms)
  - `matplotlib`, `seaborn` (visual analytics)

---

## Repository Structure
```text
├── ADVANCED_ML_08_HELIA_261224416.ipynb
├── wholesale_clients.csv   # required input (place in repo root)
└── README.md
