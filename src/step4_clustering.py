"""
STEP 4 – Clustering (COMP 257) [IMPROVED]

Clustering on reduced UMIST face features using:
1) K-Means (tuned k by silhouette + ALSO forced k=20 comparison)
2) Hierarchical (Agglomerative; tuned linkage)
(Optional) 3) DBSCAN (grid search)

What it does:
- Loads scaled train split from outputs/umist_splits.npz
- Reduces dimensionality with PCA (50D for clustering, 2D for plotting)
- Produces 2D plots colored by cluster assignment
- Computes purity + label composition (top labels per cluster)
- Saves results JSON + summary CSV for easy report writing
"""

import os
import json
import csv
import traceback
import numpy as np
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
from collections import Counter

OUT_DIR = "outputs"
SPLITS_PATH = os.path.join(OUT_DIR, "umist_splits.npz")
STEP4_DIR = os.path.join(OUT_DIR, "step4")
RANDOM_STATE = 42


# -----------------------
# Utilities
# -----------------------
def load_data(path):
    data = np.load(path)
    return data["X_train"], data["X_val"], data["X_test"], data["y_train"], data["y_val"], data["y_test"]


def compute_purity(y_true, y_pred):
    """
    Purity = sum(max label count in each cluster) / total points
    """
    total = len(y_true)
    if total == 0:
        return 0.0

    purity_sum = 0
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        labels = y_true[idx]
        if len(labels) == 0:
            continue
        most_common = Counter(labels).most_common(1)[0][1]
        purity_sum += most_common

    return purity_sum / total


def cluster_composition_table(y_true, y_pred, top_k=3):
    """
    Returns a dict:
      cluster_id -> {size, top_labels: [{label, count, pct}, ...]}
    """
    result = {}
    for c in np.unique(y_pred):
        idx = np.where(y_pred == c)[0]
        labels = y_true[idx]
        size = len(labels)
        counts = Counter(labels).most_common(top_k)
        top_labels = []
        for lab, cnt in counts:
            top_labels.append(
                {"label": int(lab), "count": int(cnt), "pct": float(cnt / size) if size else 0.0}
            )
        result[int(c)] = {"size": int(size), "top_labels": top_labels}
    return result


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)
    print(f"Saved: {path}", flush=True)


def save_summary_csv(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fieldnames = ["method", "params", "silhouette", "purity", "notes"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print(f"Saved: {path}", flush=True)


def plot_clusters_2d(X_2d, y_pred, title, save_path):
    plt.figure(figsize=(7, 6))
    plt.scatter(X_2d[:, 0], X_2d[:, 1], c=y_pred, s=12, cmap="tab20")
    plt.title(title)
    plt.xlabel("PCA 1")
    plt.ylabel("PCA 2")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Saved plot: {save_path}", flush=True)


def safe_silhouette(X, labels):
    # silhouette_score needs at least 2 clusters and not all points in one cluster
    unique = set(labels)
    if len(unique) < 2:
        return None
    return silhouette_score(X, labels)


# -----------------------
# Main pipeline
# -----------------------
def main():
    print("=== STEP 4 STARTING ===", flush=True)
    print("Current working directory:", os.getcwd(), flush=True)
    print("Expecting splits at:", os.path.abspath(SPLITS_PATH), flush=True)

    os.makedirs(STEP4_DIR, exist_ok=True)
    save_json({"status": "started"}, os.path.join(STEP4_DIR, "status.json"))

    if not os.path.exists(SPLITS_PATH):
        raise FileNotFoundError(f"Could not find {SPLITS_PATH}. Run your Step1/2 script first to generate it.")

    X_train, X_val, X_test, y_train, y_val, y_test = load_data(SPLITS_PATH)

    # Cluster on TRAIN only (clean + standard)
    X = X_train
    y = y_train
    n_classes = len(np.unique(y))

    print("Train shape:", X.shape, flush=True)
    print("Num subjects (labels):", n_classes, flush=True)

    # -----------------------
    # PCA reduction
    # -----------------------
    # PCA-50 for clustering
    pca_cluster = PCA(n_components=50, random_state=RANDOM_STATE)
    X_pca50 = pca_cluster.fit_transform(X)

    # PCA-2 for visualization
    pca_viz = PCA(n_components=2, random_state=RANDOM_STATE)
    X_pca2 = pca_viz.fit_transform(X)

    save_json(
        {
            "pca50_cumulative_explained_variance": float(np.sum(pca_cluster.explained_variance_ratio_)),
            "pca2_cumulative_explained_variance": float(np.sum(pca_viz.explained_variance_ratio_)),
        },
        os.path.join(STEP4_DIR, "pca_info.json"),
    )

    results = {}
    summary_rows = []

    # -----------------------
    # 1) K-MEANS (tuned) + forced k=20 comparison
    # -----------------------
    print("\n--- KMeans tuning ---", flush=True)

    # Candidates include known true label count + a few alternatives
    k_candidates = sorted(set([n_classes, 15, 25, 30]))
    best_k, best_sil = None, -1.0

    for k in k_candidates:
        km = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        pred = km.fit_predict(X_pca50)
        sil = safe_silhouette(X_pca50, pred)
        print(f"KMeans k={k} silhouette={sil}", flush=True)
        if sil is not None and sil > best_sil:
            best_sil = sil
            best_k = k

    if best_k is None:
        best_k = n_classes  # fallback

    # Best-by-silhouette model
    km_best = KMeans(n_clusters=best_k, n_init=20, random_state=RANDOM_STATE)
    y_km_best = km_best.fit_predict(X_pca50)
    purity_km_best = compute_purity(y, y_km_best)
    comp_km_best = cluster_composition_table(y, y_km_best, top_k=3)

    results["kmeans_best_by_silhouette"] = {
        "chosen_k": int(best_k),
        "silhouette": float(best_sil) if best_sil >= 0 else None,
        "purity": float(purity_km_best),
        "composition_top3": comp_km_best,
    }

    plot_clusters_2d(
        X_pca2,
        y_km_best,
        title=f"K-Means (best k={best_k}) on PCA-50 features",
        save_path=os.path.join(STEP4_DIR, "kmeans_best_pca2.png"),
    )

    summary_rows.append(
        {
            "method": "KMeans",
            "params": f"k={best_k} (best by silhouette)",
            "silhouette": float(best_sil) if best_sil >= 0 else "",
            "purity": float(purity_km_best),
            "notes": "k chosen by silhouette over candidate grid",
        }
    )

    # Forced k=20 model (because dataset has 20 subjects)
    forced_k = n_classes  # should be 20 for UMIST
    km_20 = KMeans(n_clusters=forced_k, n_init=20, random_state=RANDOM_STATE)
    y_km_20 = km_20.fit_predict(X_pca50)
    sil_km_20 = safe_silhouette(X_pca50, y_km_20)
    purity_km_20 = compute_purity(y, y_km_20)
    comp_km_20 = cluster_composition_table(y, y_km_20, top_k=3)

    results["kmeans_forced_k20"] = {
        "chosen_k": int(forced_k),
        "silhouette": float(sil_km_20) if sil_km_20 is not None else None,
        "purity": float(purity_km_20),
        "composition_top3": comp_km_20,
    }

    plot_clusters_2d(
        X_pca2,
        y_km_20,
        title=f"K-Means (forced k={forced_k}) on PCA-50 features",
        save_path=os.path.join(STEP4_DIR, "kmeans_k20_pca2.png"),
    )

    summary_rows.append(
        {
            "method": "KMeans",
            "params": f"k={forced_k} (forced)",
            "silhouette": float(sil_km_20) if sil_km_20 is not None else "",
            "purity": float(purity_km_20),
            "notes": "forced to match known number of subjects",
        }
    )

    # -----------------------
    # 2) HIERARCHICAL (AGGLOMERATIVE)
    # -----------------------
    print("\n--- Agglomerative tuning ---", flush=True)

    linkages = ["ward", "complete", "average"]
    best_link, best_sil_h = None, -1.0

    for link in linkages:
        agg_tmp = AgglomerativeClustering(n_clusters=n_classes, linkage=link)
        y_h = agg_tmp.fit_predict(X_pca50)
        sil = safe_silhouette(X_pca50, y_h)
        print(f"Agglomerative linkage={link} silhouette={sil}", flush=True)
        if sil is not None and sil > best_sil_h:
            best_sil_h = sil
            best_link = link

    if best_link is None:
        best_link = "ward"

    agg = AgglomerativeClustering(n_clusters=n_classes, linkage=best_link)
    y_agg = agg.fit_predict(X_pca50)
    purity_agg = compute_purity(y, y_agg)
    comp_agg = cluster_composition_table(y, y_agg, top_k=3)

    results["hierarchical"] = {
        "n_clusters": int(n_classes),
        "linkage": best_link,
        "silhouette": float(best_sil_h) if best_sil_h >= 0 else None,
        "purity": float(purity_agg),
        "composition_top3": comp_agg,
    }

    plot_clusters_2d(
        X_pca2,
        y_agg,
        title=f"Agglomerative (linkage={best_link}, k={n_classes}) on PCA-50 features",
        save_path=os.path.join(STEP4_DIR, "hierarchical_pca2.png"),
    )

    summary_rows.append(
        {
            "method": "Agglomerative",
            "params": f"linkage={best_link}, k={n_classes}",
            "silhouette": float(best_sil_h) if best_sil_h >= 0 else "",
            "purity": float(purity_agg),
            "notes": "linkage chosen by silhouette among ward/complete/average",
        }
    )

    # -----------------------
    # OPTIONAL: DBSCAN
    # -----------------------
    print("\n--- DBSCAN grid search (optional) ---", flush=True)

    eps_candidates = [2.0, 2.5, 3.0, 3.5]
    ms_candidates = [3, 5, 8]

    best_db = None
    best_db_sil = -1.0

    for eps in eps_candidates:
        for ms in ms_candidates:
            db = DBSCAN(eps=eps, min_samples=ms)
            y_db = db.fit_predict(X_pca50)
            n_clusters = len(set(y_db)) - (1 if -1 in y_db else 0)
            if n_clusters < 2:
                continue
            sil = safe_silhouette(X_pca50, y_db)
            if sil is not None and sil > best_db_sil:
                best_db_sil = sil
                best_db = (eps, ms, y_db)

    if best_db is None:
        results["dbscan"] = {"status": "no_stable_solution_found_in_grid"}
        summary_rows.append(
            {
                "method": "DBSCAN",
                "params": f"eps in {eps_candidates}, min_samples in {ms_candidates}",
                "silhouette": "",
                "purity": "",
                "notes": "no stable multi-cluster solution found in tested grid",
            }
        )
        print("DBSCAN: no stable multi-cluster solution found in tested grid.", flush=True)
    else:
        eps, ms, y_db = best_db
        purity_db = compute_purity(y, y_db)
        comp_db = cluster_composition_table(y, y_db, top_k=3)

        results["dbscan"] = {
            "eps": float(eps),
            "min_samples": int(ms),
            "silhouette": float(best_db_sil),
            "purity": float(purity_db),
            "n_clusters_excluding_noise": int(len(set(y_db)) - (1 if -1 in y_db else 0)),
            "noise_points": int(np.sum(y_db == -1)),
            "composition_top3": comp_db,
        }

        plot_clusters_2d(
            X_pca2,
            y_db,
            title=f"DBSCAN (eps={eps}, min_samples={ms}) on PCA-50 features (-1=noise)",
            save_path=os.path.join(STEP4_DIR, "dbscan_pca2.png"),
        )

        summary_rows.append(
            {
                "method": "DBSCAN",
                "params": f"eps={eps}, min_samples={ms}",
                "silhouette": float(best_db_sil),
                "purity": float(purity_db),
                "notes": f"clusters (excl noise)={len(set(y_db)) - (1 if -1 in y_db else 0)}, noise={int(np.sum(y_db == -1))}",
            }
        )

    # -----------------------
    # Save results
    # -----------------------
    save_json(results, os.path.join(STEP4_DIR, "clustering_results.json"))
    save_summary_csv(summary_rows, os.path.join(STEP4_DIR, "summary.csv"))

    print("\n=== STEP 4 COMPLETED ===", flush=True)
    print("Outputs saved to:", os.path.abspath(STEP4_DIR), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\n!!! STEP 4 FAILED WITH EXCEPTION !!!", flush=True)
        traceback.print_exc()
        raise

