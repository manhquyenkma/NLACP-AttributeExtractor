import json
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

# ===================================================================
# attribute_cluster.py
# Module 2: Attribute Clustering (Alohaly et al. 2019)
#
# FIX 2 (Tuần này):   TF-IDF + auto-tune eps (k-distance) + min_samples=2
# FIX 3 (Tuần sau):   Thay TF-IDF bằng GloVe (en_core_web_md)
# ===================================================================

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH  = os.path.join(BASE_DIR, "dataset", "policy_dataset.json")
OUTPUT_PATH   = os.path.join(BASE_DIR, "dataset", "attribute_clusters.json")


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_attribute_names(dataset):
    attributes = []
    for policy in dataset["policies"]:
        for attr in policy["attributes"]:
            attributes.append(attr["name"].lower())
    return list(set(attributes))


def vectorize_attributes(attributes):
    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(attributes)
    return X


# ── FIX 2: Tính eps tự động bằng k-distance graph ──
def compute_auto_eps(X, min_pts=2):
    """
    Theo Alohaly 2019: vẽ k-distance graph, lấy trung bình khoảng cách
    đến k nearest neighbor để xác định eps phù hợp với dataset.
    """
    if X.shape[0] < min_pts:
        return 0.5   # fallback nếu quá ít điểm

    nbrs = NearestNeighbors(n_neighbors=min_pts, metric="cosine").fit(X)
    distances, _ = nbrs.kneighbors(X)
    eps = float(np.mean(distances[:, -1]))
    print(f"Auto-computed eps = {eps:.4f}")
    return eps


def run_dbscan(X):
    eps    = compute_auto_eps(X, min_pts=2)   # auto-tune
    model  = DBSCAN(eps=eps, min_samples=2, metric="cosine")   # min_samples=2 theo bài báo
    labels = model.fit_predict(X)
    return labels


def build_clusters(attributes, labels):
    clusters = {}
    for attr, label in zip(attributes, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(attr)

    result = {"clusters": []}
    for label, attrs in clusters.items():
        result["clusters"].append({
            "cluster_id": int(label),
            "attributes": attrs
        })
    return result


def save_clusters(data):
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def main():
    print("\n" + "="*50)
    print("  Module 2: Attribute Clustering")
    print("  TF-IDF + Auto-tune eps + DBSCAN (min_samples=2)")
    print("="*50 + "\n")

    dataset    = load_dataset()
    attributes = extract_attribute_names(dataset)

    print("Attributes found:", attributes)

    if not attributes:
        print("Khong co attributes nao de cluster.")
        return

    X      = vectorize_attributes(attributes)
    labels = run_dbscan(X)

    clusters = build_clusters(attributes, labels)
    save_clusters(clusters)

    print(f"\nClusters ({len(clusters['clusters'])} total):")
    for c in clusters["clusters"]:
        label = "OUTLIER" if c["cluster_id"] == -1 else f"Cluster {c['cluster_id']}"
        print(f"  {label}: {c['attributes']}")

    print("\nAttribute clusters saved.")


if __name__ == "__main__":
    main()