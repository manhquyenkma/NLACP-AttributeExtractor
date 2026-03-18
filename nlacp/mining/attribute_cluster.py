import json
import os
import numpy as np
from collections import Counter   # FIX 6: cần cho _compute_cluster_short_name
from sklearn.cluster import DBSCAN
import spacy
from sklearn.neighbors import NearestNeighbors

# ===================================================================
# attribute_cluster.py  (nlacp/mining/)
# Module 2: Attribute Clustering (Alohaly et al. 2019)
#   GloVe Vectors + auto-tune eps + DBSCAN min_samples=2
# ===================================================================

# nlacp/mining/ → nlacp/ → project root
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH  = os.path.join(BASE_DIR, "outputs", "policies", "policy_dataset.json")
OUTPUT_PATH   = os.path.join(BASE_DIR, "outputs", "clusters",  "attribute_clusters.json")

# Dùng model lớn có GloVe vectors
try:
    nlp_vec = spacy.load("en_core_web_md")
    print("[INFO] Using en_core_web_md (300d GloVe vectors)")
except OSError:
    print("[WARNING] en_core_web_md not found. Falling back to en_core_web_sm (WARNING: poor embeddings!)")
    nlp_vec = spacy.load("en_core_web_sm")


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_attribute_names(dataset):
    attributes = []
    for policy in dataset.get("policies", []):
        for attr in policy.get("attributes", []):
            # FIX 6a: fallback về value nếu không có name (env attrs)
            name = attr.get("name") or attr.get("value") or ""
            if name:
                attributes.append(name.lower())
    return list(set(attributes))


def vectorize_attributes(attributes):
    vectors = []
    for attr in attributes:
        vec = nlp_vec(attr).vector   # 300d GloVe-like
        vectors.append(vec)
    return np.array(vectors)


# ── FIX 2: Tính eps tự động bằng k-distance graph ──
def compute_auto_eps(X, min_pts=2):
    """
    Theo Alohaly 2019: vẽ k-distance graph, lấy trung bình khoảng cách
    đến k nearest neighbor để xác định eps phù hợp với dataset.
    """
    if X.shape[0] < min_pts:
        return 0.5   # fallback nếu quá ít điểm

    nbrs = NearestNeighbors(n_neighbors=min_pts, metric="euclidean").fit(X)
    distances, _ = nbrs.kneighbors(X)
    eps = float(np.mean(distances[:, -1]))
    print(f"Auto-computed eps = {eps:.4f}")
    return eps


def run_dbscan(X):
    eps    = compute_auto_eps(X, min_pts=2)   # auto-tune
    model  = DBSCAN(eps=eps, min_samples=2, metric="euclidean")   # metric euclidean cho GloVe
    labels = model.fit_predict(X)
    return labels


def _compute_cluster_short_name(attr_list):
    """FIX 6b: Tính short_name đại diện cho cluster — token phổ biến nhất."""
    stop = {"a", "an", "the", "of", "in", "at", "on", "by", "to", "for"}
    all_tokens = []
    for attr in attr_list:
        all_tokens.extend(str(attr).lower().split())
    filtered = [t for t in all_tokens if t not in stop and len(t) > 2]
    if not filtered:
        return attr_list[0] if attr_list else "cluster"
    return Counter(filtered).most_common(1)[0][0]


def build_clusters(attributes, labels):
    clusters = {}
    for attr, label in zip(attributes, labels):
        clusters.setdefault(label, []).append(attr)

    result = {"clusters": []}
    for label, attrs in clusters.items():
        result["clusters"].append({
            "cluster_id": int(label),
            "short_name": _compute_cluster_short_name(attrs),   # FIX 6b: field mới
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