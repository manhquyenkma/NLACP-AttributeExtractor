"""
attribute_cluster_glove.py
FIX 3 — GloVe Embeddings (Tuần sau)

Thay thế TF-IDF bằng spaCy word vectors (300d GloVe-like từ en_core_web_md).
Khi sẵn sàng dùng FIX 3:
  1. pip install spacy
  2. python -m spacy download en_core_web_md
  3. Copy file này thành mining/attribute_cluster.py

Tại sao GloVe tốt hơn TF-IDF:
  TF-IDF: "senior" ≠ "junior" (2 sparse vectors hoàn toàn khác)
  GloVe:  "senior" ≈ "junior"  (gần nhau trong không gian 300d)
  → DBSCAN với GloVe sẽ gom "senior", "junior" vào CÙNG cluster "rank"
"""
import json
import os
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.neighbors import NearestNeighbors
import spacy

BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "policy_dataset.json")
OUTPUT_PATH  = os.path.join(BASE_DIR, "dataset", "attribute_clusters.json")

# Dùng model lớn có GloVe vectors
try:
    nlp_vec = spacy.load("en_core_web_md")
    print("[INFO] Using en_core_web_md (300d GloVe vectors)")
except OSError:
    raise SystemExit(
        "[ERROR] en_core_web_md not found.\n"
        "        Run: python -m spacy download en_core_web_md"
    )


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_attribute_names(dataset):
    attributes = []
    for policy in dataset["policies"]:
        for attr in policy["attributes"]:
            attributes.append(attr["name"].lower())
    return list(set(attributes))


# ── FIX 3: GloVe Embeddings ──
def vectorize_attributes(attributes):
    """
    Dùng spaCy word vectors (300d) thay vì TF-IDF.
    Nguyên lý compositionality: multi-word = tổng vector các từ.
    """
    vectors = []
    for attr in attributes:
        vec = nlp_vec(attr).vector   # 300d GloVe-like
        vectors.append(vec)
    return np.array(vectors)


def compute_auto_eps(X, min_pts=2):
    """Auto-tune eps bằng k-distance graph."""
    if X.shape[0] < min_pts:
        return 0.5
    nbrs = NearestNeighbors(n_neighbors=min_pts, metric="euclidean").fit(X)
    distances, _ = nbrs.kneighbors(X)
    eps = float(np.mean(distances[:, -1]))
    print(f"Auto-computed eps = {eps:.4f}")
    return eps


def run_dbscan(X):
    eps   = compute_auto_eps(X, min_pts=2)
    # Dùng euclidean thay cosine khi vectors là dense (GloVe)
    model = DBSCAN(eps=eps, min_samples=2, metric="euclidean")
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
    print("\n" + "="*55)
    print("  Module 2: Attribute Clustering (FIX 3 — GloVe)")
    print("="*55 + "\n")

    dataset    = load_dataset()
    attributes = extract_attribute_names(dataset)
    print("Attributes:", attributes)

    if not attributes:
        print("No attributes found.")
        return

    X      = vectorize_attributes(attributes)
    labels = run_dbscan(X)

    clusters = build_clusters(attributes, labels)
    save_clusters(clusters)

    print(f"\nClusters ({len(clusters['clusters'])} total):")
    for c in clusters["clusters"]:
        label = "OUTLIER" if c["cluster_id"] == -1 else f"Cluster {c['cluster_id']}"
        print(f"  {label}: {c['attributes']}")

    print("\nDone — attribute_clusters.json saved.")


if __name__ == "__main__":
    main()
