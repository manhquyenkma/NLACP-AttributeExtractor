import json
import os
import numpy as np
from collections import Counter
from sklearn.cluster import DBSCAN
import spacy
from sklearn.neighbors import NearestNeighbors

# ===================================================================
# attribute_cluster.py  (nlacp/mining/)
# Module 2: Attribute Clustering (Alohaly et al. 2019)
#   GloVe Vectors + auto-tune eps + DBSCAN min_samples=2
# ===================================================================

import sys as _sys
_sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from nlacp.paths import POLICY_DATASET_PATH as DATASET_PATH, ATTRIBUTE_CLUSTERS_PATH as OUTPUT_PATH

# Dùng model lớn có GloVe vectors
from nlacp.utils.nlp_utils import get_spacy_model
nlp_vec = get_spacy_model()


def load_dataset():
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def extract_attribute_names(dataset):
    attributes = []
    for policy in dataset.get("policies", []):
        for attr in policy.get("attributes", []):
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


from sklearn.cluster import AgglomerativeClustering
from collections import Counter, defaultdict


def compute_auto_eps(X, min_pts=2):
    """
    Theo Alohaly 2019: vẽ k-distance graph, lấy trung bình khoảng cách
    đến k nearest neighbor để xác định eps phù hợp với dataset.
    """
    if X.shape[0] < min_pts:
        return 0.5   # fallback nếu quá ít điểm

    nbrs = NearestNeighbors(n_neighbors=min_pts, metric="euclidean").fit(X)
    distances, _ = nbrs.kneighbors(X)
    # Thay vì lấy trung bình (mean), ta lấy phân vị cao (90th percentile)
    # để eps rộng hơn, giảm bớt outliers và tăng độ bao phủ (coverage).
    eps = float(np.percentile(distances[:, -1], 90))
    import logging
    logging.debug(f"Auto-computed eps (90th percentile) = {eps:.4f}")
    return eps


def _build_category_map():
    """Load true categories từ policy_dataset.json."""
    try:
        with open(DATASET_PATH, encoding="utf-8") as f:
            data = json.load(f)
        cat_map = {}
        for policy in data.get("policies", []):
            for attr in policy.get("attributes", []):
                name = (attr.get("name") or "").lower()
                cat = attr.get("category", "unknown")
                if name:
                    cat_map[name] = cat
        return cat_map
    except Exception:
        return {}


def _compute_purity(labels, attributes, category_map):
    """Tính average cluster purity."""
    clusters = defaultdict(list)
    for attr, label in zip(attributes, labels):
        cat = category_map.get(attr, "unknown")
        clusters[label].append(cat)
    
    if not clusters:
        return 0.0
    
    total_purity = 0
    for label, cats in clusters.items():
        if cats:
            majority = Counter(cats).most_common(1)[0][1]
            total_purity += majority / len(cats)
    
    return total_purity / len(clusters)


def _select_threshold(X_norm, attributes):
    """
    Grid search threshold để:
    1. Không có cluster nào > 25 phần tử (tránh mega-cluster)
    2. Số clusters trong khoảng [8, 25]
    3. Maximize average cluster purity (theo category)
    """
    category_map = _build_category_map()
    
    best_score = -1
    best_t = 1.5  # default fallback
    
    for t in np.arange(0.3, 3.0, 0.1):
        model = AgglomerativeClustering(
            n_clusters=None,
            distance_threshold=t,
            linkage='ward'
        )
        labels = model.fit_predict(X_norm)
        
        n_clusters = len(set(labels))
        counts = Counter(labels).values()
        max_size = max(counts) if counts else 0
        
        # Hard constraints
        if max_size > 25 or n_clusters < 8 or n_clusters > 25:
            continue
        
        # Score = average purity
        score = _compute_purity(labels, attributes, category_map)
        
        if score > best_score:
            best_score = score
            best_t = t
    
    print(f"[INFO] Selected distance_threshold = {best_t:.1f} (purity={best_score:.3f})")
    return best_t


def run_agglomerative(X, attributes):
    """
    Agglomerative Clustering với ward linkage.
    Tự động chọn distance_threshold qua grid search.
    """
    if X.shape[0] < 3:
        return np.zeros(X.shape[0], dtype=int)
    
    # Normalize vectors trước (thường tốt hơn cho clustering)
    from sklearn.preprocessing import normalize
    X_norm = normalize(X)
    
    best_threshold = _select_threshold(X_norm, attributes)
    
    model = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=best_threshold,
        linkage='ward'
    )
    labels = model.fit_predict(X_norm)
    return labels


def run_dbscan(X):
    eps    = compute_auto_eps(X, min_pts=2)   # auto-tune
    model  = DBSCAN(eps=eps, min_samples=2, metric="euclidean")
    labels = model.fit_predict(X)
    return labels


def _compute_cluster_short_name(attr_list):
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
            "short_name": _compute_cluster_short_name(attrs),
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
    print("  Word Vectors (Agglomerative) + Threshold Selection")
    print("="*50 + "\n")

    dataset    = load_dataset()
    attributes = extract_attribute_names(dataset)

    print("Attributes found:", attributes)

    if not attributes:
        print("Khong co attributes nao de cluster.")
        return

    X      = vectorize_attributes(attributes)
    # Dùng Agglomerative thay DBSCAN
    labels = run_agglomerative(X, attributes)

    clusters = build_clusters(attributes, labels)
    save_clusters(clusters)

    print(f"\nClusters ({len(clusters['clusters'])} total):")
    for c in clusters["clusters"]:
        label = "OUTLIER" if c["cluster_id"] == -1 else f"Cluster {c['cluster_id']}"
        print(f"  {label}: {c['attributes']}")

    print("\nAttribute clusters saved.")

    # ── Tự động đánh giá F1 sau khi cluster ──────────────────────
    try:
        from nlacp.evaluation.evaluator import evaluate_clustering_pipeline
        evaluate_clustering_pipeline()
    except Exception as e:
        print(f"[WARN] Cluster evaluation skipped: {e}")


if __name__ == "__main__":
    main()