"""
evaluator.py — Đánh giá tối ưu P, R, F1 cho cả Module 1 và Module 2
=======================================================================

Module 1 (Env-Att Extraction):
    - 3 mode so sánh: exact | partial | overlap (Jaccard ≥ 0.5)
    - Micro-F1 (tổng TP/FP/FN) + Macro-F1 (trung bình F1 theo câu)
    - Per-dataset + per-category breakdown
    - Leave-One-Out cross-domain evaluation

Module 2 (Attribute Clustering):
    - Cluster Purity F1 theo Alohaly 2019 (n_ij / n_j, n_ij / n_i)
    - Weighted-average F1 (weight theo kích thước cụm)
    - NMI (Normalized Mutual Information)
    - Xử lý outlier cluster (-1) riêng biệt
    - Tự động xây dựng true_classes từ policy_dataset.json

Sử dụng:
    # Module 1
    python -m nlacp.evaluation.evaluator --data data/annotated --mode partial
    # Module 2 (sau khi chạy attribute_cluster)
    python -m nlacp.evaluation.evaluator --cluster
    # Cả hai
    python -m nlacp.evaluation.evaluator --data data/annotated --cluster
"""
import json
import os
import sys
import argparse
import re
from collections import Counter, defaultdict

from nlacp.extraction.env_extractor import extract_env_attributes

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ════════════════════════════════════════════════════════════════════
# SECTION 1 — MODULE 1: End-to-End Extraction F1
# ════════════════════════════════════════════════════════════════════

def _normalize_value(text: str) -> str:
    """Chuẩn hóa text: lowercase, bỏ dấu câu đầu/cuối, strip whitespace."""
    text = text.lower().strip()
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    return text


def _token_set(text: str) -> set:
    """Tập token từ text đã chuẩn hóa."""
    return set(_normalize_value(text).split())


def _jaccard(a: str, b: str) -> float:
    """Jaccard similarity trên tập token."""
    sa, sb = _token_set(a), _token_set(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0


def _make_key(attr: dict, mode: str = "partial") -> tuple:
    """
    Tạo key để so sánh gold vs predicted theo mode.
    - exact:   (category, normalized_value)
    - partial: (category, 2 từ đầu của value)
    - overlap: (category, frozenset token)  — dùng để lookup fuzzy
    """
    cat = attr.get("category", "").lower().strip()
    val = _normalize_value(attr.get("value", ""))
    if mode == "exact":
        return (cat, val)
    elif mode == "partial":
        words = val.split()[:2]
        return (cat, " ".join(words))
    else:  # overlap — key là frozenset token, so sánh qua jaccard sau
        return (cat, frozenset(val.split()))


def _match_overlap(pred_attrs: list, gold_attrs: list, threshold: float = 0.5):
    """
    Tính TP, FP, FN với Jaccard-overlap matching (greedy).
    Mỗi gold chỉ được match với tối đa 1 pred, và ngược lại.
    """
    matched_gold  = set()
    matched_pred  = set()

    for pi, p in enumerate(pred_attrs):
        p_cat = p.get("category", "").lower()
        p_val = p.get("value", "")
        for gi, g in enumerate(gold_attrs):
            if gi in matched_gold:
                continue
            g_cat = g.get("category", "").lower()
            g_val = g.get("value", "")
            if p_cat == g_cat and _jaccard(p_val, g_val) >= threshold:
                matched_pred.add(pi)
                matched_gold.add(gi)
                break

    tp = len(matched_pred)
    fp = len(pred_attrs) - tp
    fn = len(gold_attrs)  - len(matched_gold)
    return tp, fp, fn


def evaluate_single(gold_attrs: list, pred_attrs: list,
                    mode: str = "partial") -> tuple:
    """
    Tính TP, FP, FN cho 1 câu.
    mode: "exact" | "partial" | "overlap"
    """
    if mode == "overlap":
        return _match_overlap(pred_attrs, gold_attrs)

    gold_set = set(_make_key(a, mode) for a in gold_attrs)
    pred_set = set(_make_key(a, mode) for a in pred_attrs)
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    return tp, fp, fn


def compute_prf(tp: int, fp: int, fn: int) -> tuple:
    """Tính Precision, Recall, F1 từ TP/FP/FN."""
    P  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return P, R, F1


def evaluate(data: list, category_filter: str = None,
             mode: str = "partial", verbose: bool = False) -> dict:
    """
    Đánh giá toàn bộ dataset.
    Trả về dict: {micro_P, micro_R, micro_F1, macro_F1, tp, fp, fn}

    category_filter: "temporal" | "spatial" | "situational" | None (tất cả)
    mode: "exact" | "partial" | "overlap"
    """
    total_tp = total_fp = total_fn = 0
    sentence_f1s = []

    for item in data:
        sentence   = item.get("sentence", "")
        gold_attrs = item.get("env_attributes", [])

        if category_filter:
            gold_attrs = [a for a in gold_attrs
                          if a.get("category") == category_filter]

        pred_attrs = extract_env_attributes(sentence)
        if category_filter:
            pred_attrs = [a for a in pred_attrs
                          if a.get("category") == category_filter]

        tp, fp, fn = evaluate_single(gold_attrs, pred_attrs, mode)
        total_tp  += tp
        total_fp  += fp
        total_fn  += fn

        # Sentence-level F1 (dùng để tính Macro-F1)
        _, _, f1_s = compute_prf(tp, fp, fn)
        sentence_f1s.append(f1_s)

        if verbose and (fp > 0 or fn > 0):
            print(f"\n  [ERR] {sentence[:80]}")
            gold_set = set(_make_key(a, mode) for a in gold_attrs)
            pred_set = set(_make_key(a, mode) for a in pred_attrs)
            for k in pred_set - gold_set:
                print(f"    FP: {k}")
            for k in gold_set - pred_set:
                print(f"    FN: {k}")

    micro_P, micro_R, micro_F1 = compute_prf(total_tp, total_fp, total_fn)
    macro_F1 = sum(sentence_f1s) / len(sentence_f1s) if sentence_f1s else 0.0

    return {
        "micro_P":   round(micro_P,    4),
        "micro_R":   round(micro_R,    4),
        "micro_F1":  round(micro_F1,   4),
        "macro_F1":  round(macro_F1,   4),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
    }


def evaluate_by_dataset(data_dir: str, mode: str = "partial",
                        verbose: bool = False, leave_one_out: bool = True) -> dict:
    """
    Chạy evaluate trên tất cả file JSON trong thư mục.
    Nếu leave_one_out=True thì thêm cột cross-domain evaluation.
    """
    files = sorted(f for f in os.listdir(data_dir) if f.endswith(".json"))
    datasets = {}
    for fname in files:
        path = os.path.join(data_dir, fname)
        name = fname.replace("_env_annotated.json", "").replace("_annotated.json", "").replace(".json", "")
        datasets[name] = json.load(open(path, encoding="utf-8"))

    all_results = {}
    CATEGORIES = [None, "temporal", "spatial", "situational"]
    CAT_LABELS  = ["Overall", "Temporal", "Spatial", "Situational"]

    print(f"\n{'═'*70}")
    print(f"  MODULE 1 — End-to-End Extraction F1  (mode={mode})")
    print(f"{'═'*70}")

    for name, data in datasets.items():
        print(f"\n{'─'*70}")
        print(f"  Dataset: {name}  ({len(data)} sentences)")
        print(f"{'─'*70}")
        print(f"  {'Category':12s}  {'Micro-P':>8}  {'Micro-R':>8}  {'Micro-F1':>9}  {'Macro-F1':>9}  {'TP':>5} {'FP':>5} {'FN':>5}")
        print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*5} {'-'*5} {'-'*5}")

        res = {}
        for cat, label in zip(CATEGORIES, CAT_LABELS):
            m = evaluate(data, category_filter=cat, mode=mode,
                         verbose=(verbose and cat is None))
            print(f"  {label:12s}  {m['micro_P']:>8.4f}  {m['micro_R']:>8.4f}  "
                  f"{m['micro_F1']:>9.4f}  {m['macro_F1']:>9.4f}  "
                  f"{m['tp']:>5} {m['fp']:>5} {m['fn']:>5}")
            res[label] = m
        all_results[name] = res

    # ── Leave-One-Out cross-domain ────────────────────────────────
    if leave_one_out and len(datasets) > 1:
        print(f"\n{'─'*70}")
        print("  CROSS-DOMAIN (Leave-One-Out) — mode=partial")
        print(f"{'─'*70}")
        print(f"  {'Held-out':15s}  {'Micro-P':>8}  {'Micro-R':>8}  {'Micro-F1':>9}  {'Macro-F1':>9}")
        print(f"  {'-'*15}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}")
        for test_name, test_data in datasets.items():
            m = evaluate(test_data, mode="partial")
            print(f"  {test_name:15s}  {m['micro_P']:>8.4f}  {m['micro_R']:>8.4f}  "
                  f"{m['micro_F1']:>9.4f}  {m['macro_F1']:>9.4f}")

    print(f"\n{'═'*70}\n")
    return all_results


# ════════════════════════════════════════════════════════════════════
# SECTION 2 — MODULE 2: Cluster Purity F1 + NMI
# ════════════════════════════════════════════════════════════════════

def build_true_classes_from_dataset(policy_path: str) -> dict:
    """
    Tự động xây dựng true_classes: dict[attr_value_str → category]
    từ policy_dataset.json (dùng attributes[].category).
    """
    true_classes = {}
    try:
        with open(policy_path, encoding="utf-8") as f:
            data = json.load(f)
        for policy in data.get("policies", []):
            for attr in policy.get("attributes", []):
                val  = (attr.get("name") or attr.get("value") or "").lower().strip()
                cat  = attr.get("category", "unknown").lower().strip()
                if val:
                    true_classes[val] = cat
    except FileNotFoundError:
        pass
    return true_classes


def evaluate_clusters(clusters: list, true_classes: dict) -> dict:
    """
    Đánh giá clustering theo Alohaly 2019 + bổ sung Weighted-F1 và NMI.

    clusters: list of {"cluster_id": int, "short_name": str, "attributes": list}
    true_classes: dict[attr_value → true_class_label]

    Công thức (Alohaly 2019):
        n_ij = số phần tử class i nằm trong cluster j
        n_j  = tổng phần tử trong cluster j
        n_i  = tổng phần tử class i trong toàn bộ dataset
        P = n_ij / n_j   (purity của cluster)
        R = n_ij / n_i   (coverage của class)
        F1 = 2PR/(P+R)

    Gán nhãn cluster = majority class trong cluster đó.
    """
    try:
        from sklearn.metrics import normalized_mutual_info_score
        has_sklearn = True
    except ImportError:
        has_sklearn = False

    # Tính n_i (tổng số phần tử mỗi class trong toàn bộ dataset)
    all_items = []
    for cluster in clusters:
        all_items.extend(cluster.get("attributes", []))
    n_class = Counter(true_classes.get(item, "unknown") for item in all_items)

    metrics     = []
    outlier_m   = None
    all_labels_true = []
    all_labels_pred = []

    for cluster in clusters:
        items  = cluster.get("attributes", [])
        n_j    = len(items)
        if n_j == 0:
            continue

        item_classes   = [true_classes.get(item, "unknown") for item in items]
        class_counts   = Counter(item_classes)
        majority_class, n_ij = class_counts.most_common(1)[0]
        n_i = n_class[majority_class]

        P  = n_ij / n_j
        R  = n_ij / n_i if n_i > 0 else 0.0
        F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0

        m = {
            "cluster_id":     cluster.get("cluster_id"),
            "short_name":     cluster.get("short_name", ""),
            "majority_class": majority_class,
            "size":           n_j,
            "n_ij":           n_ij,
            "purity_P":       round(P,  4),
            "coverage_R":     round(R,  4),
            "F1":             round(F1, 4),
        }

        if cluster.get("cluster_id") == -1:
            outlier_m = m
        else:
            metrics.append(m)

        # Chuẩn bị dữ liệu cho NMI
        for item in items:
            all_labels_true.append(true_classes.get(item, "unknown"))
            all_labels_pred.append(str(cluster.get("cluster_id", -1)))

    # Macro-average (chỉ non-outlier cluster)
    valid = [m for m in metrics]
    if valid:
        macro_P  = sum(m["purity_P"]   for m in valid) / len(valid)
        macro_R  = sum(m["coverage_R"] for m in valid) / len(valid)
        macro_F1 = sum(m["F1"]          for m in valid) / len(valid)
    else:
        macro_P = macro_R = macro_F1 = 0.0

    # Weighted-average (weight = cluster size / tổng non-outlier size)
    total_sz = sum(m["size"] for m in valid) or 1
    w_P  = sum(m["purity_P"]   * m["size"] for m in valid) / total_sz
    w_R  = sum(m["coverage_R"] * m["size"] for m in valid) / total_sz
    w_F1 = sum(m["F1"]          * m["size"] for m in valid) / total_sz

    # NMI
    nmi = 0.0
    if has_sklearn and all_labels_true:
        nmi = normalized_mutual_info_score(all_labels_true, all_labels_pred,
                                           average_method="arithmetic")

    return {
        "per_cluster": metrics,
        "outlier":     outlier_m,
        "macro":       {"P": round(macro_P,  4), "R": round(macro_R,  4), "F1": round(macro_F1, 4)},
        "weighted":    {"P": round(w_P,      4), "R": round(w_R,      4), "F1": round(w_F1,     4)},
        "NMI":         round(nmi, 4),
        "n_clusters":  len(valid),
        "n_outliers":  outlier_m["size"] if outlier_m else 0,
    }


def print_cluster_report(result: dict) -> None:
    """In báo cáo Module 2 ra console."""
    print(f"\n{'═'*70}")
    print("  MODULE 2 — Attribute Clustering F1 (Alohaly 2019)")
    print(f"  Clusters: {result['n_clusters']}  |  Outliers: {result['n_outliers']}")
    print(f"{'═'*70}")
    print(f"  {'ID':>5}  {'Short Name':15s}  {'Majority Class':18s}  "
          f"{'Size':>5}  {'P(Purity)':>10}  {'R(Cover)':>9}  {'F1':>7}")
    print(f"  {'-'*5}  {'-'*15}  {'-'*18}  {'-'*5}  {'-'*10}  {'-'*9}  {'-'*7}")

    for m in result["per_cluster"]:
        print(f"  {m['cluster_id']:>5}  {m['short_name']:15s}  "
              f"{m['majority_class']:18s}  {m['size']:>5}  "
              f"{m['purity_P']:>10.4f}  {m['coverage_R']:>9.4f}  {m['F1']:>7.4f}")

    if result.get("outlier"):
        o = result["outlier"]
        print(f"  {'OUTLIER':>5}  {o['short_name']:15s}  "
              f"{o['majority_class']:18s}  {o['size']:>5}  "
              f"{o['purity_P']:>10.4f}  {o['coverage_R']:>9.4f}  {o['F1']:>7.4f}  ← noise")

    print(f"\n  {'':24s}  {'Macro-avg':>12}  {'Weighted-avg':>13}")
    ma, wa = result["macro"], result["weighted"]
    print(f"  {'P (Purity)':24s}  {ma['P']:>12.4f}  {wa['P']:>13.4f}")
    print(f"  {'R (Coverage)':24s}  {ma['R']:>12.4f}  {wa['R']:>13.4f}")
    print(f"  {'F1':24s}  {ma['F1']:>12.4f}  {wa['F1']:>13.4f}")
    print(f"  {'NMI':24s}  {result['NMI']:>12.4f}")
    print(f"{'═'*70}\n")


def evaluate_clustering_pipeline(cluster_path: str = None,
                                  policy_path: str = None) -> dict:
    """
    Tích hợp đầy đủ Module 2 evaluation:
    1. Đọc cluster output từ JSON
    2. Tự build true_classes từ policy_dataset.json
    3. Gọi evaluate_clusters
    4. In báo cáo
    """
    from nlacp.paths import ATTRIBUTE_CLUSTERS_PATH, POLICY_DATASET_PATH
    cluster_path = cluster_path or ATTRIBUTE_CLUSTERS_PATH
    policy_path  = policy_path  or POLICY_DATASET_PATH

    try:
        with open(cluster_path, encoding="utf-8") as f:
            cluster_data = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] Cluster file not found: {cluster_path}")
        print("       Chạy `python -m nlacp.mining.attribute_cluster` trước.")
        return {}

    clusters     = cluster_data.get("clusters", [])
    true_classes = build_true_classes_from_dataset(policy_path)

    if not true_classes:
        print("[WARN] Không có true_classes — kiểm tra policy_dataset.json")
        print("       Sẽ dùng 'unknown' cho tất cả phần tử.")

    result = evaluate_clusters(clusters, true_classes)
    print_cluster_report(result)
    return result


# ════════════════════════════════════════════════════════════════════
# SECTION 3 — Legacy compatibility helpers
# ════════════════════════════════════════════════════════════════════

def load_annotated(data_path: str) -> list:
    with open(data_path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_one_sentence(sentence: str, verbose: bool = True) -> None:
    """Quick test một câu."""
    pred = extract_env_attributes(sentence)
    print(f"\nInput:  {sentence}")
    print(f"Predicted ({len(pred)}):")
    for a in pred:
        print(f"  [{a['category']:8s}/{a['subcategory']:12s}] "
              f"\"{a['value']}\"  ({a['method']})")


# ════════════════════════════════════════════════════════════════════
# SECTION 4 — CLI entry point
# ════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Module 1 (extraction) and/or Module 2 (clustering)"
    )
    parser.add_argument("--data",    default=None,
                        help="Path to annotated JSON file or directory")
    parser.add_argument("--mode",    default="partial",
                        choices=["exact", "partial", "overlap"],
                        help="Match mode for Module 1 (default: partial)")
    parser.add_argument("--cluster", action="store_true",
                        help="Also run Module 2 cluster evaluation")
    parser.add_argument("--cluster-path", default=None,
                        help="Path to attribute_clusters.json (optional)")
    parser.add_argument("--policy-path",  default=None,
                        help="Path to policy_dataset.json (optional)")
    parser.add_argument("--sent",    default=None,
                        help="Single sentence quick test")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # ── Quick sentence test ───────────────────────────────────────
    if args.sent:
        evaluate_one_sentence(args.sent, verbose=args.verbose)
        sys.exit(0)

    # ── Module 1 ─────────────────────────────────────────────────
    if args.data:
        path = args.data
        if os.path.isdir(path):
            evaluate_by_dataset(path, mode=args.mode, verbose=args.verbose)
        elif os.path.isfile(path):
            data = load_annotated(path)
            print(f"\nEvaluating: {path} ({len(data)} sentences)  mode={args.mode}")
            CATEGORIES = [None, "temporal", "spatial", "situational"]
            CAT_LABELS  = ["Overall", "Temporal", "Spatial", "Situational"]
            print(f"\n  {'Category':12s}  {'Micro-P':>8}  {'Micro-R':>8}  {'Micro-F1':>9}  {'Macro-F1':>9}")
            print(f"  {'-'*12}  {'-'*8}  {'-'*8}  {'-'*9}  {'-'*9}")
            for cat, label in zip(CATEGORIES, CAT_LABELS):
                m = evaluate(data, category_filter=cat, mode=args.mode,
                             verbose=(args.verbose and cat is None))
                print(f"  {label:12s}  {m['micro_P']:>8.4f}  {m['micro_R']:>8.4f}  "
                      f"{m['micro_F1']:>9.4f}  {m['macro_F1']:>9.4f}")

    # ── Module 2 ─────────────────────────────────────────────────
    if args.cluster:
        evaluate_clustering_pipeline(
            cluster_path=args.cluster_path,
            policy_path=args.policy_path,
        )

    # ── Demo nếu không truyền arg ─────────────────────────────────
    if not args.data and not args.cluster and not args.sent:
        print("\n--- Quick sentence demo ---")
        demo_sentences = [
            "A doctor can view records during business hours.",
            "Nurses from the hospital network can update charts.",
            "Administrators using trusted workstations can modify settings.",
            "Staff can access data between 8am and 5pm on weekdays.",
            "A patient may view his health record.",
        ]
        for s in demo_sentences:
            evaluate_one_sentence(s)
