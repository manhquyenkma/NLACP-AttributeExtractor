#!/usr/bin/env python
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.paths import POLICY_DATASET_PATH, ATTRIBUTE_CLUSTERS_PATH
from nlacp.evaluation.evaluator import evaluate_clusters

def main():
    print("="*60)
    print("  Module 2 Evaluation - DBSCAN Attribute Clustering")
    print("="*60)

    if not os.path.exists(POLICY_DATASET_PATH) or not os.path.exists(ATTRIBUTE_CLUSTERS_PATH):
        print("[ERROR] Missing dataset or cluster output JSON files.")
        print("Please run nlacp/mining/attribute_cluster.py first.")
        return

    # 1. Load dataset to determine true class mappings
    with open(POLICY_DATASET_PATH, "r", encoding="utf-8") as f:
        policy_data = json.load(f)

    true_classes = {}
    for policy in policy_data.get("policies", []):
        for attr in policy.get("attributes", []):
            name = (attr.get("name") or attr.get("value") or "").lower()
            cat = attr.get("category", "unclassified")
            if name and cat != "unclassified":
                true_classes[name] = cat
        # Also map environments if they are clustered
        for env in policy.get("environment", []):
            name = (env.get("normalized") or env.get("full_value") or "").lower()
            cat = env.get("type", "environment")
            if name:
                true_classes[name] = cat

    # 2. Load cluster outputs
    with open(ATTRIBUTE_CLUSTERS_PATH, "r", encoding="utf-8") as f:
        cluster_data = json.load(f)
    
    clusters = cluster_data.get("clusters", [])
    if not clusters:
        print("[WARN] No clusters found to evaluate.")
        return

    # 3. Evaluate clusters
    avg_P, avg_R, avg_F1, details = evaluate_clusters(clusters, true_classes)

    # 4. Save and output results
    output_log = os.path.join(PROJECT_ROOT, "outputs", "logs", "cluster_eval.txt")
    os.makedirs(os.path.dirname(output_log), exist_ok=True)

    lines = []
    lines.append(f"Evaluating {len(clusters)} clusters using True Classes from Policy Dataset:")
    lines.append(f"  Average Precision : {avg_P:.4f}")
    lines.append(f"  Average Recall    : {avg_R:.4f}")
    lines.append(f"  Average F1-score  : {avg_F1:.4f}\n")
    lines.append("-" * 60)
    lines.append("Detailed Cluster Metrics:")
    for detail in details:
        lines.append(f"  Cluster {detail['cluster_id']:2d} ({detail['short_name']:<15}): "
                     f"Class='{detail['majority_class']}', Size={detail['size']}, "
                     f"P={detail['P']:.2f}, R={detail['R']:.2f}, F1={detail['F1']:.2f}")

    with open(output_log, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    for line in lines[:6]:
        print(line)
        
    print(f"[OK] Full cluster evaluation written to {output_log}")

if __name__ == "__main__":
    main()
