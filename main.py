import subprocess
import sys
import os

# ===================================================================
# main.py — Entry point toàn bộ pipeline ABAC
# (Alohaly et al. 2019 — 5-Module Framework)
#
# Pipeline:
#   Module 1: NLP Extraction   (src/nlp_engine.py)
#   Module 2: Clustering       (mining/attribute_cluster.py)
#   Module 3: Hierarchy        (mining/namespace_hierarchy.py)
#   Module 4: Category         (tích hợp trong Module 1)
#   Module 5: Data Type        (tích hợp trong Module 1)
# ===================================================================

PYTHON = sys.executable

def run(script, label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    result = subprocess.run([PYTHON, script])
    if result.returncode != 0:
        print(f"\n[WARN] {script} exited with code {result.returncode}")


def main():
    base = os.path.dirname(os.path.abspath(__file__))

    print("\n" + "="*55)
    print("  ABAC Policy Framework — Full Pipeline")
    print("  Based on: Alohaly et al. (2019)")
    print("  Cybersecurity 2:2")
    print("="*55)

    # Module 1 + 4 + 5: NLP Extraction + Category + Data Type
    run(os.path.join(base, "src", "nlp_engine.py"),
        "Module 1 / 4 / 5: NLP Extraction + Category + Data Type")

    # Module 2: Attribute Clustering
    run(os.path.join(base, "mining", "attribute_cluster.py"),
        "Module 2: Attribute Clustering (DBSCAN + Word Embeddings)")

    # Module 3: Hierarchical Namespace
    run(os.path.join(base, "mining", "namespace_hierarchy.py"),
        "Module 3: Hierarchical Namespace Assignment")

    # Summary
    print("\n" + "="*55)
    print("  Pipeline Completed!")
    print("  Output files:")
    print(f"    dataset/policy_dataset.json")
    print(f"    dataset/attribute_clusters.json")
    print(f"    dataset/namespace_hierarchy.json")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()