#!/usr/bin/env python
"""
scripts/run_pipeline.py
Entry point — chạy toàn bộ ABAC pipeline.

Usage:
    python scripts/run_pipeline.py
    python scripts/run_pipeline.py --sentence "Nurses can read records during business hours."
"""
import argparse
import json
import os
import sys

# ── Ensure project root on path ───────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.pipeline.pipeline import process_sentence
from nlacp.mining.attribute_cluster import main as run_clustering
from nlacp.mining.namespace_hierarchy import main as run_hierarchy


def run_interactive():
    print("\n" + "="*55)
    print("  ABAC Policy Extractor — Full Pipeline")
    print("  (Alohaly et al. 2019 — 6-Module Framework)")
    print("="*55)
    print("\nType 'exit' to stop. Example:")
    print("  A senior nurse can read medical records during business hours.\n")

    while True:
        try:
            sentence = input("Enter policy sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not sentence:
            continue
        if sentence.lower() == "exit":
            break

        result = process_sentence(sentence)

        print("\n--- Extracted ABAC Policy ---")
        print(f"  Subject:  {result.get('subject')}")
        print(f"  Actions:  {result.get('actions', [])}")
        print(f"  Object:   {result.get('object')}")

        env_attrs   = [a for a in result.get("attributes", []) if a.get("category") == "environment"]
        other_attrs = [a for a in result.get("attributes", []) if a.get("category") != "environment"]

        if env_attrs:
            print("  Environment:")
            for a in env_attrs:
                print(f"    {a.get('namespace','?')} = \"{a.get('short_name','?')}\"  [{a.get('data_type','?')}]")
        if other_attrs:
            print("  Attributes:")
            for a in other_attrs:
                print(f"    [{a.get('category','?').upper()}] {a.get('namespace','?')} = \"{a.get('short_name','?')}\"")
        print()


def run_full_pipeline():
    """Chạy toàn bộ pipeline: NLP → Clustering → Hierarchy."""
    print("\n" + "="*55)
    print("  ABAC Full Pipeline Run")
    print("="*55)
    print("\n[Step 1/3] NLP extraction... (enter sentences, type 'done' to finish)")

    sentences_processed = 0
    while True:
        try:
            sentence = input("  Sentence (or 'done'): ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if sentence.lower() in ("done", "exit", ""):
            break
        process_sentence(sentence)
        sentences_processed += 1

    if sentences_processed > 0:
        print(f"\n[Step 2/3] Attribute clustering...")
        run_clustering()
        print(f"\n[Step 3/3] Building namespace hierarchy...")
        run_hierarchy()

    print("\n" + "="*55)
    print("  Pipeline done!")
    print("  Outputs in: outputs/policies/, outputs/clusters/, outputs/hierarchy/")
    print("="*55)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABAC Policy Extractor Pipeline")
    parser.add_argument("--sentence", default=None, help="Process a single sentence (non-interactive)")
    parser.add_argument("--full",     action="store_true", help="Run full pipeline (NLP + clustering + hierarchy)")
    args = parser.parse_args()

    if args.sentence:
        result = process_sentence(args.sentence)
        print(json.dumps(result, indent=2, ensure_ascii=False))
    elif args.full:
        run_full_pipeline()
    else:
        run_interactive()