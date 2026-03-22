#!/usr/bin/env python
"""
scripts/run_pipeline.py
Entry point — chay toan bo ABAC pipeline (2-step).

Usage:
    python scripts/run_pipeline.py                         # Full 2-step pipeline
    python scripts/run_pipeline.py --sentence "..."        # Quick single-sentence test (old mode)
    python scripts/run_pipeline.py --step2                 # Chi chay Step 2 (da co policy_dataset.json)
"""
import argparse
import json
import os
import sys

# Ensure project root on path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.pipeline.pipeline import process_sentence


def run_single_sentence(sentence):
    """Quick test: xu ly 1 cau va in ra ket qua."""
    result = process_sentence(sentence)

    print("\n--- Extracted ABAC Policy ---")
    print(f"  Subject  : {result.get('subject')}")
    print(f"  Actions  : {result.get('actions', [])}")
    print(f"  Object   : {result.get('object')}")

    # Environment dung cho: result["environment"]
    env_list = result.get("environment", [])
    if env_list:
        print(f"  Environment ({len(env_list)}):")
        for e in env_list:
            if "namespace" in e:
                print(f"    [{e.get('type','?')}] {e.get('full_value','?')} "
                      f"-> {e.get('namespace','?')}  [{e.get('data_type','?')}]")
            else:
                print(f"    {e.get('value', e)}")
    else:
        print("  Environment: (none detected)")

    # SA/OA Attributes dung cho: result["attributes"]
    attrs = result.get("attributes", [])
    if attrs:
        print(f"  Attributes ({len(attrs)}):")
        for a in attrs:
            print(f"    [{a.get('category','?').upper()}] "
                  f"{a.get('namespace','?')} = \"{a.get('short_name','?')}\"")
    else:
        print("  Attributes: (none detected)")
    print()

    # Also print raw JSON for debugging
    import json
    print(json.dumps(result, indent=2, ensure_ascii=False))


def run_full_pipeline():
    """Chay full 2-step pipeline."""
    print("\n" + "=" * 55)
    print("  ABAC Full Pipeline (2-Step)")
    print("=" * 55)

    # Step 1
    print("\n[STEP 1] Data Processing...")
    from scripts.data_processing import main as run_step1
    run_step1()

    # Step 2
    print("\n[STEP 2] ABAC Extraction...")
    from scripts.ABAC_extraction import main as run_step2
    run_step2()

    print("\n" + "=" * 55)
    print("  Pipeline HOAN TAT!")
    print("=" * 55)


def run_step2_only():
    """Chi chay Step 2 tren policy_dataset.json co san."""
    from nlacp.paths import POLICY_DATASET_PATH
    if not os.path.exists(POLICY_DATASET_PATH):
        print(f"[ERROR] {POLICY_DATASET_PATH} khong ton tai. Chay Step 1 truoc.")
        return

    print("\n[STEP 2] ABAC Extraction (FAST MODE)...")
    from scripts.ABAC_extraction import main as run_step2
    run_step2()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ABAC Policy Extractor Pipeline")
    parser.add_argument("--sentence", default=None,
                        help="Xu ly 1 cau (non-interactive, backward compatible)")
    parser.add_argument("--step2", action="store_true",
                        help="Chi chay Step 2 (da co policy_dataset.json)")
    args = parser.parse_args()

    if args.sentence:
        run_single_sentence(args.sentence)
    elif args.step2:
        run_step2_only()
    else:
        run_full_pipeline()