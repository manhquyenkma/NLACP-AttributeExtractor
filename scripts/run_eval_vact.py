#!/usr/bin/env python
"""
scripts/run_eval_vact.py
Evaluate env_extractor on the VACT synthetic dataset.

Usage:
    python scripts/run_eval_vact.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.evaluation.evaluator import evaluate, load_annotated, compute_prf

data_path   = os.path.join(PROJECT_ROOT, "data", "annotated", "vact_env_annotated.json")
output_file = os.path.join(PROJECT_ROOT, "outputs", "logs", "vact_eval.txt")

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    sys.stdout = f
    data = load_annotated(data_path)
    print(f"\nEvaluating: {data_path} ({len(data)} sentences)")
    for cat in [None, "temporal", "spatial"]:
        label = cat.capitalize() if cat else "Overall"
        P, R, F1, tp, fp, fn = evaluate(data, cat, verbose=True)
        print(f"  {label:10s}: P={P:.4f}  R={R:.4f}  F1={F1:.4f}")

sys.stdout = sys.__stdout__
print(f"[OK] VACT eval written to {output_file}")
