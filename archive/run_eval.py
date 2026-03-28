#!/usr/bin/env python
"""
scripts/run_eval.py
Evaluate env_extractor over all annotated datasets in data/annotated/.

Usage:
    python scripts/run_eval.py
    python scripts/run_eval.py --verbose
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)
from nlacp.evaluation.evaluator import evaluate_by_dataset
import contextlib

data_dir    = os.path.join(PROJECT_ROOT, "data", "annotated")
output_file = os.path.join(PROJECT_ROOT, "outputs", "logs", "evaluator_results.txt")

os.makedirs(os.path.dirname(output_file), exist_ok=True)
with open(output_file, "w", encoding="utf-8") as f:
    with contextlib.redirect_stdout(f):
        evaluate_by_dataset(data_dir, verbose=False)
print(f"[OK] Results written to {output_file}")
