import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator import evaluate, load_annotated, compute_prf

output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "vact_out_utf8.txt")
data_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "annotated", "vact_env_annotated.json")

with open(output_file, "w", encoding="utf-8") as f:
    sys.stdout = f
    data = load_annotated(data_path)
    print(f"\nEvaluating: {data_path} ({len(data)} sentences)")
    for cat in [None, "temporal", "spatial"]:
        label = cat.capitalize() if cat else "Overall"
        P, R, F1, tp, fp, fn = evaluate(data, cat, verbose=True)
        print(f"  {label:10s}: P={P:.4f}  R={R:.4f}  F1={F1:.4f}")
