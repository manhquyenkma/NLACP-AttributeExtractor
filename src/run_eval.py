import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evaluator import evaluate_by_dataset

output_file = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "evaluator_results_clean.txt")
data_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "annotated")

with open(output_file, "w", encoding="utf-8") as f:
    sys.stdout = f
    evaluate_by_dataset(data_dir, verbose=False)
