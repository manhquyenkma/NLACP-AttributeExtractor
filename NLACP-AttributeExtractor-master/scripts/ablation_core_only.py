#!/usr/bin/env python
"""
scripts/ablation_core_only.py

Ablation study for Domain vs Core Environment Extraction.
Runs the pipeline twice (core_only=False vs core_only=True) and compares F1 scores.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import json
from nlacp.pipeline.pipeline_v2 import AcpParser
from scripts.unified_eval import run as unified_eval_run

def run_ablation(dataset_name: str, input_file: str, gold_file: str, format: str):
    print(f"\n{'='*60}")
    print(f"Ablation Study: Core Only vs Full Gazetteer - {dataset_name}")
    print(f"{'='*60}")
    
    with open(input_file, "r", encoding="utf-8") as f:
        sentences = [line.strip() for line in f if line.strip()]
        
    # Run with FULL gazetteer (core_only = False)
    out_full = os.path.join(PROJECT_ROOT, "outputs", "policies", f"{dataset_name}_full.json")
    os.makedirs(os.path.dirname(out_full), exist_ok=True)
    parser_full = AcpParser(strict_validation=False, core_only=False)
    mach_policies_full = []
    for i, s in enumerate(sentences, 1):
        p, _ = parser_full.parse_sentence(s, policy_id=i)
        if p: mach_policies_full.append(json.loads(p.model_dump_json()))
        else: mach_policies_full.append({"id": i, "sentence": s})
    with open(out_full, "w", encoding="utf-8") as f:
        json.dump({"policies": mach_policies_full}, f, indent=2, ensure_ascii=False)
    
    # Run with CORE ONLY gazetteer (core_only = True)
    out_core = os.path.join(PROJECT_ROOT, "outputs", "policies", f"{dataset_name}_core.json")
    parser_core = AcpParser(strict_validation=False, core_only=True)
    mach_policies_core = []
    for i, s in enumerate(sentences, 1):
        p, _ = parser_core.parse_sentence(s, policy_id=i)
        if p: mach_policies_core.append(json.loads(p.model_dump_json()))
        else: mach_policies_core.append({"id": i, "sentence": s})
    with open(out_core, "w", encoding="utf-8") as f:
        json.dump({"policies": mach_policies_core}, f, indent=2, ensure_ascii=False)
    
    # Evaluate
    print("\n--- RESULTS: FULL GAZETTEER ---")
    res_full, _ = unified_eval_run("Full", format, gold_file, out_full)
    
    print("\n--- RESULTS: CORE ONLY GAZETTEER ---")
    res_core, _ = unified_eval_run("Core", format, gold_file, out_core)
    
    # Compare
    diff_env = res_full["Env"] - res_core["Env"]
    print(f"\n--- COMPARISON ---")
    print(f"Env F1 Full: {res_full['Env']:.4f}")
    print(f"Env F1 Core: {res_core['Env']:.4f}")
    print(f"Difference:  {diff_env:+.4f}")
    if diff_env > 0:
        print("Conclusion: Domain-specific hints IMPROVE extraction.")
    else:
        print("Conclusion: Domain-specific hints DO NOT improve extraction.")

if __name__ == "__main__":
    vact_input = os.path.join(PROJECT_ROOT, "vact_acp.txt")
    vact_gold = os.path.join(PROJECT_ROOT, "outputs", "policies", "vact_gold.json")
    if os.path.exists(vact_input) and os.path.exists(vact_gold):
        run_ablation("vact", vact_input, vact_gold, "vact")
    else:
        print("VACT dataset not found.")
