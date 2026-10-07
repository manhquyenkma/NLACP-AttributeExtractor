import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.pipeline.pipeline_v2 import parse_acp_sentence

CONFIGS = [
    ("vact", "outputs/policies/vact_gold.json", "outputs/policies/vact_machine.json"),
    ("itrust", "outputs/policies/itrust_gold.json", "outputs/policies/itrust_machine.json"),
    ("edocument", "outputs/policies/edocument_gold.json", "outputs/policies/edocument_machine.json"),
]

def regenerate():
    for name, gold_path, mach_path in CONFIGS:
        full_gold = os.path.join(PROJECT_ROOT, gold_path)
        full_mach = os.path.join(PROJECT_ROOT, mach_path)
        print(f"Regenerating {mach_path} from {gold_path}...")
        with open(full_gold, "r", encoding="utf-8") as f:
            gold_data = json.load(f)
        
        pols = gold_data.get("policies", gold_data) if isinstance(gold_data, dict) else gold_data
        mach_policies = []
        for p in pols:
            sentence = p.get("sentence", "")
            pid = p.get("id", 1)
            pol = parse_acp_sentence(sentence, policy_id=pid)
            if pol is not None:
                mach_policies.append(json.loads(pol.model_dump_json()))
            else:
                mach_policies.append({"id": pid, "sentence": sentence})
        
        out_data = {"policies": mach_policies}
        with open(full_mach, "w", encoding="utf-8") as f:
            json.dump(out_data, f, indent=4, ensure_ascii=False)
        print(f"Saved {len(mach_policies)} policies to {mach_path}")

if __name__ == "__main__":
    regenerate()
