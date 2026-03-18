"""
candidate_generator.py
Sinh positive + negative candidates để train CNN.
Output: outputs/policies/relation_candidate.json
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.extraction.relation_candidate import generate_candidates

OUT_PATH = os.path.join(PROJECT_ROOT, "outputs", "policies", "relation_candidate.json")

def process_file(input_path):
    if not os.path.exists(input_path):
        print(f"[ERROR] Can't find file: {input_path}")
        print("Please provide a valid .jsonl or .txt path (e.g., data/raw/cyber_acp.jsonl)")
        return

    print(f"[INFO] Reading sentences from {input_path}...")
    sentences = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"): continue
            
            # Support both jsonl and plain text
            if input_path.endswith(".jsonl"):
                try:
                    data = json.loads(line)
                    if "text" in data:
                        sentences.append(data["text"])
                except json.JSONDecodeError:
                    pass
            else:
                sentences.append(line)

    print(f"[INFO] Generating positive & negative candidates for {len(sentences)} sentences...")
    results = {"sentences": [generate_candidates(s) for s in sentences]}
    
    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
        
    print(f"[OK] Generated {len(results['sentences'])} sentence records")
    print(f"[OK] Saved to: {OUT_PATH}")
    print("\nTo annotate this data, run: python scripts/annotate.py")


if __name__ == "__main__":
    default_path = os.path.join(PROJECT_ROOT, "data", "raw", "cyber_acp.jsonl")
    path = sys.argv[1] if len(sys.argv) > 1 else default_path
    
    print("=========================================================")
    print("  CNN RELATION CANDIDATE GENERATOR")
    print("=========================================================\n")
    process_file(path)
