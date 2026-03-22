"""Quick test: runs Step 1 and Step 2 non-interactively."""
import json, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from nlacp.extraction.relation_candidate import extract_relations, parse_sentence

DATASET_DIR    = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset")
CANDIDATE_PATH = os.path.join(DATASET_DIR, "relation_candidate.json")
POLICY_PATH    = os.path.join(DATASET_DIR, "policy_dataset.json")

# --- Step 1A: Extract ---
sentences = [
    "A senior nurse can approve medical records during business hours within the hospital.",
    "Managers in the finance department can approve expense reports.",
]

relations = []
for idx, sentence in enumerate(sentences, 1):
    tokens = parse_sentence(sentence)
    result = extract_relations(sentence, tokens)
    pairs = [[a["value"], a["name"]] for a in result.get("attributes", [])]
    relations.append({
        "id": idx,
        "sentence": result["sentence"],
        "subject": result.get("subject"),
        "actions": result.get("actions", []),
        "object": result.get("object"),
        "attributes": result.get("attributes", []),
        "relation_pairs": pairs
    })

os.makedirs(DATASET_DIR, exist_ok=True)
with open(CANDIDATE_PATH, "w", encoding="utf-8") as f:
    json.dump({"relations": relations}, f, indent=4, ensure_ascii=False)
print(f"[OK] Step 1A: {len(relations)} relations saved to relation_candidate.json")
for r in relations:
    print(f"  #{r['id']}: {len(r['relation_pairs'])} pairs -> {r['relation_pairs']}")

# --- Step 1B: Auto-validate all pairs as 'y' ---
policies = []
for rel in relations:
    policies.append({
        "id": rel["id"],
        "sentence": rel["sentence"],
        "subject": rel.get("subject"),
        "actions": rel.get("actions", []),
        "object": rel.get("object"),
        "environment": [],
        "attributes": rel.get("attributes", []),
        "relation_pairs": rel.get("relation_pairs", [])
    })

with open(POLICY_PATH, "w", encoding="utf-8") as f:
    json.dump({"policies": policies}, f, indent=4, ensure_ascii=False)
print(f"[OK] Step 1B: {len(policies)} policies saved to policy_dataset.json")

# --- Step 2: ABAC Extraction ---
print("\n[INFO] Running Step 2...")
from ABAC_extraction import fill_environment, fill_attributes

with open(POLICY_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

for policy in data["policies"]:
    envs = fill_environment(policy)
    fill_attributes(policy)
    print(f"\n  Policy #{policy['id']}: {policy['sentence'][:60]}...")
    print(f"    Environment: {len(envs)} entries")
    for e in envs:
        print(f"      [{e['type']}] {e['preposition']} {e.get('modifier','')} {e['head']}")
    attrs = policy.get("attributes", [])
    print(f"    Attributes:  {len(attrs)} entries")
    for a in attrs:
        print(f"      [{a.get('category','')}] {a.get('name','')} -> {a.get('value','')}"
              f"  short_name={a.get('short_name','')} ns={a.get('namespace','')}")

with open(POLICY_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"\n[OK] Final policy_dataset.json saved!")
print("\n=== VERIFICATION ===")

# Verify output
with open(POLICY_PATH, "r", encoding="utf-8") as f:
    final = json.load(f)

p1 = final["policies"][0]
assert p1["subject"] == "nurse", f"Expected subject 'nurse', got {p1['subject']}"
assert len(p1["environment"]) >= 1, "Expected at least 1 environment entry"
assert any(e["type"] == "temporal" for e in p1["environment"]), "Expected temporal env"
assert len(p1["attributes"]) >= 1, "Expected at least 1 attribute"

# Check no environment overlap in attributes
env_heads = {e["head"].lower() for e in p1["environment"]}
for attr in p1["attributes"]:
    name = attr.get("name", "").lower()
    value = attr.get("value", "").lower()
    # name should not be an env preposition
    assert name not in {"during", "within", "between", "after", "before"}, \
        f"Attribute name '{name}' should not be an env preposition"

print("ALL CHECKS PASSED!")
