"""
verify_fixes.py — Quick verification script for all 8 bug fixes
"""
import sys
import os
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

print("Loading spaCy..."); import spacy  # noqa
print("=" * 60)
print("VERIFY: Bug fixes in NLACP-AttributeExtractor")
print("=" * 60)

# ─── STEP 1: relation_candidate ───────────────────────────────
print("\n--- STEP 1: relation_candidate ---")
from nlacp.extraction.relation_candidate import parse_sentence, extract_relations

tests_rc = [
    ("Nurses can read and write medical records during business hours within the hospital.",
     {"expected_obj": "records", "expected_write": "Write"}),
    ("An on-call senior nurse may change the list of approved lab procedures.",
     {"expected_obj": "procedures", "expected_write": None}),
]

for s, expect in tests_rc:
    r = extract_relations(s, parse_sentence(s))
    obj_ok     = r["object"] == expect["expected_obj"]
    write_ok   = (expect["expected_write"] is None) or (expect["expected_write"] in r["actions"])
    status = "OK" if obj_ok and write_ok else "FAIL"
    print(f"  [{status}] object={r['object']!r} actions={r['actions']}")
    if not obj_ok:
        print(f"         EXPECTED object={expect['expected_obj']!r}")

# ─── STEP 2: short_name_suggester ─────────────────────────────
print("\n--- STEP 2: short_name_suggester ---")
from nlacp.extraction.short_name_suggester import suggest_short_names

env_attrs = [
    {"category": "temporal", "value": "during business hours"},
    {"category": "spatial",  "value": "within the hospital"},
]
result_sn = suggest_short_names(env_attrs)
for r in result_sn:
    has_trigger = any(t in r["short_name"] for t in ["during", "within", "business"])
    status = "OK" if has_trigger else "FAIL"
    print(f"  [{status}] '{r['value']}' => '{r['short_name']}'")

# ─── STEP 3 & 4: category_identifier + namespace_assigner ─────
print("\n--- STEP 3 & 4: category_identifier + namespace_assigner ---")
from nlacp.normalization.category_identifier import identify_categories
from nlacp.normalization.namespace_assigner import assign_namespaces

cat_test = [
    {"category": "temporal", "value": "during business hours", "short_name": "during_business_hour"},
    {"category": "spatial",  "value": "within the hospital",   "short_name": "within_hospital"},
    {"category": "subject",  "name": "senior",  "value": "nurse", "short_name": "senior_nurse"},
]
cat_result = identify_categories(cat_test, "test")
ns_result  = assign_namespaces(cat_result, "nurse", "records")
for r in ns_result:
    ns = r.get("namespace", "")
    ok = not ns.startswith("unknown:") and not ns.startswith("object:") if r.get("category") == "environment" else True
    status = "OK" if ok else "FAIL"
    print(f"  [{status}] [{r.get('category')}] => {ns!r}")

# ─── STEP 5: nlp_engine (swap order) ──────────────────────────
print("\n--- STEP 5: nlp_engine ---")
from nlacp.pipeline.pipeline import process_sentence
result_pipeline = process_sentence(
    "A senior nurse can read and write medical records during business hours within the hospital."
)
print(f"  Subject: {result_pipeline.get('subject')}")
print(f"  Actions: {result_pipeline.get('actions')}")
print(f"  Object:  {result_pipeline.get('object')}")
print("  Attributes:")
for a in result_pipeline.get("attributes", []):
    print(f"    [{a.get('category')}] ns={a.get('namespace')} dt={a.get('data_type')}")

obj_ok = result_pipeline.get("object") == "records"
env_ok = any(a.get("namespace", "").startswith("environment:time:") 
             for a in result_pipeline.get("environment", []))
print(f"\n  [{'OK' if obj_ok else 'FAIL'}] object == 'records'")
print(f"  [{'OK' if env_ok else 'FAIL'}] environment:time namespace exists")

# ─── STEP 6: data_type_infer ──────────────────────────────────
print("\n--- STEP 6: data_type_infer ---")
from nlacp.normalization.data_type_infer import infer_data_type
dt = infer_data_type("during business hours", category="environment", sub_category="temporal")
print(f"  [{'OK' if dt == 'datetime' else 'FAIL'}] 'during business hours' => {dt!r} (want 'datetime')")

print("\n" + "=" * 60)
print("Verification done.")
