"""
tests/test_pipeline.py — Integration test for full pipeline.
Usage: python tests/test_pipeline.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.pipeline.pipeline import process_sentence

TEST_SENTENCE = "A senior nurse can read and write medical records during business hours within the hospital."
OUTPUT_FILE   = os.path.join(PROJECT_ROOT, "outputs", "logs", "pipeline_demo.txt")

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

result = process_sentence(TEST_SENTENCE)

lines = [
    f"INPUT: {TEST_SENTENCE}",
    "",
    "--- Extracted ABAC Policy (6 Modules) ---",
    f"Subject:  {result.get('subject')}",
    f"Actions:  {result.get('actions', [])}",
    f"Object:   {result.get('object')}",
]

env_attrs   = [a for a in result.get("attributes", []) if a.get("category") == "environment"]
other_attrs = [a for a in result.get("attributes", []) if a.get("category") != "environment"]

env_strings = [
    f"{a.get('namespace','?')} = \"{a.get('short_name','?')}\""
    for a in env_attrs
]
lines.append(f"Environment: {', '.join(env_strings)}" if env_strings else "Environment: (none)")
lines.append("Other Attributes:")
for a in other_attrs:
    lines.append(f"  [{a.get('category','?').upper()}] {a.get('namespace','?')} = \"{a.get('short_name','?')}\"")

output = "\n".join(lines) + "\n"

with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(output)

print(output)
print(f"[OK] Output written to {OUTPUT_FILE}")

# Basic assertions
assert result.get("object") == "records",   f"FAIL: object={result.get('object')!r}, expected 'records'"
assert "Write" in result.get("actions", []), f"FAIL: 'Write' not in actions={result.get('actions')}"
assert any(a.get("namespace","").startswith("environment:") for a in result.get("attributes",[])), \
    "FAIL: no environment namespace found"

print("[PASS] All basic assertions passed.")
