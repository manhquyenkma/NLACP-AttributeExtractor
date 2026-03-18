"""
tests/test_extractor_fix.py — Tests for env_extractor patterns.
Usage: python tests/test_extractor_fix.py
"""
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.extraction.env_extractor import extract_env_attributes

TESTS = [
    ("=== Temporal/Relative ===", [
        ("A doctor can view records during business hours.", "temporal"),
        ("Staff may access the system between 8am and 5pm on weekdays.", "temporal"),
        ("Nurses on night shift may update patient charts.", "temporal"),
    ]),
    ("=== Spatial/Network ===", [
        ("Managers accessing from internal VPN can approve reports.", "spatial"),
        ("Administrators using trusted workstations can modify settings.", "spatial"),
        ("Nurses from the hospital network can update charts.", "spatial"),
    ]),
    ("=== No Env-att (should return empty) ===", [
        ("A registered patient may view his full health record.", None),
        ("The administrator may delete outdated logs.", None),
    ]),
]

total = passed = 0
for section, cases in TESTS:
    print(f"\n{section}")
    for sentence, expected_cat in cases:
        total += 1
        attrs = extract_env_attributes(sentence)
        cats  = [a["category"] for a in attrs]

        if expected_cat is None:
            ok = len(attrs) == 0
        else:
            ok = expected_cat in cats

        status = "PASS" if ok else "FAIL"
        if ok:
            passed += 1
        print(f"  [{status}] {sentence[:70]}")
        if not ok:
            print(f"         expected={expected_cat!r}  got={cats}")

print(f"\n{'='*50}")
print(f"Results: {passed}/{total} passed")
if passed < total:
    sys.exit(1)
