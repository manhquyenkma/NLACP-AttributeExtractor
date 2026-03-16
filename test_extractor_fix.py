"""test_extractor_fix.py — Test device patterns sau khi fix"""
import sys
sys.path.insert(0, "src")
from env_extractor import extract_env_attributes

tests = [
    "Administrators using trusted workstations can modify system settings.",
    "Doctors using hospital-issued devices can access patient records.",
    "Staff via VPN can access internal systems.",
    "IT staff through the admin console can view server logs.",
    "A doctor can view patient records during business hours.",
    "Nurses from the hospital network can update patient charts.",
    "A senior nurse on night shift may change approved lab procedures.",
    "A registered patient may view his full health record.",
]

print("\n=== Device Pattern Test ===\n")
for s in tests:
    attrs = extract_env_attributes(s)
    print(f"IN:  {s[:70]}")
    if attrs:
        for a in attrs:
            print(f"  [{a['category']}/{a['subcategory']}] '{a['value']}' ({a['trigger']})")
    else:
        print("  (none)")
    print()
