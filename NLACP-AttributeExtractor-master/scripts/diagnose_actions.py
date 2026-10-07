import json
import sys
from unified_eval import run

cfgs = [
    ('VACT', 'vact', 'outputs/policies/vact_gold.json', 'outputs/policies/vact_machine.json'),
    ('iTrust', 'itrust', 'outputs/policies/itrust_gold.json', 'outputs/policies/itrust_machine.json'),
    ('eDocument', 'edoc', 'outputs/policies/edocument_gold.json', 'outputs/policies/edocument_machine.json'),
]

for name, fmt, g, m in cfgs:
    res, errs = run(name, fmt, g, m)
    action_errs = errs.get('Action', [])
    print(f"=== {name} Action Errors ({len(action_errs)} errors) ===")
    for pid, s, gset, mset in action_errs:
        print(f"  ID {pid}: gold={gset} vs mach={mset} | {s[:70]}")
