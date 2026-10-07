import json, sys, os
sys.path.insert(0, os.getcwd())
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from scripts.unified_eval import (
    load, gold_subjects, m_subjects, gold_actions, m_actions,
    gold_objects, m_objects, gold_envs, m_envs, score_field
)

configs = [
    ('iTrust', 'itrust', 'outputs/policies/itrust_gold.json', 'outputs/policies/itrust_machine.json'),
    ('VACT', 'vact', 'outputs/policies/vact_gold.json', 'outputs/policies/vact_machine.json'),
    ('eDocument', 'edoc', 'outputs/policies/edocument_gold.json', 'outputs/policies/edocument_machine.json')
]

field_defs = [
    ('Subject', gold_subjects, m_subjects, True),
    ('Action', gold_actions, m_actions, True),
    ('Object', gold_objects, m_objects, True),
    ('Environment', gold_envs, m_envs, True),
]

print("=== BẢNG KẾT QUẢ CHÍNH THỨC ===")
print(f"{'Dataset':<10} | {'Field':<11} | {'Precision':<9} | {'Recall':<6} | {'F1':<6}")
print("-" * 55)

for name, fmt, gpath, mpath in configs:
    gold = load(gpath)
    mach = load(mpath)
    all_tp = all_fp = all_fn = 0
    first = True
    for fname, gext, mext, fuzzy in field_defs:
        r, _ = score_field(gold, mach, gext, mext, fmt, fuzzy=fuzzy)
        all_tp += r['TP']
        all_fp += r['FP']
        all_fn += r['FN']
        dname = name if first else ''
        first = False
        p_val = r['P']
        r_val = r['R']
        f1_val = r['F1']
        print(f"{dname:<10} | {fname:<11} | {p_val:.4f}    | {r_val:.4f} | {f1_val:.4f}")
    
    tot_p = all_tp / (all_tp + all_fp) if all_tp + all_fp else 0.0
    tot_r = all_tp / (all_tp + all_fn) if all_tp + all_fn else 0.0
    tot_f1 = 2 * tot_p * tot_r / (tot_p + tot_r) if tot_p + tot_r else 0.0
    print(f"{'':<10} | {'Overall':<11} | {tot_p:.4f}    | {tot_r:.4f} | {tot_f1:.4f}")
    print("-" * 55)
