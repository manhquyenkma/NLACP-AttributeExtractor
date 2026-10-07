import sys, os
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from scripts.unified_eval import load, gold_subjects, m_subjects, gold_actions, m_actions, gold_objects, m_objects, gold_envs, m_envs, score_field

datasets = [
    ('VACT', 'vact', 'outputs/policies/vact_gold.json', 'outputs/policies/vact_machine.json'),
    ('iTrust', 'itrust', 'outputs/policies/itrust_gold.json', 'outputs/policies/itrust_machine.json'),
    ('eDocument', 'edoc', 'outputs/policies/edocument_gold.json', 'outputs/policies/edocument_machine.json'),
]

print("="*75)
print("SO SÁNH CHI TIẾT: STRICT EXACT-MATCH vs. FUZZY SEMANTIC MATCH")
print("="*75)

for mode_name, fuzzy in [("STRICT EXACT-MATCH (fuzzy=False)", False), ("FUZZY SEMANTIC MATCH (fuzzy=True)", True)]:
    print(f"\n--- {mode_name} ---")
    print(f"{'Dataset':<10} | {'Subject':<8} | {'Action':<8} | {'Object':<8} | {'Env':<8} | {'Overall F1':<10}")
    print("-" * 65)
    for name, fmt, gpath, mpath in datasets:
        g = load(gpath)
        m = load(mpath)
        s, _ = score_field(g, m, gold_subjects, m_subjects, fmt, fuzzy=fuzzy)
        a, _ = score_field(g, m, gold_actions, m_actions, fmt, fuzzy=fuzzy)
        o, _ = score_field(g, m, gold_objects, m_objects, fmt, fuzzy=fuzzy)
        e, _ = score_field(g, m, gold_envs, m_envs, fmt, fuzzy=fuzzy)
        
        total_tp = s['TP'] + a['TP'] + o['TP'] + e['TP']
        total_fp = s['FP'] + a['FP'] + o['FP'] + e['FP']
        total_fn = s['FN'] + a['FN'] + o['FN'] + e['FN']
        P = total_tp / (total_tp + total_fp) if total_tp + total_fp else 0
        R = total_tp / (total_tp + total_fn) if total_tp + total_fn else 0
        F1 = 2*P*R / (P+R) if P+R else 0
        print(f"{name:<10} | {s['F1']:<8.4f} | {a['F1']:<8.4f} | {o['F1']:<8.4f} | {e['F1']:<8.4f} | {F1:<10.4f}")
