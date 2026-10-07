import spacy, json, os, sys
sys.path.insert(0, os.getcwd())
if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
from scripts.unified_eval import load, gold_envs, m_envs, score_field

configs = [
    ('VACT', 'vact', 'outputs/policies/vact_gold.json'),
    ('iTrust', 'itrust', 'outputs/policies/itrust_gold.json'),
    ('eDocument', 'edoc', 'outputs/policies/edocument_gold.json')
]

models = ['en_core_web_sm', 'en_core_web_trf']

results = {}
for model_name in models:
    try:
        nlp = spacy.load(model_name)
    except Exception as e:
        print(f"Could not load {model_name}: {e}")
        continue
    
    results[model_name] = {}
    for dname, fmt, gpath in configs:
        gold = load(gpath)
        mach = {}
        for pid, p in gold.items():
            sent = p.get('sentence', '')
            doc = nlp(sent)
            envs = []
            for ent in doc.ents:
                if ent.label_ in ['DATE', 'TIME', 'GPE', 'ORG', 'LOC']:
                    envs.append({'trigger_phrase': ent.text, 'full_value': ent.text})
            mach[pid] = {'id': pid, 'environments': envs}
        r, _ = score_field(gold, mach, gold_envs, m_envs, fmt, fuzzy=True)
        results[model_name][dname] = r['F1']
        print(f"Model: {model_name} | Dataset: {dname:<10} | P: {r['P']:.4f} | R: {r['R']:.4f} | F1: {r['F1']:.4f}")

print("\n=== SUMMARY NER BASELINES ===")
for model_name in results:
    row = " | ".join(f"{d}: {results[model_name][d]:.4f}" for d in results[model_name])
    print(f"{model_name}: {row}")
