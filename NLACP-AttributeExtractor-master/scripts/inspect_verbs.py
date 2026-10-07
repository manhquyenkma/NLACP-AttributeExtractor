import json

for ds in ['edocument', 'vact']:
    gold = json.load(open(f'outputs/policies/{ds}_gold.json', encoding='utf-8'))
    verbs = set()
    for p in gold['policies']:
        for a in p.get('actions', []):
            v = a.get('verb') if isinstance(a, dict) else a
            if v: verbs.add(v.lower().strip())
    print(f"=== {ds} unique gold verbs ===")
    print(sorted(list(verbs)))
