import json

gold = json.load(open('outputs/policies/itrust_gold.json', encoding='utf-8'))
print("=== All iTrust sentences and their gold actions ===")
for p in gold['policies']:
    acts = [a.get('verb') if isinstance(a, dict) else a for a in p.get('actions', [])]
    print(f"ID {p['id']:2d}: {p['sentence'][:65]:<65} -> {acts}")
