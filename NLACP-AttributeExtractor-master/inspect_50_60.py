import json

with open('outputs/policies/policy_dataset_gold.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for p in data['policies'][50:60]:
    print(f"ID: {p['id']}")
    print(f"Sentence: {p['sentence']}")
    subjects = [s.get('ref_tokens') for s in p.get('subjects', [])]
    print(f"Subject: {subjects}")
    resource = p.get('resource', {}).get('ref_tokens') if p.get('resource') else None
    print(f"Resource: {resource}")
    print('-'*40)
