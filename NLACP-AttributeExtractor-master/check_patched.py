import json

with open('outputs/policies/policy_dataset_gold.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

for p in data['policies']:
    if p['id'] in [68, 69, 72, 75, 92, 95, 96, 99]:
        print(f"ID: {p['id']}")
        print(f"Sentence: {p['sentence']}")
        print(f"  Decision: {p.get('authorization_decision')}")
        print(f"  Subjects: {len(p.get('subjects', []))} items")
        print(f"  Actions : {len(p.get('actions', []))} items")
        print(f"  Resource: {p.get('resource', {}).get('namespace')}")
        print(f"  Envs    : {len(p.get('environments', []))} items")
        print(f"  Context : {len(p.get('context', []))} items")
        print('-'*50)
