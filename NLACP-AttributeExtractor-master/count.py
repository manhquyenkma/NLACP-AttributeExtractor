import json

with open('VACT_final.txt', 'r', encoding='utf-8') as f:
    lines = [l.strip() for l in f if l.strip()]
print(f'Total sentences in VACT_final.txt: {len(lines)}')

with open('outputs/policies/policy_dataset.json', 'r', encoding='utf-8') as f:
    data = json.load(f)
print(f'Total policies extracted: {len(data["policies"])}')

errors = [p['id'] for p in data['policies'] if p.get('authorization_decision') is None]
print(f'Total severely failed policies (fallback to NLP): {len(errors)}')
