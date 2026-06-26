import json

with open('outputs/policies/policy_dataset_gold.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('report_50_100.md', 'w', encoding='utf-8') as f:
    for p in data['policies'][49:100]:  # 50 to 100 inclusive
        f.write(f"### ID: {p['id']}\n")
        f.write(f"**Sentence:** {p['sentence']}\n")
        
        subs = []
        for s in p.get('subjects', []):
            role = s.get('role')
            quals = s.get('qualifiers')
            subs.append(f"role={role}, qualifiers={quals}")
            
        f.write(f"**Subject:** {subs}\n")
        
        res = p.get('resource')
        if res:
            f.write(f"**Resource:** label={res.get('label')}, qualifier={res.get('qualifier')}, ref={res.get('ref_tokens')}\n")
        else:
            f.write("**Resource:** None\n")
        f.write("\n")
