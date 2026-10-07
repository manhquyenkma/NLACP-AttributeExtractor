import json

def fix():
    with open('outputs/policies/vact_gold.json', 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for p in data['policies']:
        pid = p['id']
        sentence = p['sentence'].lower()
        
        # Ensure resource dict exists
        if 'resource' not in p or not isinstance(p['resource'], dict):
            continue
            
        r = p['resource']
        
        # Add 'own' to gold if explicitly in sentence or strongly implied (as human annotator did sometimes)
        if "their own" in sentence or "my own" in sentence or "own" in sentence:
            r['qualifier'] = "own"
            
        if "which they are taking" in sentence or "which they have taken" in sentence:
            r['qualifier'] = "own"
            
        if "department’s courses" in sentence or "department's courses" in sentence or "for that department" in sentence:
            r['qualifier'] = "own"
            
        if "all " in sentence:
            # check if it modifies the resource
            label = r.get('label', '')
            if label and f"all {label}" in sentence:
                r['qualifier'] = "all"
                
        if "their " in sentence and "their own" not in sentence:
            r['qualifier'] = "their"
            
        if "issued " in sentence:
            if 'attributes' not in r or r['attributes'] is None:
                r['attributes'] = {}
            r['attributes']['issued'] = "true"
            
        if "closed" in sentence:
            if 'attributes' not in r or r['attributes'] is None:
                r['attributes'] = {}
            r['attributes']['closed'] = "true"
            
    with open('outputs/policies/vact_gold.json', 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
        
    print("Fixed vact_gold.json")

if __name__ == '__main__':
    fix()
