import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.abspath('.'))
sys.path.insert(0, PROJECT_ROOT)

import spacy
# Use transformer model
nlp = spacy.load('en_core_web_trf')

def run_baseline(machine_path, out_path):
    with open(machine_path, 'r', encoding='utf-8') as f:
        machine = json.load(f)
        
    for p in machine['policies']:
        sentence = p['sentence']
        doc = nlp(sentence)
        envs = []
        for ent in doc.ents:
            if ent.label_ in ['DATE', 'TIME', 'GPE', 'ORG', 'LOC']:
                envs.append({
                    'category': 'environment',
                    'env_type': ent.label_.lower(),
                    'phrase': ent.text,
                    'full_value': ent.text
                })
        p['environment'] = envs
        p['environments'] = envs
            
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(machine, f, indent=4, ensure_ascii=False)
    p_len = len(machine['policies'])
    print('Done processing {} policies'.format(p_len))

if __name__ == '__main__':
    run_baseline('outputs/policies/policy_dataset.json', 'outputs/policies/vact_baseline_trf.json')
    run_baseline('outputs/policies/iTrust_machine.json', 'outputs/policies/itrust_baseline_trf.json')
