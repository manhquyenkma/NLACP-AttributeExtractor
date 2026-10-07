import json, sys, os
import spacy

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.extraction.subject_extractor import extract_subjects

nlp = spacy.load('en_core_web_sm')
edoc = json.load(open('outputs/policies/edocument_gold.json', encoding='utf-8'))
for p in edoc['policies'][:10]:
    s = p.get('sentence')
    doc = nlp(s)
    subjs = extract_subjects(s, doc)
    roles = [sub.role for sub in subjs]
    print(f"ID {p['id']:2d}: Gold={p.get('subjects')} -> Extracted={roles}")
