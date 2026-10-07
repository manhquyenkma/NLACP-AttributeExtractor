import json, sys, os
sys.path.insert(0, os.getcwd())
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')

from nlacp.pipeline.pipeline_v2 import parse_acp_sentence
from nlacp.utils.nlp_utils import get_spacy_model

nlp = get_spacy_model()
vact = {p['id']: p for p in json.load(open('outputs/policies/vact_gold.json', encoding='utf-8'))['policies']}

for pid in [23, 75, 92]:
    p = vact[pid]
    sent = p['sentence']
    print(f"\n==================== POLICY {pid} ====================")
    print(f"Sentence: {sent}")
    print(f"Gold Subj: {p.get('subjects') or p.get('subject')}")
    print(f"Gold Action: {p.get('actions') or p.get('action')}")
    print(f"Gold Resource: {p.get('resources') or p.get('object') or p.get('resource')}")
    print(f"Gold Env: {p.get('environments') or p.get('environment')}")
    
    doc = nlp(sent)
    print("\nTokens:")
    for t in doc:
        print(f"  {t.i:2d}: {t.text:<15} lemma={t.lemma_:<12} pos={t.pos_:<6} dep={t.dep_:<10} head={t.head.text}")
    
    pol = parse_acp_sentence(sent, policy_id=pid)
    if pol:
        print("Parsed Policy:")
        print("  Subj:", [s.model_dump() for s in pol.subjects])
        print("  Act: ", [a.model_dump() for a in pol.actions])
        print("  Res: ", pol.resource.model_dump() if pol.resource else None)
        print("  Env: ", [e.model_dump() for e in pol.environments])
