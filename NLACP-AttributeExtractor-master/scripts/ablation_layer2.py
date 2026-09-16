import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import nlacp.extraction.env_extractor as ee
from scripts.compute_fscore import _compute_tp_fp_fn_sets, _env_key
from nlacp.extraction.relation_candidate import parse_sentence, extract_relations

GOLD_PATH = os.path.join(PROJECT_ROOT, "outputs", "policies", "vact_gold.json")

def run():
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        gold_data = json.load(f)
    policies = gold_data.get("policies", [])
    
    total_tp_with, total_fp_with, total_fn_with = 0, 0, 0
    total_tp_without, total_fp_without, total_fn_without = 0, 0, 0
    
    discarded_examples = []
    
    orig_disambiguate = ee._disambiguate
    
    for policy in policies:
        sentence = policy["sentence"]
        gold_envs = policy.get("environments", policy.get("environment", []))
        gold_keys = {_env_key(e) for e in gold_envs if _env_key(e)}
        
        tokens = parse_sentence(sentence)
        rel = extract_relations(sentence, tokens)
        sub = rel.get("subject", [])
        obj = rel.get("object", [])
        
        # subject_tokens requires list of words.
        # rel["subject"] returns list of tokens.
        subject_tokens = set()
        for p in sub:
            for w in str(p).split():
                subject_tokens.add(w.lower())
        
        object_tokens = set()
        for p in obj:
            for w in str(p).split():
                object_tokens.add(w.lower())

        # WITH layer 2
        ee._disambiguate = orig_disambiguate
        ext_with = ee.extract_env_attributes(sentence, subject_tokens, object_tokens)
        keys_with = {_env_key(e) for e in ext_with if _env_key(e)}
        tp1, fp1, fn1 = _compute_tp_fp_fn_sets(gold_keys, keys_with)
        total_tp_with += tp1; total_fp_with += fp1; total_fn_with += fn1
        
        # WITHOUT layer 2
        ee._disambiguate = lambda candidates, st, ot: candidates
        ext_without = ee.extract_env_attributes(sentence, subject_tokens, object_tokens)
        keys_without = {_env_key(e) for e in ext_without if _env_key(e)}
        tp2, fp2, fn2 = _compute_tp_fp_fn_sets(gold_keys, keys_without)
        total_tp_without += tp2; total_fp_without += fp2; total_fn_without += fn2
        
        # FIND differences
        for k in keys_without:
            if k not in keys_with:
                if k not in gold_keys:
                    discarded_examples.append({
                        "sentence": sentence,
                        "candidate": k,
                        "reason": "Filtered by Layer 2 Disambiguation"
                    })
    
    ee._disambiguate = orig_disambiguate
    
    def get_f1(tp, fp, fn):
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        return (2 * p * r) / (p + r) if (p + r) > 0 else 0
        
    f1_with = get_f1(total_tp_with, total_fp_with, total_fn_with)
    f1_without = get_f1(total_tp_without, total_fp_without, total_fn_without)
    
    print(f"With Layer 2: ENV F1={f1_with:.4f}, FP={total_fp_with}")
    print(f"Without Layer 2: ENV F1={f1_without:.4f}, FP={total_fp_without}")
    print(f"Δ FP reduction: {total_fp_without - total_fp_with}")
    
    for ex in discarded_examples[:3]:
        print(f"Sentence: {ex['sentence']}")
        print(f"Candidate: {ex['candidate']}")

if __name__ == "__main__":
    run()
