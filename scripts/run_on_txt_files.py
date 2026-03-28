#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/run_on_txt_files.py

This script processes the raw policy sentences from VACT_ACP.txt and iTrust_gold.txt
and runs the FULL 2-step ABAC extraction pipeline, including the interactive 
CNN candidate validation step.
"""
import sys, os, json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.extraction.relation_candidate import extract_relations, parse_sentence
from nlacp.extraction.env_extractor import extract_env_attributes
from nlacp.extraction.short_name_suggester import suggest_short_name

DATASET_DIR = os.path.join(PROJECT_ROOT, "dataset")
CANDIDATE_PATH = os.path.join(DATASET_DIR, "relation_candidate.json")
POLICY_PATH = os.path.join(DATASET_DIR, "policy_dataset.json")

def load_sentences(txt_file):
    with open(txt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip() and not line.startswith("#")]

def step1_generate_candidates(sentences):
    """
    Step 1A: Parse sentences and extract subject/object/action hypotheses
    """
    print(f"\n[STEP 1A] Extracting relation candidates from {len(sentences)} sentences...")
    relations = []
    
    # Pre-read existing
    existing = []
    if os.path.exists(CANDIDATE_PATH):
        try:
            with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
                existing = json.load(f).get("relations", [])
        except Exception:
            pass

    # Filter already processed to save time, but for demonstration we'll just rewrite
    
    for idx, sent in enumerate(sentences, 1):
        rel = extract_relations(sent, [])
        
        ENV_PREPS = {"during","within","after","before","between","via","through","using","at","on","from"}
        relation_pairs = []
        for attr in rel.get("attributes", []):
            if attr["name"].lower() in ENV_PREPS:
                continue
            pair = [attr["value"], attr["name"]]
            if pair not in relation_pairs:
                relation_pairs.append(pair)
                
        relations.append({
            "id": idx,
            "sentence": rel["sentence"],
            "subject": rel.get("subject", ""),
            "object": rel.get("object", ""),
            "actions": rel.get("actions", []),
            "relation_pairs": relation_pairs
        })

    with open(CANDIDATE_PATH, "w", encoding="utf-8") as f:
        json.dump({"relations": relations}, f, indent=4, ensure_ascii=False)
    print(f" -> Saved {len(relations)} relation candidates to {CANDIDATE_PATH}")


def step1_validate_cnn():
    """
    Step 1B: Interactive validation of candidates (Simulating CNN/Human validation)
    """
    print("\n[STEP 1B] CNN Validation (Interactive)")
    if not os.path.exists(CANDIDATE_PATH):
        print("Candidate file not found!")
        return []

    with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
        relations = json.load(f).get("relations", [])

    validated_policies = []
    
    print("\n--- [CNN VALIDATION] ---")
    print("For each pair, type: y (yes/keep), n (no/drop), a (accept all), s (skip all)")
    
    for rel in relations:
        print(f"\nSentence: {rel['sentence']}")
        pairs = rel.get("relation_pairs", [])
        
        valid_pairs = []
        auto_mode = None
        
        for i, pair in enumerate(pairs):
            if auto_mode == "accept_all":
                choice = "y"
            elif auto_mode == "skip_all":
                choice = "n"
            else:
                choice = input(f"    [{i+1}]  {pair[0]} <-> {pair[1]} \n    Dung hay sai? (y/n/a/s): ").strip().lower()
                
            if choice == "a":
                auto_mode = "accept_all"
                valid_pairs.append(pair)
            elif choice == "s":
                auto_mode = "skip_all"
            elif choice == "y":
                valid_pairs.append(pair)
                
        # Reconstruct final policy based on validation
        rel["relation_pairs"] = valid_pairs
        validated_policies.append(rel)

    return validated_policies

def step2_extract_abac(validated_policies):
    """
    Step 2: ABAC Attribute Extraction
    Build the final subject/object/environment structure based on validated relations
    """
    print("\n[STEP 2] ABAC Extraction & Formatting...")
    final_policies = []
    
    for pol in validated_policies:
        sent = pol["sentence"]
        
        # S/A/O
        subj = pol.get("subject", "")
        obj = pol.get("object", "")
        acts = pol.get("actions", [])
        
        # Validated attributes
        valid_pairs = pol.get("relation_pairs", [])
        abac_attrs = []
        
        for vp in valid_pairs:
            val, name = vp[0], vp[1]
            cat = "subject" if val == subj else "object" if val == obj else "unclassified"
            abac_attrs.append({
                "category": cat,
                "name": name,
                "value": val,
                "short_name": suggest_short_name(name, val),
                "is_active": True
            })
            
        # Parse Environment/Context (not part of the CNN validation, extracted cleanly)
        envs = extract_env_attributes(sent)
        
        final_policies.append({
            "id": pol["id"],
            "sentence": sent,
            "subject": subj,
            "object": obj,
            "actions": acts,
            "attributes": abac_attrs,
            "environment": envs
        })
        
    with open(POLICY_PATH, "w", encoding="utf-8") as f:
        json.dump({"policies": final_policies}, f, indent=4, ensure_ascii=False)
        
    print(f" -> Saved {len(final_policies)} ABAC policies to {POLICY_PATH}")
    print("\nPipeline Complete!\n")

def main():
    print("=" * 60)
    print("  NLACP - FULL PIPELINE (WITH CNN VALIDATION)")
    print("=" * 60)
    
    vact_txt = os.path.join(PROJECT_ROOT, "VACT_ACP.txt")
    itrust_txt = os.path.join(PROJECT_ROOT, "iTrust_gold.txt")
    
    # For a manageable demo, let's just process the first 5 from VACT
    # Because asking for 184 sentences manually takes forever!
    print("Loading test dataset (First 5 sentences of VACT_ACP)...")
    sents = load_sentences(vact_txt)[:5]
    
    step1_generate_candidates(sents)
    validated = step1_validate_cnn()
    if validated:
        step2_extract_abac(validated)

if __name__ == "__main__":
    main()
