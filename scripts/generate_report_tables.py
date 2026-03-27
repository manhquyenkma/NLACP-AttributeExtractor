import json
import csv
from collections import defaultdict
import os
import sys

root = r"c:\Users\PAV\Downloads\NLACP-AttributeExtractor"
sys.path.insert(0, root)

from nlacp.extraction.env_extractor import extract_env_attributes

def norm(t):
    import re
    t = t.lower().strip()
    return re.sub(r'^[^\w]+|[^\w]+$', '', t)

# Mapping from raw Source to 3 defined datasets
SOURCE_MAP = {
    "t2p": "iTrust",
    "acre": "iTrust",               # assuming acre is healthcare/iTrust
    "ibm": "UHP_Sample_Policies",
    "cyber": "UHP_Sample_Policies",
    "self_created": "Collected_KMA_ACP",
    "collected": "Collected_KMA_ACP"
}

def main():
    root = r"c:\Users\PAV\Downloads\NLACP-AttributeExtractor"
    
    # 1. Load CSV to get Sentence -> Dataset mapping
    sent_to_dataset = {}
    csv_path = os.path.join(root, "data", "annotation_sheet.csv")
    with open(csv_path, encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        for row in reader:
            s_raw = row.get("Source", "").strip()
            ds = SOURCE_MAP.get(s_raw, "Unknown")
            sent = row.get("Sentence", "").strip()
            sent_to_dataset[sent] = ds

    # 2. Count attributes in policy_dataset.json (TABLE 1)
    policy_path = os.path.join(root, "outputs", "policies", "policy_dataset.json")
    with open(policy_path, encoding='utf-8') as f:
        pdata = json.load(f)
        
    counts = defaultdict(lambda: {"sentences": 0, "SA": 0, "OA": 0, "CA": 0, "Actions": 0})
    
    # Dictionaries to store P,R,F1 stats for Context
    context_stats = defaultdict(lambda: {"tp":0, "fp":0, "fn":0})
    
    # Read LLM Gold CSV for Context
    llm_csv_path = os.path.join(root, "data", "annotation_llm_gold.csv")
    llm_gold = {}
    with open(llm_csv_path, encoding='utf-8-sig') as f:
        for row in csv.DictReader(f):
            sent = row["Sentence"].strip()
            tf = row.get("temporal_final","").strip()
            sf = row.get("spatial_final","").strip()
            g = []
            if tf: g.append(norm(tf))
            if sf: g.append(norm(sf))
            llm_gold[sent] = g
            
    # Process policy dataset
    for pol in pdata["policies"]:
        sent = pol["sentence"].strip()
        ds = sent_to_dataset.get(sent, "Unknown")
        
        counts[ds]["sentences"] += 1
        
        # Actions
        actions = pol.get("actions", [])
        counts[ds]["Actions"] += len(actions)
        
        # Subject / Object attributes
        attrs = pol.get("attributes", [])
        counts[ds]["SA"] += sum(1 for a in attrs if a.get("category") == "subject")
        counts[ds]["OA"] += sum(1 for a in attrs if a.get("category") == "object")
        
        # Context (Env) Gold counts from LLM CSV
        gold_envs = llm_gold.get(sent, [])
        counts[ds]["CA"] += len(gold_envs)
        
        # Evaluate Context Attribute
        pred_envs = [norm(e.get("value","")) for e in extract_env_attributes(sent)]
        
        # match logic
        mp, mg = set(), set()
        for pi, p in enumerate(pred_envs):
            for gi, g in enumerate(gold_envs):
                if gi in mg: continue
                # partial match top 2 words
                if p.split()[:2] == g.split()[:2]:
                    mp.add(pi); mg.add(gi); break
        
        tp = len(mp)
        fp = len(pred_envs) - tp
        fn = len(gold_envs) - len(mg)
        context_stats[ds]["tp"] += tp
        context_stats[ds]["fp"] += fp
        context_stats[ds]["fn"] += fn

    # Print Table 1
    print("\n" + "="*80)
    print(" TABLE 1: COUNTS")
    print("="*80)
    print(f"{'Dataset':30} | {'Sentences':>10} | {'Subj-attr':>10} | {'Obj-attr':>10} | {'Ctx-attr':>10} | {'Actions':>10}")
    print("-" * 80)
    for ds in ["iTrust", "UHP_Sample_Policies", "Collected_KMA_ACP", "Unknown"]:
        c = counts[ds]
        if c["sentences"] > 0:
            print(f"{ds:30} | {c['sentences']:>10} | {c['SA']:>10} | {c['OA']:>10} | {c['CA']:>10} | {c['Actions']:>10}")

    # Print Table 2 Context
    print("\n" + "="*80)
    print(" TABLE 2: METRICS (CONTEXT ONLY FOR NOW)")
    print("="*80)
    print(f"{'Dataset':30} | {'Precision':>10} | {'Recall':>10} | {'F1':>10}")
    print("-" * 80)
    for ds in ["iTrust", "UHP_Sample_Policies", "Collected_KMA_ACP"]:
        st = context_stats[ds]
        t=st["tp"]; f=st["fp"]; n=st["fn"]
        P = t/(t+f) if t+f > 0 else 0
        R = t/(t+n) if t+n > 0 else 0
        F1 = 2*P*R/(P+R) if P+R > 0 else 0
        print(f"{ds:30} | {P*100:>9.2f}% | {R*100:>9.2f}% | {F1*100:>9.2f}%")

if __name__ == "__main__":
    main()
