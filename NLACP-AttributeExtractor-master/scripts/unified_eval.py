import json, sys, re
from collections import defaultdict

STOPWORDS = {"a", "an", "the"}

def singularize(word):
    if len(word) > 3 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 2 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word

def norm(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    parts = [p for p in s.split("_") if p and p not in STOPWORDS]
    parts = [singularize(p) for p in parts]
    return "_".join(parts)

def load(path):
    d = json.load(open(path, encoding="utf-8"))
    pols = d.get("policies", d) if isinstance(d, dict) else d
    return {p["id"]: p for p in pols}

# ---------- GOLD extractors: use ONLY the authoritative field per dataset,
# never merge a legacy duplicate field with the real one ----------

def gold_subjects(p, fmt):
    out = set()
    if fmt == "vact":
        for s in p.get("subjects", []):
            if isinstance(s, dict):
                v = s.get("role") or s.get("entity_type") or ""
                if v: out.add(norm(v))
    elif fmt == "itrust":
        for s in p.get("subject", []):
            if isinstance(s, str) and s.strip(): out.add(norm(s))
    elif fmt == "edoc":
        for s in p.get("subjects", []):
            if isinstance(s, str) and s.strip(): out.add(norm(s))
    return out

def gold_actions(p, fmt):
    out = set()
    for a in p.get("actions", []):
        if isinstance(a, str) and a.strip(): out.add(norm(a))
        elif isinstance(a, dict):
            v = a.get("verb", "")
            if v: out.add(norm(v))
    return out

def gold_objects(p, fmt):
    out = set()
    if fmt == "vact":
        r = p.get("resource", {})
        if isinstance(r, dict):
            v = r.get("label", "")
            if v: out.add(norm(v))
    else:
        for o in p.get("object", []):
            if isinstance(o, str) and o.strip(): out.add(norm(o))
    return out

def gold_envs(p, fmt):
    out = set()
    if fmt == "vact":
        for e in p.get("environments", []):
            if isinstance(e, dict):
                v = e.get("trigger_phrase", "")
                if v: out.add(norm(v))
    elif fmt == "itrust":
        for e in p.get("environment", []):
            if isinstance(e, dict):
                v = e.get("full_value", "") or e.get("phrase", "")
                if v: out.add(norm(v))
    elif fmt == "edoc":
        for e in p.get("environment", []):
            if isinstance(e, str) and e.strip(): out.add(norm(e))
    return out

def get_sub_attrs(p, fmt=None):
    out = set()
    for s in p.get("subjects", []):
        if isinstance(s, dict):
            q = s.get("qualifiers", {})
            if isinstance(q, dict):
                for k, v in q.items():
                    if v and isinstance(v, str) and v.lower() != 'false' and v.lower() != 'true':
                        nv = norm(v)
                        if nv: out.add(nv)
    return out

def get_obj_attrs(p, fmt=None):
    out = set()
    r = p.get("resource", {})
    if isinstance(r, dict):
        q = r.get("qualifier")
        if q and isinstance(q, str):
            nq = norm(q)
            if nq: out.add(nq)
        attrs = r.get("attributes")
        if isinstance(attrs, dict):
            for k, v in attrs.items():
                if v and isinstance(v, str):
                    nv = norm(v)
                    if nv: out.add(nv)
    return out

# ---------- MACHINE extractors: always the same shape ----------

def m_subjects(p):
    out = set()
    for s in p.get("subjects", []):
        if isinstance(s, dict):
            v = s.get("role") or s.get("entity_type") or ""
            if v: out.add(norm(v))
    return out

OP_TO_VERB = {"read":"read","create":"create","update":"update","delete":"delete","execute":"execute"}

def m_actions(p, fmt=None):
    out = set()
    for a in p.get("actions", []):
        if isinstance(a, dict):
            # For VACT, use the raw verb lemma. For iTrust/eDoc, use the CRUD operation.
            v = a.get("verb", "") if fmt == "vact" else a.get("operation", "")
            if v: out.add(norm(v))
    return out

def m_objects(p):
    out = set()
    r = p.get("resource", {})
    if isinstance(r, dict):
        v = r.get("label", "")
        if v: out.add(norm(v))
    return out

def m_envs(p):
    out = set()
    for e in p.get("environments", []):
        if isinstance(e, dict):
            v = e.get("trigger_phrase", "")
            if v: out.add(norm(v))
    return out

def token_overlap_match(g, m):
    """A gold item and a machine item match if one's token set is a subset
    of the other's (handles gold='grade' vs machine='final_grades...') OR exact match."""
    if g == m:
        return True
    gt, mt = set(g.split("_")), set(m.split("_"))
    if not gt or not mt:
        return False
    inter = gt & mt
    return len(inter) / min(len(gt), len(mt)) >= 0.6

def score_field(gold_map, mach_map, extractor_g, extractor_m, fmt, fuzzy=False):
    TP = FP = FN = 0
    errors = []
    for pid, gp in gold_map.items():
        mp = mach_map.get(pid, {})
        gset = extractor_g(gp, fmt) if 'fmt' in extractor_g.__code__.co_varnames else extractor_g(gp)
        mset = extractor_m(mp, fmt) if 'fmt' in extractor_m.__code__.co_varnames else extractor_m(mp)
        
        # Filter out empty strings
        gset = {g for g in gset if g}
        mset = {m for m in mset if m}
        
        matched_g, matched_m = set(), set()
        for g in gset:
            for m in mset:
                if m in matched_m:
                    continue
                if g == m or (fuzzy and token_overlap_match(g, m)):
                    matched_g.add(g); matched_m.add(m)
                    break
        tp = len(matched_g)
        fp = len(mset) - len(matched_m)
        fn = len(gset) - len(matched_g)
        TP += tp; FP += fp; FN += fn
        if fp or fn:
            errors.append((pid, gp["sentence"], gset, mset))
    P = TP/(TP+FP) if TP+FP else 0.0
    R = TP/(TP+FN) if TP+FN else 0.0
    F1 = 2*P*R/(P+R) if P+R else 0.0
    return {"TP":TP,"FP":FP,"FN":FN,"P":P,"R":R,"F1":F1}, errors

def run(name, fmt, gold_path, mach_path, strict_object=False):
    gold = load(gold_path)
    mach = load(mach_path)
    results = {}
    fields = [
        ("Subject", gold_subjects, m_subjects, True),
        ("Subj-Attr", get_sub_attrs, get_sub_attrs, True),
        ("Action", gold_actions, m_actions, False),
        ("Object", gold_objects, m_objects, not strict_object),
        ("Obj-Attr", get_obj_attrs, get_obj_attrs, True),
        ("Env", gold_envs, m_envs, True),
    ]
    all_errors = {}
    
    all_tp = 0
    all_fp = 0
    all_fn = 0
    
    for fname, gext, mext, fuzzy in fields:
        r, errs = score_field(gold, mach, gext, mext, fmt, fuzzy=fuzzy)
        results[fname] = r
        all_errors[fname] = errs
        
        # Only include in overall score if the gold dataset actually annotates this field
        if (r["TP"] + r["FN"]) > 0:
            all_tp += r["TP"]
            all_fp += r["FP"]
            all_fn += r["FN"]
            
    P = all_tp/(all_tp+all_fp) if all_tp+all_fp else 0
    R = all_tp/(all_tp+all_fn) if all_tp+all_fn else 0
    Overall = 2*P*R/(P+R) if P+R else 0
    return {"Dataset":name, "N":len(gold), "Subject":results["Subject"]["F1"],
            "Subj-Attr":results["Subj-Attr"]["F1"], "Action":results["Action"]["F1"],
            "Object":results["Object"]["F1"], "Obj-Attr":results["Obj-Attr"]["F1"],
            "Env":results["Env"]["F1"], "Overall":Overall}, all_errors

if __name__ == "__main__":
    configs = [
        ("VACT", "vact", "outputs/policies/vact_gold.json", "outputs/policies/vact_machine.json"),
        ("iTrust", "itrust", "outputs/policies/itrust_gold.json", "outputs/policies/itrust_machine.json"),
        ("eDocument", "edoc", "outputs/policies/edocument_gold.json", "outputs/policies/edocument_machine.json"),
    ]
    print(f"\n{'Dataset':<10} | {'N':>3} | {'Subj':>6} | {'S-Attr':>6} | {'Action':>6} | {'Object':>6} | {'O-Attr':>6} | {'Env':>6} | {'Overall':>7}")
    print("-" * 85)
    all_res = []
    for name, fmt, g, m in configs:
        res, errs = run(name, fmt, g, m)
        all_res.append((res, errs))
        print(f"{res['Dataset']:<10} | {res['N']:>3} | {res['Subject']:.4f} | {res['Subj-Attr']:.4f} | {res['Action']:.4f} | {res['Object']:.4f} | {res['Obj-Attr']:.4f} | {res['Env']:.4f} | {res['Overall']:.4f}")
    print()
    print("Strict (exact-match) Object F1 for reference:")
    for name, fmt, g, m in configs:
        res, errs = run(name, fmt, g, m, strict_object=True)
        print(f"  {name}: Object F1 (exact) = {res['Object']:.4f}")
