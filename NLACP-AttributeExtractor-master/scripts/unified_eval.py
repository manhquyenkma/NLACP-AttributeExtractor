import json, sys, re
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
from collections import defaultdict

STOPWORDS = {"a", "an", "the"}
PREPS = {"in", "on", "at", "during", "from", "within", "between", "for", "if", "only", "even"}

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

def norm_env(s):
    if not isinstance(s, str):
        return ""
    s = s.lower().strip()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    parts = [p for p in s.split("_") if p and p not in STOPWORDS]
    while parts and parts[0] in PREPS:
        parts = parts[1:]
    parts = [singularize(p) for p in parts]
    return "_".join(parts)

def load(path):
    d = json.load(open(path, encoding="utf-8"))
    pols = d.get("policies", d) if isinstance(d, dict) else d
    return {p["id"]: p for p in pols}

# ---------- GOLD extractors: use ONLY the authoritative field per dataset ----------

def gold_subjects(p, fmt):
    out = set()
    subjs = p.get("subjects", []) or p.get("subject", [])
    if isinstance(subjs, str): subjs = [subjs]
    for s in subjs:
        if isinstance(s, str) and s.strip():
            out.add(norm(s))
        elif isinstance(s, dict):
            v = s.get("role") or s.get("entity_type") or s.get("label") or s.get("value") or ""
            if v: out.add(norm(v))
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
    objs = p.get("resources", []) or p.get("object", []) or [p.get("resource", {})]
    if isinstance(objs, str): objs = [objs]
    for o in objs:
        if isinstance(o, str) and o.strip():
            out.add(norm(o))
        elif isinstance(o, dict):
            v = o.get("label") or o.get("type") or o.get("value") or ""
            if v: out.add(norm(v))
    return out

def gold_envs(p, fmt):
    out = set()
    if fmt == "vact":
        for e in p.get("environments", []):
            if isinstance(e, dict):
                v = e.get("trigger_phrase", "")
                if v: out.add(norm_env(v))
    elif fmt == "itrust":
        for e in p.get("environment", []):
            if isinstance(e, dict):
                v = e.get("full_value", "") or e.get("phrase", "")
                if v: out.add(norm_env(v))
    elif fmt == "edoc":
        for e in p.get("environment", []):
            if isinstance(e, str) and e.strip(): out.add(norm_env(e))
    return out

def get_sub_attrs(p, fmt=None):
    out = set()
    for s in (p.get("subjects", []) or p.get("subject", [])):
        if isinstance(s, dict):
            q = s.get("qualifiers", {})
            if isinstance(q, dict):
                for k, v in q.items():
                    if v and isinstance(v, str) and v.lower() != 'false':
                        nv = norm(k) if v.lower() == 'true' else norm(v)
                        if nv: out.add(nv)
    return out

def get_obj_attrs(p, fmt=None):
    out = set()
    objs = p.get("resources", []) or p.get("object", []) or [p.get("resource", {})]
    if isinstance(objs, str): objs = [objs]
    for r in objs:
        if isinstance(r, dict):
            q = r.get("qualifier")
            if q and isinstance(q, str):
                nq = norm(q)
                if nq: out.add(nq)
            attrs = r.get("attributes", {})
            if isinstance(attrs, dict):
                for k, v in attrs.items():
                    if v and isinstance(v, str) and v.lower() != 'false':
                        nv = norm(k) if v.lower() == 'true' else norm(v)
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

def m_actions(p, fmt=None):
    out = set()
    for a in p.get("actions", []):
        if isinstance(a, dict):
            if fmt == "itrust":
                v = a.get("operation", "") or a.get("verb", "")
            else:
                v = a.get("verb", "")
            if v: out.add(norm(v))
    return out

def m_objects(p):
    out = set()
    for o in (p.get("resources", []) or [p.get("resource", {})]):
        if isinstance(o, dict):
            v = o.get("label", "")
            if v: out.add(norm(v))
    return out

def m_envs(p):
    out = set()
    for e in p.get("environments", []):
        if isinstance(e, dict):
            v = e.get("trigger_phrase", "")
            if v: out.add(norm_env(v))
    return out

def token_overlap_match(g, m, threshold=0.6):
    """Token Jaccard IoU matching. Replaces subset matching to prevent inflating F1."""
    if g == m:
        return True
    gt, mt = set(g.split("_")), set(m.split("_"))
    if not gt or not mt:
        return False
    inter = gt & mt
    union = gt | mt
    return (len(inter) / len(union)) >= threshold

# Action synonym groups: verbs in the same group are considered equivalent
ACTION_SYNONYM_GROUPS = [
    {"read", "view", "see", "check", "access", "browse", "look", "choose"},
    {"create", "add", "upload", "log", "register", "submit"},
    {"update", "edit", "modify", "change", "alter", "revise"},
    {"delete", "remove", "cancel"},
    {"manage", "configure", "administer", "control"},
    {"send", "transmit", "forward", "dispatch"},
    {"assign", "allocate", "designate"},
    {"schedule", "book", "arrange"},
    {"approve", "accept", "grant"},
    {"deny", "reject", "refuse"},
    {"search", "find", "query", "look_up"},
]

def _build_synonym_map():
    m = {}
    for group in ACTION_SYNONYM_GROUPS:
        for v in group:
            m[v] = group
    return m

_ACTION_SYN = _build_synonym_map()

def action_synonym_match(g, m):
    """Return True if g and m are synonyms (same CRUD group)."""
    if g == m:
        return True
    group = _ACTION_SYN.get(g)
    if group and m in group:
        return True
    return False

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
                if g == m or (fuzzy and (token_overlap_match(g, m) or action_synonym_match(g, m))):
                    matched_g.add(g); matched_m.add(m)
                    break
        tp = len(matched_g)
        fp = len(mset) - len(matched_m)
        fn = len(gset) - len(matched_g)
        TP += tp; FP += fp; FN += fn
        if fp or fn:
            errors.append((pid, gp.get("sentence", ""), gset, mset))
    P = TP/(TP+FP) if TP+FP else 0.0
    R = TP/(TP+FN) if TP+FN else 0.0
    F1 = 2*P*R/(P+R) if P+R else 0.0
    return {"TP":TP,"FP":FP,"FN":FN,"P":P,"R":R,"F1":F1}, errors

def run(name, fmt, gold_path, mach_path, mode="relaxed"):
    gold = load(gold_path)
    mach = load(mach_path)
    results = {}
    is_fuzzy = (mode == "relaxed")
    
    fields = [
        ("Subject", gold_subjects, m_subjects, is_fuzzy),
        ("Subj-Attr", get_sub_attrs, get_sub_attrs, is_fuzzy),
        ("Action", gold_actions, m_actions, is_fuzzy),
        ("Object", gold_objects, m_objects, is_fuzzy),
        ("Obj-Attr", get_obj_attrs, get_obj_attrs, is_fuzzy),
        ("Env", gold_envs, m_envs, is_fuzzy),
    ]
    all_errors = {}
    
    all_tp = 0
    all_fp = 0
    all_fn = 0
    
    for fname, gext, mext, fz in fields:
        r, errs = score_field(gold, mach, gext, mext, fmt, fuzzy=fz)
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
    
    # Format unannotated fields as None
    s_attr_f1 = results["Subj-Attr"]["F1"] if (results["Subj-Attr"]["TP"] + results["Subj-Attr"]["FN"]) > 0 else None
    o_attr_f1 = results["Obj-Attr"]["F1"] if (results["Obj-Attr"]["TP"] + results["Obj-Attr"]["FN"]) > 0 else None
    
    return {
        "Dataset": name,
        "N": len(gold),
        "Subject": results["Subject"]["F1"],
        "Subj-Attr": s_attr_f1,
        "Action": results["Action"]["F1"],
        "Object": results["Object"]["F1"],
        "Obj-Attr": o_attr_f1,
        "Env": results["Env"]["F1"],
        "Overall": Overall
    }, all_errors

def fmt_val(v):
    if v is None:
        return "   N/A"
    return f"{v:.4f}"

def print_table(title, results):
    print(f"\n{title}")
    print(f"{'Dataset':<10} | {'N':>3} | {'Subj':>6} | {'S-Attr':>6} | {'Action':>6} | {'Object':>6} | {'O-Attr':>6} | {'Env':>6} | {'Overall':>7}")
    print("-" * 85)
    for res in results:
        print(f"{res['Dataset']:<10} | {res['N']:>3} | {fmt_val(res['Subject'])} | {fmt_val(res['Subj-Attr'])} | {fmt_val(res['Action'])} | {fmt_val(res['Object'])} | {fmt_val(res['Obj-Attr'])} | {fmt_val(res['Env'])} | {fmt_val(res['Overall'])}")

if __name__ == "__main__":
    configs = [
        ("VACT", "vact", "outputs/policies/vact_gold.json", "outputs/policies/vact_machine.json"),
        ("iTrust", "itrust", "outputs/policies/itrust_gold.json", "outputs/policies/itrust_machine.json"),
        ("eDocument", "edoc", "outputs/policies/edocument_gold.json", "outputs/policies/edocument_machine.json"),
    ]
    
    # 1. Strict Evaluation
    strict_res = []
    for name, fmt, g, m in configs:
        res, _ = run(name, fmt, g, m, mode="strict")
        strict_res.append(res)
    print_table("=== TABLE 1: STRICT EVALUATION (Exact-Match) ===", strict_res)
    
    # 2. Relaxed Evaluation
    relaxed_res = []
    for name, fmt, g, m in configs:
        res, _ = run(name, fmt, g, m, mode="relaxed")
        relaxed_res.append(res)
    print_table("=== TABLE 2: RELAXED EVALUATION (Token IoU >= 0.6 + Action Synonyms) ===", relaxed_res)
    print()
