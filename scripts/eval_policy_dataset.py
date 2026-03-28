#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_policy_dataset.py — Evaluate the CNN-validated policy_dataset.json
against the manual gold standards (iTrust and KMA_ACP).

Usage:
    python scripts/eval_policy_dataset.py
    python scripts/eval_policy_dataset.py --mode exact
"""

import sys, io, os, json, re, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

POLICY_JSON        = os.path.join(PROJECT_ROOT, "dataset", "policy_dataset.json")
ITRUST_GOLD_JSON   = os.path.join(PROJECT_ROOT, "dataset", "itrust_manual_gold.json")
VACT_GOLD_JSON     = os.path.join(PROJECT_ROOT, "dataset", "vact_manual_gold.json")
ITRUST_TXT         = os.path.join(PROJECT_ROOT, "iTrust_gold.txt")
VACT_TXT           = os.path.join(PROJECT_ROOT, "VACT_ACP.txt")

# ── Helpers ───────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    if not text:
        return ""
    text = text.lower().strip()
    return re.sub(r"^[^\w]+|[^\w]+$", "", text)

def _match(a: str, b: str, mode: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if not na and not nb:
        return True
    if not na or not nb:
        return False
    if mode == "exact":
        return na == nb
    elif mode == "partial":
        return na.split()[:2] == nb.split()[:2]
    else:
        sa, sb = set(na.split()), set(nb.split())
        return len(sa & sb) / len(sa | sb) >= 0.5 if (sa | sb) else False

def match_lists(gold: list, pred: list, mode: str):
    matched_g, matched_p = set(), set()
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gold):
            if gi in matched_g:
                continue
            if _match(p, g, mode):
                matched_p.add(pi)
                matched_g.add(gi)
                break
    tp = len(matched_p)
    return tp, len(pred) - tp, len(gold) - len(matched_g)

def prf(tp, fp, fn):
    P  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return round(P, 4), round(R, 4), round(F1, 4)

def empty_stat():
    return {"tp": 0, "fp": 0, "fn": 0, "gold_n": 0, "pred_n": 0}

def accumulate(stat, gold, pred, mode):
    tp, fp, fn = match_lists(gold, pred, mode)
    stat["tp"]     += tp
    stat["fp"]     += fp
    stat["fn"]     += fn
    stat["gold_n"] += len(gold)
    stat["pred_n"] += len(pred)

# ── Loaders ───────────────────────────────────────────────────────────────────

def load_txt(path):
    if not os.path.exists(path):
        return []
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

def load_manual_gold(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    for sent, attrs in data.items():
        out[sent] = {
            "subject": [_norm(v) for v in attrs.get("subject", []) if v],
            "object":  [_norm(v) for v in attrs.get("object",  []) if v],
            "context": [_norm(v) for v in attrs.get("context", []) if v],
            "actions": [_norm(v) for v in attrs.get("actions", []) if v],
        }
    return out

def load_policy_predictions(path):
    out = {}
    if not os.path.exists(path):
        return out
    with open(path, encoding="utf-8") as f:
        data = json.load(f).get("policies", [])
    
    for pol in data:
        sent = pol.get("sentence", "")
        subj_base = _norm(pol.get("subject") or "")
        obj_base  = _norm(pol.get("object")  or "")
        acts = [_norm(a) for a in pol.get("actions", []) if a]
        
        subj_attrs = []
        obj_attrs  = []
        ctx_attrs  = []
        
        for a in pol.get("attributes", []):
            name = _norm(a.get("name", ""))
            val  = _norm(a.get("value", ""))
            cat  = a.get("category", "")
            
            # Context attributes
            if cat == "context":
                ctx_attrs.append(f"{name} {val}".strip())
            
            # Build Full NP modifying subject/object
            if cat == "subject" or val == subj_base:
                subj_attrs.append(name)
            elif cat == "object" or val == obj_base:
                obj_attrs.append(name)
                
        subj = (" ".join(subj_attrs) + " " + subj_base).strip() if subj_base else ""
        obj  = (" ".join(obj_attrs) + " " + obj_base).strip() if obj_base else ""
        
        # Environments
        envs = pol.get("environment", [])
        pred_ctx = list(ctx_attrs)
        for e in envs:
            val = e.get("full_value") or e.get("value") or ""
            if val:
                pred_ctx.append(_norm(val))
                
        out[_norm(sent)] = {
            "subj": [subj] if subj else [],
            "obj":  [obj]  if obj  else [],
            "acts": acts,
            "ctx":  pred_ctx,
        }
    return out

# ── Evaluations ───────────────────────────────────────────────────────────────

def evaluate_dataset(ds_name, sentences, manual_gold, predictions, mode, verbose):
    sa_stat  = empty_stat()
    oa_stat  = empty_stat()
    ctx_stat = empty_stat()
    act_stat = empty_stat()
    matched = 0
    pred_found = 0

    for sent in sentences:
        norm_s = _norm(sent)
        
        gold = manual_gold.get(sent)
        if not gold:
            for k, v in manual_gold.items():
                if _norm(k) == norm_s:
                    gold = v
                    break

        gold_subj = gold["subject"] if gold else []
        gold_obj  = gold["object"]  if gold else []
        gold_ctx  = gold["context"] if gold else []
        gold_acts = gold["actions"] if gold else []
        if gold:
            matched += 1
            
        pred = predictions.get(norm_s)
        if not pred:
            # Maybe slight difference in formatting
            for k, v in predictions.items():
                if k in norm_s or norm_s in k:
                    pred = v
                    break
                    
        if pred:
            pred_found += 1
            pred_subj = pred["subj"]
            pred_obj  = pred["obj"]
            pred_ctx_ = pred["ctx"]
            pred_acts = pred["acts"]
        else:
            pred_subj, pred_obj, pred_ctx_, pred_acts = [], [], [], []

        accumulate(sa_stat,  gold_subj, pred_subj, mode)
        accumulate(oa_stat,  gold_obj,  pred_obj,  mode)
        accumulate(ctx_stat, gold_ctx,  pred_ctx_, mode)
        accumulate(act_stat, gold_acts, pred_acts, mode)

    print(f"[INFO] {ds_name} - Sentences: {len(sentences)}, Gold Matched: {matched}/{len(sentences)}, Predictions Found: {pred_found}/{len(sentences)}")
    return {
        "n_sents": len(sentences),
        "subject": sa_stat,
        "object":  oa_stat,
        "context": ctx_stat,
        "actions": act_stat,
    }

# ── Table printer ─────────────────────────────────────────────────────────────

def fmt_prf(stat):
    if stat is None:
        return "  N/A  ", "  N/A  ", "  N/A  "
    P, R, F1 = prf(stat["tp"], stat["fp"], stat["fn"])
    return f"{P:.4f}", f"{R:.4f}", f"{F1:.4f}"

COL_TYPES = [
    ("Subject-attribute", "subject"),
    ("Object-attribute",  "object"),
    ("Context-attribute", "context"),
    ("Actions' Names",    "actions"),
]

def print_table(itrust, kma, mode):
    DS_W = 22
    N_W  = 6
    M_W  = 8
    dashes = DS_W + (N_W + 3 * M_W + 6) * 4 + 3

    print()
    print("=" * dashes)
    print(f"  NLACP ATTRIBUTE EXTRACTION — EVALUATION TABLE (POST-CNN)  (mode={mode})")
    print("=" * dashes)

    h1 = f"  {'Dataset':<{DS_W}}"
    h2 = f"  {'':<{DS_W}}"
    for lbl, _ in COL_TYPES:
        w = N_W + 3 * M_W + 5
        h1 += f"  {lbl:^{w}}"
        h2 += f"  {'#':>{N_W}} {'P':>{M_W}} {'R':>{M_W}} {'F1':>{M_W}}"
    print(h1)
    print(h2)
    print("  " + "-" * (dashes - 2))

    for ds_label, res in [("iTrust", itrust), ("KMA_ACP", kma)]:
        row = f"  {ds_label:<{DS_W}}"
        for _, atype in COL_TYPES:
            stat = res[atype]
            if stat is None:
                row += f"  {'0':>{N_W}} {'N/A':>{M_W}} {'N/A':>{M_W}} {'N/A':>{M_W}}"
            else:
                n = stat["gold_n"]
                P, R, F1 = fmt_prf(stat)
                row += f"  {n:>{N_W}} {P:>{M_W}} {R:>{M_W}} {F1:>{M_W}}"
        print(row)

    print("  " + "=" * (dashes - 2))
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P/R/F1 evaluation from post-CNN policy_dataset.json")
    parser.add_argument("--mode", default="partial", choices=["exact", "partial", "overlap"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n+" + "=" * 70 + "+")
    print("|" + "  NLACP — Evaluating post-CNN policy_dataset.json  ".center(70) + "|")
    print("+" + "=" * 70 + "+\n")

    itrust_sents = load_txt(ITRUST_TXT)
    vact_sents   = load_txt(VACT_TXT)

    itrust_gold  = load_manual_gold(ITRUST_GOLD_JSON)
    vact_gold    = load_manual_gold(VACT_GOLD_JSON)
    predictions  = load_policy_predictions(POLICY_JSON)

    print(f"[INFO] Loaded {len(predictions)} predicted policies from {os.path.basename(POLICY_JSON)}")
    
    print("\n--- Evaluating iTrust ---")
    itrust = evaluate_dataset("iTrust", itrust_sents, itrust_gold, predictions, args.mode, args.verbose)

    print("\n--- Evaluating Collected KMA_ACP ---")
    kma = evaluate_dataset("Collected_KMA_ACP", vact_sents, vact_gold, predictions, args.mode, args.verbose)

    print_table(itrust, kma, args.mode)


if __name__ == "__main__":
    main()
