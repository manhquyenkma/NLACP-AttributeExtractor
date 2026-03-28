#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
eval_from_txt.py — Evaluate extraction pipeline on iTrust_gold.txt and VACT_ACP.txt
=====================================================================================

Usage:
    python scripts/eval_from_txt.py
    python scripts/eval_from_txt.py --mode exact
    python scripts/eval_from_txt.py --verbose

Gold sources:
  - iTrust  (iTrust_gold.txt):   annotation_sheet.csv (source=t2p|acre)  → context only
  - KMA_ACP (VACT_ACP.txt):      policy_dataset(clone).json              → subject/object/actions
                                  annotation_llm_gold.csv                 → context (temporal+spatial)
"""

import sys, io, os, csv, json, re, argparse
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.extraction.relation_candidate import extract_relations
from nlacp.extraction.env_extractor     import extract_env_attributes

# ── Paths ────────────────────────────────────────────────────────────────────
ITRUST_TXT         = os.path.join(PROJECT_ROOT, "iTrust_gold.txt")
VACT_TXT           = os.path.join(PROJECT_ROOT, "VACT_ACP.txt")
ITRUST_GOLD_JSON   = os.path.join(PROJECT_ROOT, "dataset", "itrust_manual_gold.json")
VACT_GOLD_JSON     = os.path.join(PROJECT_ROOT, "dataset", "vact_manual_gold.json")

ITRUST_SOURCES = {"t2p", "acre"}

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
    else:  # overlap (Jaccard >= 0.5)
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
    """Load non-empty lines from a .txt file."""
    with open(path, encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip() and not ln.startswith("#")]

def load_manual_gold(path):
    """
    Load JSON gold data formatted as:
    sentence → {"subject": [], "object": [], "context": [], "actions": []}
    """
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

# ── Predictor ─────────────────────────────────────────────────────────────────

def predict(sent: str, mode: str, verbose: bool = False):
    """Run extractor on a sentence, return {subj, obj, acts, ctx}."""
    try:
        rel = extract_relations(sent, [])
        subj_base = _norm(rel.get("subject") or "")
        obj_base  = _norm(rel.get("object")  or "")
        acts = [_norm(a) for a in rel.get("actions", []) if a]
        
        # Build full noun phrase from attributes
        subj_attrs = []
        obj_attrs  = []
        for a in rel.get("attributes", []):
            name = _norm(a.get("name", ""))
            val  = _norm(a.get("value", ""))
            if val == subj_base:
                subj_attrs.append(name)
            elif val == obj_base:
                obj_attrs.append(name)
                
        # Reconstruct full NP: modifier + head noun
        subj = (" ".join(subj_attrs) + " " + subj_base).strip() if subj_base else ""
        obj  = (" ".join(obj_attrs) + " " + obj_base).strip() if obj_base else ""
        
        # If multiple targets extracted, maybe we return list. For now, matching original wrapper.
    except Exception as e:
        if verbose:
            print(f"  [WARN relations] {e}")
        subj, obj, acts = "", "", []

    try:
        envs    = extract_env_attributes(sent)
        pred_ctx = [_norm(e.get("value", "")) for e in envs if e.get("value")]
    except Exception as e:
        if verbose:
            print(f"  [WARN env] {e}")
        pred_ctx = []

    return {
        "subj": [subj] if subj else [],
        "obj":  [obj]  if obj  else [],
        "acts": acts,
        "ctx":  pred_ctx,
    }

# ── Evaluations ───────────────────────────────────────────────────────────────

def evaluate_dataset(ds_name, sentences, manual_gold, mode, verbose):
    """
    Generic evaluator for datasets using the manual JSON gold standard.
    """
    sa_stat  = empty_stat()
    oa_stat  = empty_stat()
    ctx_stat = empty_stat()
    act_stat = empty_stat()
    matched = 0

    for sent in sentences:
        gold = manual_gold.get(sent)
        if not gold:
            for k, v in manual_gold.items():
                if k.lower() == sent.lower():
                    gold = v
                    break

        gold_subj = gold["subject"] if gold else []
        gold_obj  = gold["object"]  if gold else []
        gold_ctx  = gold["context"] if gold else []
        gold_acts = gold["actions"] if gold else []
        if gold:
            matched += 1

        pred = predict(sent, mode, verbose)

        accumulate(sa_stat,  gold_subj, pred["subj"], mode)
        accumulate(oa_stat,  gold_obj,  pred["obj"],  mode)
        accumulate(ctx_stat, gold_ctx,  pred["ctx"],  mode)
        accumulate(act_stat, gold_acts, pred["acts"], mode)

    print(f"[INFO] {ds_name} sentences: {len(sentences)}, gold matched: {matched}/{len(sentences)}")
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
    print(f"  NLACP ATTRIBUTE EXTRACTION — EVALUATION TABLE  (mode={mode})")
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

    for ds_label, res in [("iTrust", itrust), ("Collected KMA_ACP", kma)]:
        row = f"  {ds_label:<{DS_W}}"
        for _, atype in COL_TYPES:
            stat = res[atype]
            if stat is None:
                key = "pred_sa" if atype == "subject" else \
                      "pred_oa" if atype == "object"  else \
                      "pred_act"
                n = res.get(key, 0)
                row += f"  {n:>{N_W}} {'N/A':>{M_W}} {'N/A':>{M_W}} {'N/A':>{M_W}}"
            else:
                n = stat["gold_n"]
                P, R, F1 = fmt_prf(stat)
                row += f"  {n:>{N_W}} {P:>{M_W}} {R:>{M_W}} {F1:>{M_W}}"
        print(row)

    print("  " + "=" * (dashes - 2))
    print()

    # Detailed breakdown
    print("  DETAIL breakdown:")
    print(f"  {'Dataset':<22}  {'Type':<20}  {'Gold#':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'P':>8}  {'R':>8}  {'F1':>8}")
    print("  " + "-" * 95)
    for ds_label, res in [("iTrust", itrust), ("Collected KMA_ACP", kma)]:
        for lbl, atype in COL_TYPES:
            stat = res[atype]
            if stat is None:
                print(f"  {ds_label:<22}  {lbl:<20}  {'N/A':>6}  {'N/A':>5}  {'N/A':>5}  {'N/A':>5}  {'N/A':>8}  {'N/A':>8}  {'N/A':>8}")
            else:
                P, R, F1 = prf(stat["tp"], stat["fp"], stat["fn"])
                print(f"  {ds_label:<22}  {lbl:<20}  {stat['gold_n']:>6}  "
                      f"{stat['tp']:>5}  {stat['fp']:>5}  {stat['fn']:>5}  "
                      f"{P:>8.4f}  {R:>8.4f}  {F1:>8.4f}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="P/R/F1 evaluation from txt files")
    parser.add_argument("--mode", default="partial", choices=["exact", "partial", "overlap"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n+" + "=" * 70 + "+")
    print("|" + "  NLACP — Evaluation from iTrust_gold.txt + VACT_ACP.txt".center(70) + "|")
    print("+" + "=" * 70 + "+\n")

    # Load data
    itrust_sents = load_txt(ITRUST_TXT)
    vact_sents   = load_txt(VACT_TXT)

    itrust_gold  = load_manual_gold(ITRUST_GOLD_JSON)
    vact_gold    = load_manual_gold(VACT_GOLD_JSON)

    print(f"[INFO] Loaded {len(itrust_sents)} iTrust sentences from {os.path.basename(ITRUST_TXT)}")
    print(f"[INFO] Loaded {len(vact_sents)} KMA_ACP sentences from {os.path.basename(VACT_TXT)}")
    print(f"[INFO] iTrust manual gold annotations: {len(itrust_gold)} entries")
    print(f"[INFO] VACT manual gold annotations: {len(vact_gold)} entries")
    print()

    # Run evaluation
    print("--- Evaluating iTrust ---")
    itrust = evaluate_dataset("iTrust", itrust_sents, itrust_gold, args.mode, args.verbose)

    print("--- Evaluating Collected KMA_ACP ---")
    kma = evaluate_dataset("Collected_KMA_ACP", vact_sents, vact_gold, args.mode, args.verbose)

    # Print results
    print_table(itrust, kma, args.mode)


if __name__ == "__main__":
    main()
