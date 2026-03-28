#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
evaluate_full_table.py — Full 4-type attribute evaluation table
================================================================

TABLE:
  Dataset | #SA  P  R  F1 | #OA  P  R  F1 | #CA  P  R  F1 | #Act P  R  F1

GOLD SOURCES:
  - iTrust (t2p+acre in annotation_sheet.csv):
      • Context: annotation_sheet.csv temporal_final / spatial_final
      • Subject/Object/Actions: NOT annotated → Gold=0, shown as N/A
  - Collected KMA_ACP (vact+collected+self_created in policy_dataset.json):
      • Subject/Object/Actions: policy_dataset.json attributes[].category
      • Context: annotation_llm_gold.csv temporal_final / spatial_final
        (fallback: policy_dataset.json environment[])

PREDICTOR:
  - Subject/Object/Actions: extract_relations()
  - Context:                extract_env_attributes()

RUN:
    python scripts/evaluate_full_table.py
    python scripts/evaluate_full_table.py --mode exact
    python scripts/evaluate_full_table.py --verbose
"""

import json, os, re, csv, argparse
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.extraction.relation_candidate import extract_relations
from nlacp.extraction.env_extractor     import extract_env_attributes
from nlacp.paths import POLICY_DATASET_PATH

ANNOTATION_SHEET_PATH = os.path.join(PROJECT_ROOT, "dataset", "annotation_sheet.csv")
LLM_GOLD_PATH         = os.path.join(PROJECT_ROOT, "dataset", "annotation_llm_gold.csv")
ANNOTATED_DIR         = os.path.join(PROJECT_ROOT, "dataset", "annotated")

ITRUST_SOURCES    = {"t2p", "acre"}
KMA_SOURCES       = {"collected", "self_created", "vact"}

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _norm(text: str) -> str:
    text = text.lower().strip()
    return re.sub(r"^[^\w]+|[^\w]+$", "", text)

def _match(a: str, b: str, mode: str) -> bool:
    na, nb = _norm(a), _norm(b)
    if mode == "exact":
        return na == nb
    elif mode == "partial":
        return na.split()[:2] == nb.split()[:2]
    else:  # overlap (Jaccard >= 0.5)
        sa, sb = set(na.split()), set(nb.split())
        if not sa and not sb: return True
        return len(sa & sb) / len(sa | sb) >= 0.5 if (sa | sb) else False

def match_lists(gold: list, pred: list, mode: str):
    """Greedy match → (tp, fp, fn)."""
    matched_g, matched_p = set(), set()
    for pi, p in enumerate(pred):
        for gi, g in enumerate(gold):
            if gi in matched_g: continue
            if _match(p, g, mode):
                matched_p.add(pi); matched_g.add(gi); break
    tp = len(matched_p)
    return tp, len(pred) - tp, len(gold) - len(matched_g)

def prf(tp, fp, fn):
    P  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return round(P, 4), round(R, 4), round(F1, 4)

# ─── Loaders ─────────────────────────────────────────────────────────────────

def load_annotation_sheet(path):
    """Returns dict[sentence → {source, temporal, spatial}]."""
    result = {}
    if not os.path.exists(path): return result
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sent = row.get("Sentence", "").strip()
            if not sent: continue
            src  = row.get("Source", "").strip().lower()
            tf   = _norm(row.get("temporal_final", "").strip())
            sf   = _norm(row.get("spatial_final", "").strip())
            ctx  = [v for v in [tf, sf] if v]
            result[sent] = {"source": src, "context_gold": ctx}
    return result

def load_llm_gold(path):
    """Returns dict[sentence → [ctx_gold_strings]]."""
    result = {}
    if not os.path.exists(path): return result
    with open(path, encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            sent = row.get("Sentence", "").strip()
            if not sent: continue
            tf   = _norm(row.get("temporal_final", "").strip())
            sf   = _norm(row.get("spatial_final", "").strip())
            ctx  = [v for v in [tf, sf] if v]
            if sent not in result:
                result[sent] = []
            result[sent].extend(ctx)
    return result

def predict_for_sentence(sent, mode, verbose=False):
    """Returns dict {subj_attrs, obj_attrs, acts, ctx_attrs}."""
    # --- Relations ---
    try:
        rel         = extract_relations(sent, [])
        subj        = _norm(rel.get("subject") or "")
        obj         = _norm(rel.get("object")  or "")
        acts        = [_norm(a) for a in rel.get("actions", []) if a]
        rel_attrs   = rel.get("attributes", [])

        def _extract_role_attrs(role_text, attrs):
            """
            Trích xuất attributes mô tả một role.
            Thử match nsubj/dobj, sau đó fallback: toàn bộ role text.
            """
            out = []
            for a in attrs:
                av = _norm(a.get("value", "") or a.get("text", ""))
                if av and _norm(role_text) == av:
                    name = _norm(a.get("name", "") or a.get("attr_type", ""))
                    comb = (name + " " + av).strip()
                    out.append(comb if comb else av)
            if not out and role_text:
                out = [role_text]
            return out

        pred_subj_attrs = _extract_role_attrs(subj, rel_attrs)
        pred_obj_attrs  = _extract_role_attrs(obj,  rel_attrs)
    except Exception as e:
        if verbose: print(f"  [ERR relations] {e}")
        pred_subj_attrs, pred_obj_attrs, acts = [], [], []

    # --- Context (env) ---
    try:
        pred_env = extract_env_attributes(sent)
        pred_ctx = [_norm(e.get("value", "")) for e in pred_env if e.get("value")]
    except Exception as e:
        if verbose: print(f"  [ERR env] {e}")
        pred_ctx = []

    return {
        "subj": pred_subj_attrs,
        "obj":  pred_obj_attrs,
        "acts": acts,
        "ctx":  pred_ctx,
    }

# ─── Evaluation per dataset ───────────────────────────────────────────────────

def empty_stat():
    return {"tp": 0, "fp": 0, "fn": 0, "gold_n": 0, "pred_n": 0}

def accumulate(stat, gold, pred, mode):
    tp, fp, fn = match_lists(gold, pred, mode)
    stat["tp"]     += tp
    stat["fp"]     += fp
    stat["fn"]     += fn
    stat["gold_n"] += len(gold)
    stat["pred_n"] += len(pred)


def evaluate_itrust(ann_sheet, mode, verbose):
    """Evaluate iTrust: only Context has gold labels."""
    itrust_sents = {s: v for s, v in ann_sheet.items()
                    if v["source"] in ITRUST_SOURCES}
    print(f"[INFO] iTrust: {len(itrust_sents)} sentences (t2p+acre)")

    ctx_stat = empty_stat()
    # subject/object/actions → no gold, count preds only
    sa_pred = 0; oa_pred = 0; act_pred = 0

    for sent, info in itrust_sents.items():
        gold_ctx = info["context_gold"]
        pred = predict_for_sentence(sent, mode, verbose)

        accumulate(ctx_stat, gold_ctx, pred["ctx"], mode)
        sa_pred  += len(pred["subj"])
        oa_pred  += len(pred["obj"])
        act_pred += len(pred["acts"])

    return {
        "n_sents": len(itrust_sents),
        "subject": None,   # No gold → N/A
        "object":  None,
        "context": ctx_stat,
        "actions": None,
        "pred_sa": sa_pred,
        "pred_oa": oa_pred,
        "pred_act": act_pred,
    }


def evaluate_kma(policy_data, llm_gold, mode, verbose):
    """Evaluate Collected KMA_ACP: all 4 types have gold labels."""
    policies = policy_data.get("policies", [])

    sa_stat  = empty_stat()
    oa_stat  = empty_stat()
    ctx_stat = empty_stat()
    act_stat = empty_stat()

    n_sents = 0
    for pol in policies:
        sent = pol.get("sentence", "").strip()
        if not sent: continue
        n_sents += 1

        attrs      = pol.get("attributes", [])
        gold_subj  = [_norm((a.get("name","")+" "+a.get("value","")).strip())
                      for a in attrs if a.get("category") == "subject"]
        gold_obj   = [_norm((a.get("name","")+" "+a.get("value","")).strip())
                      for a in attrs if a.get("category") == "object"]
        gold_acts  = [_norm(a) for a in pol.get("actions", []) if a]

        # Context gold: llm_gold CSV ưu tiên; fallback environment[]
        if sent in llm_gold and llm_gold[sent]:
            gold_ctx = llm_gold[sent]
        else:
            gold_ctx = []
            for env in pol.get("environment", []):
                val = env.get("full_value") or env.get("normalized", "")
                if val: gold_ctx.append(_norm(val))

        pred = predict_for_sentence(sent, mode, verbose)

        accumulate(sa_stat,  gold_subj, pred["subj"], mode)
        accumulate(oa_stat,  gold_obj,  pred["obj"],  mode)
        accumulate(ctx_stat, gold_ctx,  pred["ctx"],  mode)
        accumulate(act_stat, gold_acts, pred["acts"],  mode)

    print(f"[INFO] Collected_KMA_ACP: {n_sents} sentences")
    return {
        "n_sents": n_sents,
        "subject": sa_stat,
        "object":  oa_stat,
        "context": ctx_stat,
        "actions": act_stat,
    }

# ─── Print table ─────────────────────────────────────────────────────────────

def fmt_prf(stat):
    """Format P / R / F1 from stat dict, or N/A if stat is None."""
    if stat is None:
        return "  N/A ", "  N/A ", "  N/A "
    P, R, F1 = prf(stat["tp"], stat["fp"], stat["fn"])
    return f"{P:.4f}", f"{R:.4f}", f"{F1:.4f}"

def print_table(itrust, kma, mode):
    COL_TYPES = [
        ("Subject-attribute", "subject"),
        ("Object-attribute",  "object"),
        ("Context-attribute", "context"),
        ("Actions' Names",    "actions"),
    ]
    DS_W = 22
    N_W  = 6
    M_W  = 7

    dashes = DS_W + (N_W + 3*M_W + 6) * 4 + 3

    print()
    print("=" * dashes)
    print(f"  NLACP ATTRIBUTE EXTRACTION — EVALUATION TABLE  (mode={mode})")
    print("=" * dashes)

    # Header
    h1 = f"  {'Dataset':<{DS_W}}"
    h2 = f"  {'':>{DS_W}}"
    for lbl, _ in COL_TYPES:
        w = N_W + 3*M_W + 5
        h1 += f"  {lbl:^{w}}"
        h2 += f"  {'#':>{N_W}} {'P':>{M_W}} {'R':>{M_W}} {'F1':>{M_W}}"
    print(h1)
    print(h2)
    print("  " + "-" * (dashes - 2))

    def row_str(ds_label, res):
        row = f"  {ds_label:<{DS_W}}"
        for _, atype in COL_TYPES:
            stat = res[atype]
            if stat is None:
                n = res.get(f"pred_{atype[0]}a", res.get("pred_act", 0))
                row += f"  {n:>{N_W}} {'N/A':>{M_W}} {'N/A':>{M_W}} {'N/A':>{M_W}}"
            else:
                n = stat["gold_n"]
                P, R, F1 = fmt_prf(stat)
                row += f"  {n:>{N_W}} {P:>{M_W}} {R:>{M_W}} {F1:>{M_W}}"
        return row

    print(row_str("iTrust", itrust))
    print(row_str("Collected KMA_ACP", kma))
    print("  " + "=" * (dashes - 2))
    print()

    # Detail
    print("  DETAIL breakdown:")
    print(f"  {'Dataset':<22}  {'Type':<20}  {'Gold#':>6}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'P':>7}  {'R':>7}  {'F1':>7}")
    print("  " + "-" * 90)
    for ds_label, res in [("iTrust", itrust), ("Collected KMA_ACP", kma)]:
        for lbl, atype in COL_TYPES:
            stat = res[atype]
            if stat is None:
                print(f"  {ds_label:<22}  {lbl:<20}  {'N/A':>6}  {'N/A':>5}  {'N/A':>5}  {'N/A':>5}  {'N/A':>7}  {'N/A':>7}  {'N/A':>7}")
            else:
                P, R, F1 = prf(stat["tp"], stat["fp"], stat["fn"])
                print(f"  {ds_label:<22}  {lbl:<20}  {stat['gold_n']:>6}  "
                      f"{stat['tp']:>5}  {stat['fp']:>5}  {stat['fn']:>5}  "
                      f"{P:>7.4f}  {R:>7.4f}  {F1:>7.4f}")
    print()


# ─── Main ─────────────────────────────────────────────────────────────────────

def run_evaluation(mode="partial", verbose=False):
    # Load
    ann_sheet  = load_annotation_sheet(ANNOTATION_SHEET_PATH)
    llm_gold   = load_llm_gold(LLM_GOLD_PATH)
    with open(POLICY_DATASET_PATH, encoding="utf-8") as f:
        policy_data = json.load(f)

    # Evaluate
    itrust = evaluate_itrust(ann_sheet, mode, verbose)
    kma    = evaluate_kma(policy_data, llm_gold, mode, verbose)

    # Print
    print_table(itrust, kma, mode)

    return {"iTrust": itrust, "Collected_KMA_ACP": kma}


def main():
    parser = argparse.ArgumentParser(
        description="Bảng đánh giá P/R/F1 cho Subject/Object/Context/Actions"
    )
    parser.add_argument("--mode", default="partial",
                        choices=["exact", "partial", "overlap"])
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    print("\n+" + "="*70 + "+")
    print("|" + "  NLACP — Full Attribute Evaluation Table".center(70) + "|")
    print("+" + "="*70 + "+")

    run_evaluation(mode=args.mode, verbose=args.verbose)


if __name__ == "__main__":
    main()
