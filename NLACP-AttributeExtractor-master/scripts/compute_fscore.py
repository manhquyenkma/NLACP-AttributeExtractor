#!/usr/bin/env python
"""
scripts/compute_fscore.py — Tính F-score cho toàn bộ ABAC pipeline

So sánh:
  policy_dataset.json      — RAW machine output (từ auto_extraction.py)
  vact_gold.json — Gold Standard (từ interactive_verification.py)

Tính P, R, F1 cho từng trường:
  - Subject
  - Actions
  - Object
  - Environment

Usage:
    python scripts/compute_fscore.py
    python scripts/compute_fscore.py --verbose        # In chi tiết từng câu sai
    python scripts/compute_fscore.py --field subject  # Chỉ tính 1 trường
    python scripts/compute_fscore.py \\
        --machine outputs/policies/policy_dataset.json \\
        --gold    outputs/policies/vact_gold.json
"""
import json
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.paths import POLICY_DATASET_PATH, POLICY_DATASET_GOLD_PATH


# =====================================================================
#  Load
# =====================================================================

def load_json_policies(path: str, label: str) -> dict:
    """Load file JSON policies, trả về dict id → policy."""
    if not os.path.exists(path):
        print(f"[ERROR] Không tìm thấy file {label}: {path}")
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    policies = data.get("policies", [])
    return {p["id"]: p for p in policies}


# =====================================================================
#  Normalise helpers
# =====================================================================

def _norm_str(s) -> str:
    """Chuẩn hoá 1 chuỗi: lower, strip, bỏ khoảng trắng thừa, underscore → space."""
    if s is None:
        return ""
    if isinstance(s, list):
        return " ".join(s).lower().strip().replace("_", " ")
    return str(s).lower().strip().replace("_", " ")


def _to_set(value) -> set:
    """Chuyển 1 giá trị (str, list, None) thành set các token chuẩn hoá."""
    if value is None:
        return set()
    if isinstance(value, list):
        return {_norm_str(v) for v in value if v}
    val = _norm_str(value)
    return {val} if val else set()


def _env_key(env: dict) -> str:
    """
    Tạo khoá so sánh cho một environment entry.
    Dùng (2-3 từ đầu của full_value) để khớp linh hoạt (partial match).
    """
    fv = env.get("full_value", "")
    trigger = env.get("trigger", "").lower()
    if trigger not in ("except", "except for") and (env.get("env_type") == "conditional" or env.get("category") == "environment" and env.get("subcategory") == "conditional_clause"):
        fv = env.get("phrase", "")
    
    if not fv:
        tw = env.get("trigger_word", env.get("trigger", ""))
        tp = env.get("trigger_phrase", env.get("phrase", ""))
        fv = f"{tw} {tp}".strip()
    
    # Workaround: sometimes full_value is present but misses trigger_word (from att_extractor)
    if env.get("env_type") != "conditional" and "trigger_word" in env and env["trigger_word"] and not fv.startswith(env["trigger_word"]):
        fv = f"{env['trigger_word']} {fv}".strip()
        
    fv = _norm_str(fv)
    words = fv.split()[:3]          # 3 từ đầu (partial match)
    return f"{' '.join(words)}"


# =====================================================================
#  Tính TP / FP / FN cho từng trường
# =====================================================================

def _compute_tp_fp_fn_sets(gold_set: set, pred_set: set, label=""):
    tp = len(gold_set & pred_set)
    fp = len(pred_set - gold_set)
    fn = len(gold_set - pred_set)
    if label == "SUBJ_DEBUG":
        print(f"SUBJ_DEBUG: gold={gold_set}, pred={pred_set} -> tp={tp}")
    return tp, fp, fn


def prf(tp: int, fp: int, fn: int):
    P  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return P, R, F1


# =====================================================================
#  Đánh giá từng câu
# =====================================================================

import spacy
_nlp_eval = None
def _get_eval_nlp():
    global _nlp_eval
    if _nlp_eval is None:
        try:
            _nlp_eval = spacy.load("en_core_web_sm")
        except OSError:
            _nlp_eval = spacy.blank("en")
    return _nlp_eval

def _lemmatize_phrase(phrase: str) -> str:
    if not phrase:
        return ""
    doc = _get_eval_nlp()(phrase)
    return " ".join([t.lemma_ for t in doc])

def _compute_tp_fp_fn_object(gold_set: set, pred_set: set):
    """
    Fix 3B: Object label comparison with lemmatization matching.
    is_match = (pred == gold) or (pred_lemma == gold_lemma)
    """
    tp = 0
    matched_preds = set()
    matched_golds = set()
    
    for p in pred_set:
        p_lem = _lemmatize_phrase(p)
        for g in gold_set:
            g_lem = _lemmatize_phrase(g)
            if p == g or p_lem == g_lem:
                tp += 1
                matched_preds.add(p)
                matched_golds.add(g)
                break
                
    fp = len(pred_set - matched_preds)
    fn = len(gold_set - matched_golds)
    return tp, fp, fn


def evaluate_pair(gold: dict, pred: dict) -> dict:
    """
    So sánh 1 cặp (gold, pred) cho 4 trường.
    Trả về dict với TP/FP/FN cho mỗi trường.
    """
    result = {}
    # ── Subject ────────────────────────────────────────────────────
    # Handle both old 'subject' (list of str) and new 'subjects' (list of dict)
    def _extract_subject_strings(policy):
        if "subjects" in policy and policy["subjects"]:
            roles = set()
            for s in policy["subjects"]:
                if isinstance(s, dict):
                    r = s.get("role", "")
                    if r:
                        roles.add(_norm_str(r))
                    elif s.get("ref_tokens"):
                        for r in s.get("ref_tokens"):
                            roles.add(_norm_str(r))
                elif isinstance(s, str):
                    roles.add(_norm_str(s))
            if roles:
                return roles
        # Fallback to flat 'subject' field
        raw = policy.get("subject", [])
        if isinstance(raw, list):
            return {_norm_str(v) for v in raw if v}
        return {_norm_str(raw)}

    g_sub = _extract_subject_strings(gold)
    p_sub = _extract_subject_strings(pred)
    result["subject"] = _compute_tp_fp_fn_sets(g_sub, p_sub, label="SUBJ_DEBUG")

    # ── Actions ────────────────────────────────────────────────────
    def _extract_action_strings(policy):
        acts = set()
        for a in policy.get("actions", []):
            if isinstance(a, dict):
                verb = str(a.get("verb", "")).lower().strip()
                # Only add verb — NOT operation (CRUD label like READ/EXECUTE).
                # Gold uses raw verbs; adding operation causes FP inflation.
                if verb:
                    acts.add(verb)
            else:
                acts.add(_norm_str(a))
        return acts


    g_act = _extract_action_strings(gold)
    p_act = _extract_action_strings(pred)
    result["actions"] = _compute_tp_fp_fn_sets(g_act, p_act)

    # ── Object/Resource ────────────────────────────────────────────
    def _extract_object_strings(policy):
        # Priority 1: pydantic 'resource' + 'additional_resources' (prefer ref_tokens over verbose label)
        if "resource" in policy and policy["resource"]:
            labels = set()
            res = policy["resource"]
            rts = [_norm_str(r) for r in (res.get("ref_tokens") or []) if r]
            lbl = _norm_str(res.get("label", ""))
            # Use ref_tokens when available (short, consistent across datasets)
            # Use label only when ref_tokens is absent AND label is a short compound noun (≤3 words)
            if rts:
                for r in rts:
                    labels.add(r)
            elif lbl and len(lbl.split()) <= 3:
                labels.add(lbl)
            for r in policy.get("additional_resources", []) or []:
                rts2 = [_norm_str(rt) for rt in (r.get("ref_tokens") or []) if rt]
                lbl2 = _norm_str(r.get("label", ""))
                if rts2:
                    for rt in rts2:
                        labels.add(rt)
                elif lbl2 and len(lbl2.split()) <= 3:
                    labels.add(lbl2)
            if labels:
                return labels

        # Priority 2: flat 'object' field (iTrust gold, VACT gold, legacy format)
        flat_obj = policy.get("object")
        if flat_obj is not None:
            return _to_set(flat_obj)

        return set()


    g_obj = _extract_object_strings(gold)
    p_obj = _extract_object_strings(pred)
    result["object"] = _compute_tp_fp_fn_object(g_obj, p_obj)

    # ── Environment ────────────────────────────────────────────────
    # environments uses different format now
    def _extract_env_strings(policy):
        envs = set()
        for key in ["environments", "environment"]:
            if key in policy:
                for e in policy[key]:
                    if isinstance(e, dict):
                        envs.add(_env_key(e))
                    elif isinstance(e, str):
                        words = _norm_str(e).split()[:3]
                        envs.add(" ".join(words))
        return {e for e in envs if e}

    g_env = _extract_env_strings(gold)
    p_env = _extract_env_strings(pred)
    result["environment"] = _compute_tp_fp_fn_sets(g_env, p_env)

    return result


# =====================================================================
#  Verbose helpers
# =====================================================================

def _print_errors(sentence: str, field: str, gold_set: set, pred_set: set):
    fp_items = pred_set - gold_set
    fn_items = gold_set - pred_set
    if fp_items or fn_items:
        print(f"\n    [{field.upper()}] sentence: \"{sentence[:75]}\"")
        for k in sorted(fp_items):
            print(f"      FP (wrong): {k}")
        for k in sorted(fn_items):
            print(f"      FN (missing): {k}")


# =====================================================================
#  Main evaluate
# =====================================================================

FIELDS = ["subject", "actions", "object", "environment"]


def evaluate(machine_map: dict, gold_map: dict,
             field_filter: str = None, verbose: bool = False) -> dict:
    """
    So sánh machine vs gold, trả về dict kết quả theo từng field.
    """
    fields = [field_filter] if field_filter else FIELDS

    # Accumulate TP/FP/FN
    totals = {f: [0, 0, 0] for f in fields}   # [tp, fp, fn]

    common_ids = sorted(set(gold_map.keys()) & set(machine_map.keys()))
    only_gold  = set(gold_map.keys()) - set(machine_map.keys())
    only_pred  = set(machine_map.keys()) - set(gold_map.keys())

    if only_gold:
        print(f"[WARN] {len(only_gold)} id(s) có trong gold nhưng không có trong machine output — tính là FN.")
    if only_pred:
        print(f"[WARN] {len(only_pred)} id(s) có trong machine nhưng không có trong gold — bị bỏ qua.")

    for pid in common_ids:
        gold = gold_map[pid]
        pred = machine_map[pid]
        pair = evaluate_pair(gold, pred)

        for f in fields:
            tp, fp, fn = pair[f]
            totals[f][0] += tp
            totals[f][1] += fp
            totals[f][2] += fn

        if verbose:
            sentence = gold.get("sentence", "")
            # Subject verbose
            if "subject" in fields:
                g = _to_set(gold.get("subject"))
                p = _to_set(pred.get("subject"))
                _print_errors(sentence, "subject", g, p)
            # Actions verbose
            if "actions" in fields:
                def _ex_act(policy):
                    acts = set()
                    for a in policy.get("actions", []):
                        if isinstance(a, dict):
                            v = str(a.get("verb", "")).lower().strip()
                            o = str(a.get("operation", "")).lower().strip()
                            if v: acts.add(v)
                            if o: acts.add(o)
                        else:
                            acts.add(_norm_str(a))
                    return acts
                g = _ex_act(gold)
                p = _ex_act(pred)
                _print_errors(sentence, "actions", g, p)
            # Object verbose
            if "object" in fields:
                def _ex_obj(policy):
                    labels = set()
                    if "resource" in policy and policy["resource"]:
                        res = policy["resource"]
                        if res.get("ref_tokens"):
                            for r in res.get("ref_tokens"): labels.add(_norm_str(r))
                        else:
                            labels.add(_norm_str(res.get("label", str(res))))
                    for r in policy.get("additional_resources", []) or []:
                        if r.get("ref_tokens"):
                            for rt in r.get("ref_tokens"): labels.add(_norm_str(rt))
                        else:
                            labels.add(_norm_str(r.get("label", "")))
                    if not labels: return _to_set(policy.get("object"))
                    return {l for l in labels if l}
                g = _ex_obj(gold)
                p = _ex_obj(pred)
                _print_errors(sentence, "object", g, p)
            # Environment verbose
            if "environment" in fields:
                g = set()
                p = set()
                for key in ["environments", "environment"]:
                    if key in gold:
                        for e in gold[key]:
                            if isinstance(e, dict): g.add(_env_key(e))
                for key in ["environments", "environment"]:
                    if key in pred:
                        for e in pred[key]:
                            if isinstance(e, dict): p.add(_env_key(e))
                _print_errors(sentence, "environment", g, p)

    # Handle ids only in gold (count as FN for each field)
    for pid in only_gold:
        gold = gold_map[pid]
        for f in fields:
            if f == "subject":
                fn_count = len(_to_set(gold.get("subject")))
            elif f == "actions":
                fn_count = len({_norm_str(a) for a in gold.get("actions", []) if a})
            elif f == "object":
                fn_count = len(_to_set(gold.get("object")))
            else:  # environment
                fn_count = len(gold.get("environment", []))
            totals[f][2] += fn_count

    # Compute P/R/F1
    results = {}
    for f in fields:
        tp, fp, fn = totals[f]
        P, R, F1 = prf(tp, fp, fn)
        results[f] = {"P": P, "R": R, "F1": F1, "TP": tp, "FP": fp, "FN": fn}

    return results, len(common_ids)


# =====================================================================
#  Print results
# =====================================================================

def print_results(results: dict, n_sentences: int):
    print(f"\n{'='*62}")
    print(f"  F-SCORE RESULTS  ({n_sentences} sentences compared)")
    print(f"{'='*62}")
    print(f"  {'Field':<14}  {'P':>7}  {'R':>7}  {'F1':>7}   TP    FP    FN")
    print(f"  {'-'*56}")

    all_tp = all_fp = all_fn = 0
    for f, r in results.items():
        print(f"  {f.capitalize():<14}  {r['P']:>7.4f}  {r['R']:>7.4f}  {r['F1']:>7.4f}"
              f"   {r['TP']:>3}   {r['FP']:>3}   {r['FN']:>3}")
        all_tp += r["TP"]
        all_fp += r["FP"]
        all_fn += r["FN"]

    # Micro-average overall
    P, R, F1 = prf(all_tp, all_fp, all_fn)
    print(f"  {'-'*56}")
    print(f"  {'OVERALL (micro)':<14}  {P:>7.4f}  {R:>7.4f}  {F1:>7.4f}"
          f"   {all_tp:>3}   {all_fp:>3}   {all_fn:>3}")
    print(f"{'='*62}\n")


# =====================================================================
#  Entry point
# =====================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Tính F-score: so sánh policy_dataset.json vs vact_gold.json"
    )
    parser.add_argument("--machine", default=POLICY_DATASET_PATH,
                        help="Path tới file machine output (mặc định: policy_dataset.json)")
    parser.add_argument("--gold",    default=POLICY_DATASET_GOLD_PATH,
                        help="Path tới Gold Standard (mặc định: vact_gold.json)")
    parser.add_argument("--field",   default=None,
                        choices=FIELDS,
                        help="Chỉ tính F-score cho 1 trường (subject/actions/object/environment)")
    parser.add_argument("--verbose", action="store_true",
                        help="In chi tiết các câu bị FP/FN")
    args = parser.parse_args()

    print(f"\n[INFO] Machine output : {args.machine}")
    print(f"[INFO] Gold Standard  : {args.gold}")
    if args.field:
        print(f"[INFO] Chỉ đánh giá   : {args.field}")

    machine_map = load_json_policies(args.machine, "machine output")
    gold_map    = load_json_policies(args.gold,    "Gold Standard")

    print(f"[INFO] Machine: {len(machine_map)} policies  |  Gold: {len(gold_map)} policies")

    results, n = evaluate(machine_map, gold_map,
                          field_filter=args.field,
                          verbose=args.verbose)
    print_results(results, n)


if __name__ == "__main__":
    main()
