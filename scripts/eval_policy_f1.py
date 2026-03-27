#!/usr/bin/env python
# -*- coding: utf-8 -*-
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
"""
eval_policy_f1.py -- Tinh F1 cho env-attribute extraction tren policy_dataset.json
==================================================================================

Logic:
  - Dùng `environment[]` đã có trong policy_dataset.json làm GOLD
    (các entry này đã được validate qua pipeline CNN yes/no)
  - Chạy lại extract_env_attributes() on-the-fly làm PREDICTED
  - Tính Precision, Recall, F1 theo mode exact / partial / overlap

Dùng:
    python scripts/eval_policy_f1.py
    python scripts/eval_policy_f1.py --mode exact
    python scripts/eval_policy_f1.py --mode partial --verbose
    python scripts/eval_policy_f1.py --csv          # dùng annotation_sheet.csv làm gold
"""

import json
import os
import sys
import re
import argparse
import csv
from collections import defaultdict

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.extraction.env_extractor import extract_env_attributes
from nlacp.paths import POLICY_DATASET_PATH

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _normalize(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"^[^\w]+|[^\w]+$", "", text)
    return text

def _tokens(text: str) -> set:
    return set(_normalize(text).split())

def _jaccard(a: str, b: str) -> float:
    sa, sb = _tokens(a), _tokens(b)
    if not sa and not sb:
        return 1.0
    inter = len(sa & sb)
    union = len(sa | sb)
    return inter / union if union > 0 else 0.0

# Map category của gold env → category của predicted
GOLD_TYPE_MAP = {
    "temporal":        "temporal",
    "spatial":         "spatial",
    "spatial_physical": "spatial",
    "spatial_network":  "spatial",
    "spatial_device":   "spatial",
    "situational":     "situational",
}


def env_to_comparable(env_entry: dict, source: str = "policy") -> dict:
    """Chuẩn hóa một env entry về dạng chung {category, value}."""
    if source == "policy":
        cat = GOLD_TYPE_MAP.get(env_entry.get("type", ""), "unknown")
        val = env_entry.get("full_value") or env_entry.get("normalized", "")
    else:  # from extractor output
        cat = env_entry.get("category", "unknown")
        val = env_entry.get("value", "")
    return {"category": cat, "value": _normalize(val)}


def match_pair(pred_cat, pred_val, gold_cat, gold_val, mode):
    if pred_cat != gold_cat:
        return False
    if mode == "exact":
        return _normalize(pred_val) == _normalize(gold_val)
    elif mode == "partial":
        pwords = _normalize(pred_val).split()[:2]
        gwords = _normalize(gold_val).split()[:2]
        return pwords == gwords
    else:  # overlap
        return _jaccard(pred_val, gold_val) >= 0.5


def compute_tp_fp_fn(gold_list, pred_list, mode):
    matched_g = set()
    matched_p = set()
    for pi, p in enumerate(pred_list):
        for gi, g in enumerate(gold_list):
            if gi in matched_g:
                continue
            if match_pair(p["category"], p["value"],
                          g["category"], g["value"], mode):
                matched_p.add(pi)
                matched_g.add(gi)
                break
    tp = len(matched_p)
    fp = len(pred_list) - tp
    fn = len(gold_list) - len(matched_g)
    return tp, fp, fn


def prf(tp, fp, fn):
    P  = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    R  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    F1 = 2 * P * R / (P + R) if (P + R) > 0 else 0.0
    return round(P, 4), round(R, 4), round(F1, 4)


# ─── Mode A: Dùng policy_dataset.json làm gold ──────────────────────────────

def evaluate_from_policy(policy_path: str, mode: str, verbose: bool):
    with open(policy_path, encoding="utf-8") as f:
        data = json.load(f)

    policies = data.get("policies", [])
    print(f"\n[INFO] Loaded {len(policies)} policies from policy_dataset.json")

    total_tp = total_fp = total_fn = 0
    macro_f1s = []
    cat_stats = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    empty_gold = 0
    mismatch_count = 0

    for policy in policies:
        sentence = policy.get("sentence", "")
        gold_raw = policy.get("environment", [])
        gold = [env_to_comparable(e, source="policy") for e in gold_raw]
        gold = [g for g in gold if g["value"]]  # bỏ empty

        pred_raw = extract_env_attributes(sentence)
        pred = [env_to_comparable(e, source="extractor") for e in pred_raw]

        if not gold:
            empty_gold += 1
            # Nếu gold rỗng mà pred có → FP
            if pred:
                total_fp += len(pred)
                for p in pred:
                    cat_stats[p["category"]]["fp"] += 1
            macro_f1s.append(0.0)
            continue

        tp, fp, fn = compute_tp_fp_fn(gold, pred, mode)
        total_tp += tp
        total_fp += fp
        total_fn += fn

        # Per-category
        for g in gold:
            c = g["category"]
            # estimate: check if each gold was matched
            cat_stats[c]["fn"] += 1  # tentative

        # Re-run for category breakdown
        for g in gold:
            g_cat_list = [x for x in gold if x["category"] == g["category"]]
            p_cat_list = [x for x in pred if x["category"] == g["category"]]
            # updated below

        _, _, f1_s = prf(tp, fp, fn)
        macro_f1s.append(f1_s)

        if verbose and (fp > 0 or fn > 0):
            mismatch_count += 1
            print(f"\n  [#{policy.get('id', '?')}] {sentence[:80]}")
            gold_vals = [(g['category'], g['value']) for g in gold]
            pred_vals = [(p['category'], p['value']) for p in pred]
            for g in gold:
                matched = any(match_pair(p['category'], p['value'],
                                          g['category'], g['value'], mode)
                              for p in pred)
                if not matched:
                    print(f"    FN: [{g['category']}] \"{g['value']}\"")
            for p in pred:
                matched = any(match_pair(p['category'], p['value'],
                                          g['category'], g['value'], mode)
                              for g in gold)
                if not matched:
                    print(f"    FP: [{p['category']}] \"{p['value']}\"")

    micro_P, micro_R, micro_F1 = prf(total_tp, total_fp, total_fn)
    macro_F1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0

    # Per-category breakdown (re-compute properly)
    cat_results = {}
    for cat in ["temporal", "spatial", "situational"]:
        c_tp = c_fp = c_fn = 0
        for policy in policies:
            sentence = policy.get("sentence", "")
            gold_raw = policy.get("environment", [])
            gold_all = [env_to_comparable(e, source="policy") for e in gold_raw]
            gold_c = [g for g in gold_all if g["category"] == cat and g["value"]]
            pred_raw = extract_env_attributes(sentence)
            pred_all = [env_to_comparable(e, source="extractor") for e in pred_raw]
            pred_c = [p for p in pred_all if p["category"] == cat]
            tp, fp, fn = compute_tp_fp_fn(gold_c, pred_c, mode)
            c_tp += tp; c_fp += fp; c_fn += fn
        cat_results[cat] = prf(c_tp, c_fp, c_fn) + (c_tp, c_fp, c_fn)

    # ── Print Report ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  F1 Evaluation - policy_dataset.json  (mode={mode})")
    print(f"  Total policies: {len(policies)}  |  Gold-empty: {empty_gold}")
    print(f"{'='*65}")
    print(f"\n  {'Category':12s}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*5} {'-'*5} {'-'*5}")

    for cat in ["temporal", "spatial", "situational"]:
        P, R, F1, tp, fp, fn = cat_results[cat]
        print(f"  {cat:12s}  {P:>10.4f}  {R:>8.4f}  {F1:>8.4f}  {tp:>5} {fp:>5} {fn:>5}")

    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*5} {'-'*5} {'-'*5}")
    print(f"  {'MICRO':12s}  {micro_P:>10.4f}  {micro_R:>8.4f}  {micro_F1:>8.4f}  "
          f"{total_tp:>5} {total_fp:>5} {total_fn:>5}")
    print(f"  {'MACRO-F1':12s}  {'':>10}  {'':>8}  {macro_F1:>8.4f}")
    print(f"\n{'═'*65}\n")

    return {
        "micro_P": micro_P, "micro_R": micro_R, "micro_F1": micro_F1,
        "macro_F1": round(macro_F1, 4),
        "tp": total_tp, "fp": total_fp, "fn": total_fn,
        "per_category": cat_results,
    }


# ─── Mode B: Dùng annotation_sheet.csv làm gold ─────────────────────────────

def evaluate_from_csv(csv_path: str, mode: str, verbose: bool):
    rows = []
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)

    # Chỉ lấy các row có temporal_final hoặc spatial_final không rỗng
    annotated = [r for r in rows
                 if r.get("temporal_final", "").strip() or r.get("spatial_final", "").strip()]

    print(f"\n[INFO] annotation_sheet.csv: {len(rows)} rows, "
          f"{len(annotated)} có annotation thực sự")

    total_tp = total_fp = total_fn = 0
    macro_f1s = []
    cat_results_raw = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for row in rows:
        sentence = row.get("Sentence", "").strip()
        if not sentence:
            continue

        gold = []
        t_final = row.get("temporal_final", "").strip()
        s_final = row.get("spatial_final", "").strip()
        if t_final:
            gold.append({"category": "temporal", "value": _normalize(t_final)})
        if s_final:
            gold.append({"category": "spatial",  "value": _normalize(s_final)})

        pred_raw = extract_env_attributes(sentence)
        pred = [{"category": p.get("category", ""), "value": _normalize(p.get("value", ""))}
                for p in pred_raw]

        tp, fp, fn = compute_tp_fp_fn(gold, pred, mode)
        total_tp += tp; total_fp += fp; total_fn += fn

        # Per-category
        for cat in ["temporal", "spatial"]:
            g_c = [g for g in gold if g["category"] == cat]
            p_c = [p for p in pred if p["category"] == cat]
            c_tp, c_fp, c_fn = compute_tp_fp_fn(g_c, p_c, mode)
            cat_results_raw[cat]["tp"] += c_tp
            cat_results_raw[cat]["fp"] += c_fp
            cat_results_raw[cat]["fn"] += c_fn

        _, _, f1_s = prf(tp, fp, fn)
        macro_f1s.append(f1_s)

        if verbose and (fp > 0 or fn > 0):
            rowid = row.get("ID", "?")
            print(f"\n  [{rowid}] {sentence[:80]}")
            for g in gold:
                matched = any(match_pair(p["category"], p["value"],
                                         g["category"], g["value"], mode)
                              for p in pred)
                if not matched:
                    print(f"    FN: [{g['category']}] \"{g['value']}\"")
            for p in pred:
                matched = any(match_pair(p["category"], p["value"],
                                          g["category"], g["value"], mode)
                              for g in gold)
                if not matched:
                    print(f"    FP: [{p['category']}] \"{p['value']}\"")

    micro_P, micro_R, micro_F1 = prf(total_tp, total_fp, total_fn)
    macro_F1 = sum(macro_f1s) / len(macro_f1s) if macro_f1s else 0.0

    print(f"\n{'='*65}")
    print(f"  F1 Evaluation - annotation_sheet.csv  (mode={mode})")
    print(f"  Total sentences: {len(rows)}  |  Annotated: {len(annotated)}")
    print(f"{'='*65}")
    print(f"\n  {'Category':12s}  {'Precision':>10}  {'Recall':>8}  {'F1':>8}  {'TP':>5} {'FP':>5} {'FN':>5}")
    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*5} {'-'*5} {'-'*5}")

    for cat in ["temporal", "spatial"]:
        cr = cat_results_raw[cat]
        P, R, F1 = prf(cr["tp"], cr["fp"], cr["fn"])
        print(f"  {cat:12s}  {P:>10.4f}  {R:>8.4f}  {F1:>8.4f}  "
              f"{cr['tp']:>5} {cr['fp']:>5} {cr['fn']:>5}")

    print(f"  {'-'*12}  {'-'*10}  {'-'*8}  {'-'*8}  {'-'*5} {'-'*5} {'-'*5}")
    print(f"  {'MICRO':12s}  {micro_P:>10.4f}  {micro_R:>8.4f}  {micro_F1:>8.4f}  "
          f"{total_tp:>5} {total_fp:>5} {total_fn:>5}")
    print(f"  {'MACRO-F1':12s}  {'':>10}  {'':>8}  {macro_F1:>8.4f}")
    print(f"\n{'='*65}\n")


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Tính F1 env-attribute extraction trên policy_dataset.json (380 câu)"
    )
    parser.add_argument("--mode", default="partial",
                        choices=["exact", "partial", "overlap"],
                        help="Match mode (default: partial)")
    parser.add_argument("--csv", action="store_true",
                        help="Dùng annotation_sheet.csv làm gold thay vì policy_dataset.json")
    parser.add_argument("--csv-path", default=None,
                        help="Path tới file CSV để thay thế annotation_sheet.csv mặc định")
    parser.add_argument("--verbose", action="store_true",
                        help="In chi tiết FP/FN từng câu")
    parser.add_argument("--policy-path", default=None,
                        help="Override đường dẫn policy_dataset.json")
    args = parser.parse_args()

    print("\n" + "+" + "="*63 + "+")
    print("|" + "  Policy Dataset F1 Evaluation".center(63) + "|")
    print("+" + "="*63 + "+")

    if args.csv:
        csv_path = args.csv_path or os.path.join(PROJECT_ROOT, "dataset", "annotation_sheet.csv")
        if not os.path.exists(csv_path):
            print(f"[ERROR] Không tìm thấy file: {csv_path}")
            return
        evaluate_from_csv(csv_path, args.mode, args.verbose)
    else:
        policy_path = args.policy_path or POLICY_DATASET_PATH
        if not os.path.exists(policy_path):
            print(f"[ERROR] Không tìm thấy: {policy_path}")
            return
        evaluate_from_policy(policy_path, args.mode, args.verbose)


if __name__ == "__main__":
    main()
