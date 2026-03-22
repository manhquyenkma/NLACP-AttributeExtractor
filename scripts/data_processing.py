#!/usr/bin/env python
"""
data_processing.py — STEP 1: Relation Extraction + CNN Validation

Luong xu ly:
  1A. Nhap cau policy -> NLP extraction -> relation_candidate.json
  1B. CNN Validator: hien tung cap relation_pair -> user chon dung/sai
      -> policy_dataset.json (chi giu cap dung)

Usage:
    python data_processing.py
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.extraction.relation_candidate import extract_relations, parse_sentence

DATASET_DIR      = os.path.join(PROJECT_ROOT, "dataset")
CANDIDATE_PATH   = os.path.join(DATASET_DIR, "relation_candidate.json")
POLICY_PATH      = os.path.join(DATASET_DIR, "policy_dataset.json")


# =====================================================================
#  STEP 1A: Extract relations & build relation_candidate.json
# =====================================================================

def run_extraction():
    """Nhap cau tu terminal, trich xuat S-A-O + relation_pairs, ghi ra JSON."""
    print("\n" + "=" * 60)
    print("  STEP 1A: NLP Relation Extraction")
    print("=" * 60)
    print("\nNhap cau policy (tieng Anh). Go 'done' hoac 'exit' de ket thuc.")
    print("Vi du:")
    print("  A senior nurse can approve medical records during business hours.")
    print("  Managers in the finance department can approve expense reports.\n")

    relations = []
    idx = 0

    while True:
        try:
            sentence = input("Enter policy sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not sentence or sentence.lower() in ("done", "exit"):
            break

        idx += 1
        tokens  = parse_sentence(sentence)
        result  = extract_relations(sentence, tokens)

        # Tao relation_pairs tu attributes
        relation_pairs = []
        for attr in result.get("attributes", []):
            pair = [attr["value"], attr["name"]]
            if pair not in relation_pairs:
                relation_pairs.append(pair)

        entry = {
            "id":              idx,
            "sentence":        result["sentence"],
            "subject":         result.get("subject"),
            "actions":         result.get("actions", []),
            "object":          result.get("object"),
            "attributes":      result.get("attributes", []),
            "relation_pairs":  relation_pairs
        }
        relations.append(entry)

        print(f"  -> Extracted {len(relation_pairs)} relation pair(s)")
        for p in relation_pairs:
            print(f"     [{p[0]}, {p[1]}]")
        print()

    if not relations:
        print("[WARN] Khong co cau nao duoc nhap.")
        return False

    # Ghi de relation_candidate.json
    os.makedirs(DATASET_DIR, exist_ok=True)
    data = {"relations": relations}
    with open(CANDIDATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\n[OK] Saved {len(relations)} relation(s) to {CANDIDATE_PATH}")
    return True


# =====================================================================
#  STEP 1B: CNN Validator — interactive pair validation
# =====================================================================

def run_validation():
    """Doc relation_candidate.json, hien tung cap de user xac nhan dung/sai.
    Ghi ra policy_dataset.json chi voi cac cap dung."""

    print("\n" + "=" * 60)
    print("  STEP 1B: CNN Validation (Interactive)")
    print("=" * 60)

    if not os.path.exists(CANDIDATE_PATH):
        print(f"[ERROR] {CANDIDATE_PATH} khong ton tai. Chay Step 1A truoc.")
        return False

    with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    relations = data.get("relations", [])
    if not relations:
        print("[WARN] relation_candidate.json rong.")
        return False

    print(f"\nCo {len(relations)} relation(s) can xac nhan.")
    print("Voi moi cap relation_pair, nhap:")
    print("  y = dung (giu lai)")
    print("  n = sai  (loai bo)")
    print("  a = chap nhan tat ca cap con lai cua relation nay")
    print("  s = bo qua tat ca cap con lai cua relation nay\n")

    policies = []

    for rel in relations:
        print("-" * 50)
        print(f"Relation #{rel['id']}: {rel['sentence']}")
        print(f"  Subject: {rel.get('subject')}  |  Actions: {rel.get('actions')}  |  Object: {rel.get('object')}")
        print(f"  Relation pairs ({len(rel.get('relation_pairs', []))}):")

        valid_attrs = []
        auto_mode = None  # None, 'accept_all', 'skip_all'

        pairs = rel.get("relation_pairs", [])
        attrs = rel.get("attributes", [])

        # Map pair -> attribute dict
        pair_attr_map = {}
        for attr in attrs:
            key = (attr["value"], attr["name"])
            pair_attr_map[key] = attr

        for i, pair in enumerate(pairs):
            if auto_mode == "accept_all":
                choice = "y"
            elif auto_mode == "skip_all":
                choice = "n"
            else:
                print(f"\n    [{i+1}/{len(pairs)}]  {pair[0]} <-> {pair[1]}")
                try:
                    choice = input("    Dung hay sai? (y/n/a/s): ").strip().lower()
                except (EOFError, KeyboardInterrupt):
                    choice = "y"

                if choice == "a":
                    auto_mode = "accept_all"
                    choice = "y"
                elif choice == "s":
                    auto_mode = "skip_all"
                    choice = "n"
                elif choice not in ("y", "n"):
                    choice = "y"  # default accept

            if choice == "y":
                key = (pair[0], pair[1])
                attr = pair_attr_map.get(key)
                if attr:
                    valid_attrs.append(attr)
                else:
                    # Fallback: tao attr tu pair
                    valid_attrs.append({
                        "name":     pair[1],
                        "value":    pair[0],
                        "category": "unknown",
                        "dep":      "unknown"
                    })

        # Tao policy entry
        policy = {
            "id":              rel["id"],
            "sentence":        rel["sentence"],
            "subject":         rel.get("subject"),
            "actions":         rel.get("actions", []),
            "object":          rel.get("object"),
            "environment":     [],       # se duoc dien o Step 2
            "attributes":      valid_attrs,
            "relation_pairs":  [[a["value"], a["name"]] for a in valid_attrs]
        }
        policies.append(policy)
        print(f"  -> Giu lai {len(valid_attrs)}/{len(pairs)} cap.\n")

    # Ghi policy_dataset.json
    output = {"policies": policies}
    with open(POLICY_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"[OK] Saved {len(policies)} policy(ies) to {POLICY_PATH}")
    return True


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print("\n" + "=" * 60)
    print("  DATA PROCESSING — STEP 1")
    print("  Relation Extraction + CNN Validation")
    print("=" * 60)

    # Step 1A: Extraction
    ok = run_extraction()
    if not ok:
        print("\n[ERROR] Step 1A that bai.")
        return

    # Step 1B: Validation
    ok = run_validation()
    if not ok:
        print("\n[ERROR] Step 1B that bai.")
        return

    print("\n" + "=" * 60)
    print("  STEP 1 HOAN TAT!")
    print(f"  -> relation_candidate.json: {CANDIDATE_PATH}")
    print(f"  -> policy_dataset.json:     {POLICY_PATH}")
    print("=" * 60)
    print("\nTiep theo chay: python ABAC_extraction.py")


if __name__ == "__main__":
    main()
