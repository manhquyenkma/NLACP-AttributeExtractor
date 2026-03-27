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
from nlacp.paths import POLICY_DATASET_PATH as POLICY_PATH


# =====================================================================
#  Deduplication Helpers
# =====================================================================

def _sentence_fingerprint(sentence: str) -> str:
    """
    Tạo fingerprint để detect duplicate:
    lowercase + bỏ stop words + sort tokens
    """
    STOP = {"a", "an", "the", "can", "may", "will", "shall"}
    tokens = [w.lower() for w in (sentence or "").split() 
              if w.lower() not in STOP]
    return " ".join(sorted(tokens))


def deduplicate_policies(policies):
    seen_fingerprints = {}
    result = []
    count = 0
    for p in policies:
        fp = _sentence_fingerprint(p.get("sentence", ""))
        if not fp:
            result.append(p)
            continue
        if fp not in seen_fingerprints:
            seen_fingerprints[fp] = p["id"]
            result.append(p)
        else:
            print(f"[DEDUP] Policy #{p['id']} trùng với #{seen_fingerprints[fp]}, bỏ qua.")
            count += 1
    if count > 0:
        print(f"[INFO] Loại bỏ {count} policy trùng lặp.")
    return result


# =====================================================================
#  STEP 1A: Extract relations & build relation_candidate.json
# =====================================================================

def run_extraction():
    """Nhap cau tu terminal, trich xuat relation_pairs, ghi ra JSON.
    Chi luu sentence + pairs. S/A/O se duoc trich xuat o Step 2."""
    print("\n" + "=" * 60)
    print("  STEP 1A: NLP Relation Extraction")
    print("=" * 60)
    print("\nNhap cau policy (tieng Anh). Go 'done' hoac 'exit' de ket thuc.")
    print("Vi du:")
    print("  A senior nurse can approve medical records during business hours.")
    print("  Managers in the finance department can approve expense reports.\n")

    relations = []
    
    # Doc ID hien tai neu file da ton tai
    start_idx = 0
    if os.path.exists(CANDIDATE_PATH):
        try:
            with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                ids = [r.get("id", 0) for r in old_data.get("relations", [])]
                start_idx = max(ids) if ids else 0
        except Exception:
            pass
            
    idx = start_idx

    while True:
        try:
            sentence = input("Enter policy sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not sentence or sentence.lower() in ("done", "exit"):
            break

        # Validate input: minimum 3 words, at least one alphabetic word
        words = sentence.split()
        if len(words) < 3:
            print("  [SKIP] Câu quá ngắn — cần ít nhất 3 từ (vd: 'Doctors can view records').")
            continue
        if not any(w.isalpha() for w in words):
            print("  [SKIP] Câu không hợp lệ — cần chứa chữ cái.")
            continue

        idx += 1
        tokens  = parse_sentence(sentence)
        result  = extract_relations(sentence, tokens)

        # Chi lay relation_pairs thuc su, loai bo pairs la prep
        ENV_PREPS = {"during","within","after","before","between","via","through","using","at","on","from"}
        relation_pairs = []
        for attr in result.get("attributes", []):
            if attr["name"].lower() in ENV_PREPS:
                continue
            pair = [attr["value"], attr["name"]]
            if pair not in relation_pairs:
                relation_pairs.append(pair)

        entry = {
            "id":              idx,
            "sentence":        result["sentence"],
            "subject":         result.get("subject"),
            "actions":         result.get("actions", []),
            "object":          result.get("object"),
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

    # Doc du lieu cu hoac tao moi (relation_candidate.json)
    os.makedirs(DATASET_DIR, exist_ok=True)
    all_relations = []
    if os.path.exists(CANDIDATE_PATH):
        try:
            with open(CANDIDATE_PATH, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                all_relations = old_data.get("relations", [])
        except Exception:
            pass
            
    all_relations.extend(relations)
    
    # Ghi de (data cu + data moi vao file)
    data = {"relations": all_relations}
    with open(CANDIDATE_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"\n[OK] Saved {len(relations)} relation(s) to {CANDIDATE_PATH}")
    return True


# =====================================================================
#  STEP 1B: CNN Validator — interactive pair validation
# =====================================================================

def run_validation():
    """Doc relation_candidate.json, hien tung cap de user xac nhan dung/sai.
    Ghi ra policy_dataset.json CHI voi sentence + validated pairs.
    S/A/O va attributes se duoc dien o Step 2 (ABAC_extraction.py)."""

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

    # Read processed IDs from policy_dataset.json to skip them
    processed_ids = set()
    all_policies_existing = []
    if os.path.exists(POLICY_PATH):
        try:
            with open(POLICY_PATH, "r", encoding="utf-8") as f:
                old_data = json.load(f)
                all_policies_existing = old_data.get("policies", [])
                for p in all_policies_existing:
                    processed_ids.add(p.get("id"))
        except Exception:
            pass

    # Filter out already validated relations
    relations_to_validate = [r for r in relations if r["id"] not in processed_ids]

    if not relations_to_validate:
        print("[INFO] Khong co relation nao moi can xac nhan (tat ca da duoc xu ly roi).")
        return True

    print(f"\nCo {len(relations_to_validate)} relation(s) MOI can xac nhan (da bo qua {len(relations) - len(relations_to_validate)} relation cu).")
    print("Voi moi cap relation_pair, nhap:")
    print("  y = dung (giu lai)")
    print("  n = sai  (loai bo)")
    print("  a = chap nhan tat ca cap con lai cua relation nay")
    print("  s = bo qua tat ca cap con lai cua relation nay (ghi lai voi pairs rong)\n")

    new_policies = []

    for rel in relations_to_validate:
        print("-" * 50)
        print(f"Relation #{rel['id']}: {rel['sentence']}")

        pairs = rel.get("relation_pairs", [])
        print(f"  Relation pairs ({len(pairs)}):")

        valid_pairs = []
        auto_mode = None

        ENV_PREPS = {"during","within","after","before","between","via","through","using","at","on","from"}
        for i, pair in enumerate(pairs):
            if pair[1].lower() in ENV_PREPS:
                continue
                
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
                    choice = "y"

            if choice == "y":
                valid_pairs.append(pair)

        # Luôn ghi lại relation — kể cả khi không có pair nào được chọn
        # (đảm bảo bản ghi tồn tại trong policy_dataset.json)
        policy = {
            "id":              rel["id"],
            "sentence":        rel["sentence"],
            "subject":         rel.get("subject"),
            "actions":         rel.get("actions", []),
            "object":          rel.get("object"),
            "relation_pairs":  valid_pairs,
            "validated":       True,
        }
        new_policies.append(policy)
        print(f"  -> Giu lai {len(valid_pairs)}/{len(pairs)} cap.\n")

    # Ghi them policy_dataset.json:
    # - Giu nguyen tat ca ban ghi cu (ke ca fields do Step 2 da dien: environment, attributes)
    # - Neu ID moi trung voi ban ghi cu, MERGE thay vi ghi de hoan toan
    os.makedirs(os.path.dirname(POLICY_PATH), exist_ok=True)

    # Build dict {id -> existing_policy} de merge
    existing_by_id = {p["id"]: p for p in all_policies_existing}

    merged_new = []
    for new_p in new_policies:
        pid = new_p["id"]
        if pid in existing_by_id:
            # Merge: giu lai tat ca fields cu, chi cap nhat relation_pairs + validated
            merged = dict(existing_by_id[pid])
            merged["relation_pairs"] = new_p["relation_pairs"]
            merged["validated"]      = True
            merged_new.append(merged)
        else:
            merged_new.append(new_p)

    new_ids       = {p["id"] for p in merged_new}
    new_sentences = {p["sentence"].lower().strip() for p in merged_new}

    # Giu tat ca ban ghi cu khong bi trung ID/sentence
    kept_old = [p for p in all_policies_existing
                if p.get("id") not in new_ids
                and p.get("sentence", "").lower().strip() not in new_sentences]
    
    all_policies = deduplicate_policies(kept_old + merged_new)

    output = {"policies": all_policies}
    with open(POLICY_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=4, ensure_ascii=False)
    print(f"[OK] Saved {len(merged_new)} new/updated policy(ies) to {POLICY_PATH}")
    print(f"[OK] Total policies in file: {len(all_policies)} (including {len(kept_old)} kept from before)")
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
