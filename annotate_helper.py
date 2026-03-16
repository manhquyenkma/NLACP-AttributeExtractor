"""
annotate_helper.py — Công cụ hỗ trợ annotate câu NLACP nhanh

Chức năng:
  1. Nhập câu tiếng Anh
  2. Tự động parse bằng spaCy → gợi ý subject / action / object / attributes
  3. Hỏi bạn xác nhận / sửa từng phần
  4. Lưu vào dataset/annotated_corpus.json (format cho CNN sau này)

Chạy: python annotate_helper.py
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from relation_candidate import parse_sentence, extract_relations

BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH     = os.path.join(BASE_DIR, "dataset", "annotated_corpus.json")
ANNOTATOR_NAME  = "PAV"   # ← ĐỔI TÊN BẠN Ở ĐÂY


def load_corpus():
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return {"_meta": {"target": 851, "current": 0}, "policies": []}


def save_corpus(data):
    data["_meta"]["current"] = len(data["policies"])
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def prompt(msg, default=None):
    """Hỏi user, nếu Enter thì dùng default."""
    suffix = f" [{default}]" if default else ""
    ans = input(f"  {msg}{suffix}: ").strip()
    return ans if ans else default


def annotate_sentence(sentence, corpus):
    """Tự động gợi ý + hỏi xác nhận để annotate một câu."""
    # Auto-suggest bằng NLP
    tokens = parse_sentence(sentence)
    auto   = extract_relations(sentence, tokens)

    print(f"\n  --- Auto-suggest ---")
    print(f"  Subject : {auto['subject']}")
    print(f"  Action  : {auto['action']}")
    print(f"  Object  : {auto['object']}")
    print(f"  Attrs   :")
    for a in auto["attributes"]:
        print(f"    [{a['category']}] {a['name']!r} → {a['value']!r}  (dep:{a['dep']})")

    print()
    # Xác nhận hoặc sửa
    subject = prompt("Subject (Enter = giữ nguyên)", auto["subject"])
    action  = prompt("Action  (Enter = giữ nguyên)", auto["action"])
    obj     = prompt("Object  (Enter = giữ nguyên)", auto["object"])

    print("\n  Subject attributes (nhập dạng: value1,value2 hoặc Enter để dùng auto):")
    subj_auto = [a["value"] for a in auto["attributes"] if a["category"] == "subject"]
    subj_raw  = prompt(f"  subject_attrs", ",".join(subj_auto) if subj_auto else "")
    subj_vals = [v.strip() for v in subj_raw.split(",") if v.strip()] if subj_raw else []

    print("  Object attributes:")
    obj_auto = [a["value"] for a in auto["attributes"] if a["category"] == "object"]
    obj_raw  = prompt(f"  object_attrs", ",".join(obj_auto) if obj_auto else "")
    obj_vals = [v.strip() for v in obj_raw.split(",") if v.strip()] if obj_raw else []

    new_id = len(corpus["policies"]) + 1

    entry = {
        "id":        f"NLACP_{new_id:04d}",
        "source":    prompt("Source (iTrust/ibm/cyberchair/manual)", "manual"),
        "annotator": ANNOTATOR_NAME,
        "sentence":  sentence,
        "subject":   subject,
        "action":    action,
        "object":    obj,
        "subject_attributes": [{"value": v, "valid": True} for v in subj_vals],
        "object_attributes":  [{"value": v, "valid": True} for v in obj_vals]
    }

    corpus["policies"].append(entry)
    save_corpus(corpus)
    print(f"\n  [OK] Saved as {entry['id']}. Total: {len(corpus['policies'])}/851\n")
    return entry


def main():
    corpus = load_corpus()
    current = len(corpus["policies"])

    print("\n" + "="*55)
    print("  ABAC Annotation Helper")
    print("  Tiến độ: {}/{} câu".format(current, 851))
    print("="*55)
    print("\nGõ 'exit' để thoát, 'skip' để bỏ qua câu hiện tại.\n")

    while True:
        try:
            sentence = input("Enter sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not sentence:
            continue
        if sentence.lower() == "exit":
            break
        if sentence.lower() == "skip":
            continue

        # Kiểm tra trùng
        existing = [p["sentence"] for p in corpus["policies"]]
        if sentence in existing:
            print("  [WARN] Câu này đã có trong corpus — bỏ qua.\n")
            continue

        try:
            annotate_sentence(sentence, corpus)
        except Exception as e:
            print(f"  [ERROR] {e}\n")

    total = len(corpus["policies"])
    print(f"\nKết thúc. Tổng: {total}/851 câu đã annotate.")
    print(f"File: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
