"""
scripts/annotate.py — Công cụ hỗ trợ annotate câu NLACP nhanh (Cập nhật cho CNN)

Chức năng:
  1. Nhập câu tiếng Anh (hoặc đọc từ file tự động)
  2. Tự động sinh candidates bằng `generate_candidates`
  3. Hỏi người dùng xác nhận `valid: true/false` cho từng candidate
  4. Lưu vào outputs/policies/annotated_corpus.json 
     (sẵn sàng cho cnn_classifier.py)

Chạy: python scripts/annotate.py
"""
import sys
import os
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.extraction.relation_candidate import generate_candidates

BASE_DIR        = PROJECT_ROOT
OUTPUT_PATH     = os.path.join(BASE_DIR, "outputs", "policies", "annotated_corpus.json")
ANNOTATOR_NAME  = "PAV"   # ← ĐỔI TÊN BẠN Ở ĐÂY


def load_corpus():
    if os.path.exists(OUTPUT_PATH):
        with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    # Định dạng mới: sentences chứa các candidates
    return {"_meta": {"target": 851, "current": 0}, "sentences": []}


def save_corpus(data):
    data["_meta"]["current"] = len(data.get("sentences", []))
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def prompt(msg, default=None):
    """Hỏi user, nếu Enter thì dùng default."""
    suffix = f" [{default}]" if default else ""
    ans = input(f"  {msg}{suffix}: ").strip()
    return ans if ans else default


def annotate_sentence(sentence, corpus):
    """Tự động sinh candidates + hỏi xác nhận để annotate một câu."""
    # Sinh candidates (đúng + sai)
    data = generate_candidates(sentence)

    print(f"\n  --- Extract Info ---")
    print(f"  Subject : {data['subject']}")
    print(f"  Action  : {data['actions']}")
    print(f"  Object  : {data['object']}")

    print(f"\n  --- Candidate Verification ---")
    annotated_candidates = []
    
    for cand in data["candidates"]:
        elem = cand["element"]
        mod  = cand["modifier"]
        cat  = cand["category"]
        auto_valid = "y" if cand["valid"] else "n"
        
        print(f"  Candidate: [{cat}] '{mod}' -> '{elem}'")
        user_valid = prompt(f"    Valid relation? (y/n)", default=auto_valid).lower()
        
        cand["valid"] = (user_valid == "y" or user_valid == "yes")
        annotated_candidates.append(cand)

    data["candidates"] = annotated_candidates
    
    # Loại bỏ trường attributes raw không cần thiết cho CNN annotation
    if "attributes" in data:
        del data["attributes"]

    # Lưu vào corpus
    # Kiểm tra tránh trùng lặp
    sentences = corpus.setdefault("sentences", [])
    updated = False
    for i, s in enumerate(sentences):
        if s["sentence"] == sentence:
            sentences[i] = data
            updated = True
            break
            
    if not updated:
        sentences.append(data)
        
    save_corpus(corpus)
    print("\n[OK] Đã lưu vào annotated_corpus.json")


def main():
    print("=========================================================")
    print("  TRỢ LÝ ANNOTATE NLACP THÔNG MINH (Candidate Generator)")
    print(f"  Output: {OUTPUT_PATH}")
    print(f"  Annotator: {ANNOTATOR_NAME}")
    print("=========================================================\n")

    corpus = load_corpus()
    print(f"Đã load {corpus['_meta']['current']} câu đã annotate.")

    while True:
        print("\n" + "-"*50)
        sent = input("\nNhập câu policy (hoặc gõ 'exit' để thoát):\n> ").strip()
        
        if not sent:
            continue
        if sent.lower() in ("exit", "quit", "q"):
            print("Đang lưu và thoát...")
            break
            
        annotate_sentence(sent, corpus)


if __name__ == "__main__":
    main()
