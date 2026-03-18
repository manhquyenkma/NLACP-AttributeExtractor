"""
filter_env.py — Ngày 1 (7/3)
Lọc câu có env-att (temporal + spatial) từ dataset gốc.
Chạy: python data/filter_env.py
"""
import json
import os
import re

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── Trigger words ───────────────────────────────────────────────
TEMPORAL_TRIGGERS = [
    "during","between","after","before","within","throughout","until",
    "business hours","night shift","at night","working hours","daytime",
    "weekday","weekend","morning","evening","hour","shift","schedule",
    "only when","at the time","period","at night","on weekdays",
    "night time","day time","business day","office hours"
]

SPATIAL_TRIGGERS = [
    "from the","at the","within the","inside","outside",
    "on-site","remotely","network","ward","department",
    "building","internal","external","local","remote",
    "on premises","via","through the","using",
    "workstation","device","vpn","trusted","hospital network",
    "intranet","secure connection","on the","within the"
]


def filter_env_sentences(raw_file, dataset_name):
    """Lọc câu có env-att từ một file text."""
    with open(raw_file, "r", encoding="utf-8", errors="ignore") as f:
        sentences = [l.strip() for l in f if l.strip() and not l.startswith("#")]

    results = []
    for sent in sentences:
        low   = sent.lower()
        has_t = any(t in low for t in TEMPORAL_TRIGGERS)
        has_s = any(t in low for t in SPATIAL_TRIGGERS)
        results.append({
            "id":           len(results) + 1,
            "dataset":      dataset_name,
            "sentence":     sent,
            "has_temporal": has_t,
            "has_spatial":  has_s,
            "has_env":      has_t or has_s,
            "annotated":    False
        })

    env_only = [r for r in results if r["has_env"]]
    total    = len(sentences)

    print(f"\n  {dataset_name}:")
    print(f"    Tổng câu:        {total}")
    print(f"    Có temporal:     {sum(r['has_temporal'] for r in results)}")
    print(f"    Có spatial:      {sum(r['has_spatial']  for r in results)}")
    print(f"    Có env-att:      {len(env_only)} ({len(env_only)/max(total,1)*100:.1f}%)")
    return results, env_only


def filter_json_dataset(json_file, dataset_name, text_field="sentence"):
    """Lọc câu từ dataset đã ở dạng JSON (RAGent, LitroACP ...)"""
    with open(json_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Hỗ trợ list hoặc dict có key "data"/"sentences"/"policies"
    if isinstance(data, list):
        items = data
    else:
        items = data.get("data", data.get("sentences", data.get("policies", [])))

    sentences = []
    for item in items:
        if isinstance(item, str):
            sentences.append(item)
        elif isinstance(item, dict):
            text = item.get(text_field) or item.get("text") or item.get("sentence", "")
            if text:
                sentences.append(text.strip())

    # Viết ra file text tạm rồi filter
    tmp = os.path.join(BASE_DIR, "data", "raw", f"_tmp_{dataset_name}.txt")
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("\n".join(sentences))

    return filter_env_sentences(tmp, dataset_name)


def save_filtered(env_sentences, dataset_name):
    out_path = os.path.join(BASE_DIR, "data", "filtered",
                            f"{dataset_name}_env.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(env_sentences, f, indent=2, ensure_ascii=False)
    print(f"    Saved: {out_path}")
    return out_path


def main():
    print("\n" + "="*55)
    print("  Filter Script — Env-Att Sentences")
    print("  (7/3 — Ngày 1 lộ trình 17 ngày)")
    print("="*55)

    raw_dir  = os.path.join(BASE_DIR, "data", "raw")
    all_env  = []

    # ── Tìm tất cả .txt files trong data/raw ──
    txt_files = [f for f in os.listdir(raw_dir)
                 if f.endswith(".txt") and not f.startswith("_tmp")]

    if not txt_files:
        print("\n  [WARN] Không có file nào trong data/raw/")
        print("         Thêm datasets vào data/raw/ rồi chạy lại.")
        print("         Ví dụ: iTrust.txt, IBMApp.txt, CyberChair.txt")
        return

    for fname in sorted(txt_files):
        name  = fname.replace(".txt", "")
        path  = os.path.join(raw_dir, fname)
        _, env = filter_env_sentences(path, name)
        all_env.extend(env)
        save_filtered(env, name)

    # ── Tìm JSON datasets (LitroACP, RAGent) ──
    json_files = [f for f in os.listdir(raw_dir) if f.endswith(".json")]
    for fname in sorted(json_files):
        name = fname.replace(".json", "")
        path = os.path.join(raw_dir, fname)
        _, env = filter_json_dataset(path, name)
        all_env.extend(env)
        save_filtered(env, name)

    print(f"\n{'='*55}")
    print(f"  TỔNG câu có env-att: {len(all_env)}")
    print(f"  Temporal: {sum(r['has_temporal'] for r in all_env)}")
    print(f"  Spatial:  {sum(r['has_spatial']  for r in all_env)}")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
