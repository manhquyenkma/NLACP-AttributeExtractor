"""
convert_litroacp.py — Chuyển đổi LitroACP JSONL sang format env-att annotation
LitroACP dùng nhãn 'Condition' — đây chính là env-att trong ngôn ngữ của họ.
Chạy: python data/convert_litroacp.py
"""
import json
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset files từ LitroACP
LITRO_FILES = {
    "cyber":     "data/raw/cyber_acp.jsonl",
    "ibm":       "data/raw/ibm_acp.jsonl",
    "collected": "data/raw/collected_acp.jsonl",
    "t2p":       "data/raw/t2p_acp.jsonl",
    "acre":      "data/raw/acre_acp.jsonl",
}


def load_jsonl(path):
    items = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return items


def extract_conditions(item):
    """
    Lấy Condition entities từ LitroACP item.
    Condition ≈ env-att (temporal, spatial, situational context).
    """
    text     = item.get("text", "")
    entities = item.get("entities", [])

    conditions = []
    for ent in entities:
        if ent.get("label") == "Condition":
            start = ent.get("start_offset", 0)
            end   = ent.get("end_offset", 0)
            value = text[start:end]
            conditions.append({
                "value":       value,
                "start":       start,
                "end":         end,
                "source":      "LitroACP_Condition"
            })
    return conditions


def classify_condition(value):
    """Phân loại Condition thành temporal/spatial dựa trên nội dung."""
    val_lower = value.lower()

    temporal_hints = {
        "after", "before", "during", "between", "when",
        "once", "as soon as", "time", "hour", "day", "week",
        "date", "period", "phase", "step", "deadline"
    }
    spatial_hints = {
        "from", "at", "within", "inside", "outside",
        "network", "location", "site", "department", "using", "via"
    }

    has_t = any(h in val_lower for h in temporal_hints)
    has_s = any(h in val_lower for h in spatial_hints)

    if has_t and not has_s:
        return "temporal"
    if has_s and not has_t:
        return "spatial"
    if has_t and has_s:
        return "temporal"   # temporal wins nếu có cả hai
    return "situational"    # không phân loại được → situational


def convert_dataset(dataset_name, jsonl_path):
    """Chuyển đổi 1 dataset LitroACP sang format env-annotation."""
    items = load_jsonl(jsonl_path)
    output = []
    stats  = {"total": 0, "has_condition": 0, "temporal": 0, "spatial": 0}

    for item in items:
        text       = item.get("text", "").strip()
        conditions = extract_conditions(item)
        stats["total"] += 1

        env_attrs = []
        for cond in conditions:
            cat = classify_condition(cond["value"])
            env_attrs.append({
                "category": cat,
                "value":    cond["value"],
                "trigger":  "LitroACP",
                "source":   "condition_annotation"
            })
            if cat == "temporal":
                stats["temporal"] += 1
            elif cat == "spatial":
                stats["spatial"] += 1

        if conditions:
            stats["has_condition"] += 1

        output.append({
            "id":             f"{dataset_name}_{item.get('id', len(output)+1)}",
            "dataset":        dataset_name,
            "sentence":       text,
            "env_attributes": env_attrs,
            "source":         "LitroACP",
            "annotated":      True
        })

    # Lưu file
    out_path = os.path.join(BASE_DIR, "data", "annotated",
                            f"{dataset_name}_env_annotated.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  {dataset_name}:")
    print(f"    Tổng câu:           {stats['total']}")
    print(f"    Có condition(env):  {stats['has_condition']}"
          f"  ({stats['has_condition']/max(stats['total'],1)*100:.1f}%)")
    print(f"    Temporal:           {stats['temporal']}")
    print(f"    Spatial:            {stats['spatial']}")
    print(f"    Saved:              {os.path.basename(out_path)}")

    return output, stats


def main():
    print("\n" + "="*55)
    print("  Convert LitroACP -> Env-Att Annotation Format")
    print("  (Condition label = env-att)")
    print("="*55)

    all_data   = []
    all_stats  = {"total": 0, "has_condition": 0, "temporal": 0, "spatial": 0}

    for name, rel_path in LITRO_FILES.items():
        full_path = os.path.join(BASE_DIR, rel_path)
        if not os.path.exists(full_path):
            print(f"\n  [SKIP] {name}: file not found at {rel_path}")
            continue

        data, stats = convert_dataset(name, full_path)
        all_data.extend(data)
        for k in all_stats:
            all_stats[k] += stats[k]

    # Combine tất cả
    combined_path = os.path.join(BASE_DIR, "data", "annotated", "combined_env.json")
    with open(combined_path, "w", encoding="utf-8") as f:
        json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*55}")
    print(f"  TỔNG:")
    print(f"    Câu:              {all_stats['total']}")
    print(f"    Có env-att:       {all_stats['has_condition']}"
          f"  ({all_stats['has_condition']/max(all_stats['total'],1)*100:.1f}%)")
    print(f"    Temporal:         {all_stats['temporal']}")
    print(f"    Spatial:          {all_stats['spatial']}")
    print(f"    Combined saved:   data/annotated/combined_env.json")
    print("="*55 + "\n")


if __name__ == "__main__":
    main()
