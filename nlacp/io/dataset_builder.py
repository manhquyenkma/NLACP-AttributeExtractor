import json
import os

# ===================================================================
# dataset_builder.py  (nlacp/io/)
# Quản lý đọc/ghi dataset JSON
# ===================================================================

# nlacp/io/ → nlacp/ → project root
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_PATH  = os.path.join(BASE_DIR, "outputs", "policies", "policy_dataset.json")


def ensure_dataset():
    os.makedirs(os.path.dirname(DATASET_PATH), exist_ok=True)
    if not os.path.exists(DATASET_PATH):
        data = {"policies": []}
        with open(DATASET_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4)


def load_dataset():
    ensure_dataset()
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def save_dataset(data):
    with open(DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)


def _format_environment(env_attrs):
    """Chuyển từ env_extractor format sang format mới có preposition/head/modifier."""
    result = []
    for ea in env_attrs:
        value = ea.get("value", "")
        parts = value.split()
        result.append({
            "type":        ea.get("sub_category", ea.get("subcategory", "")),
            "preposition": ea.get("trigger", parts[0] if parts else ""),
            "head":        parts[-1] if len(parts) > 1 else value,
            "modifier":    parts[1] if len(parts) > 2 else None,
            "full_value":  value,
            "normalized":  ea.get("short_name", ""),
            "namespace":   ea.get("namespace", ""),
            "data_type":   ea.get("data_type", "string")
        })
    return result


def add_policy(relation_data):
    """
    Thêm một policy vào dataset.
    relation_data chứa: sentence, subject, actions, object, attributes, environment
    Mỗi attribute/environment có fields tương ứng.
    """
    dataset = load_dataset()

    # Tránh trùng câu
    for policy in dataset["policies"]:
        if policy["sentence"] == relation_data["sentence"]:
            return

    new_id = len(dataset["policies"]) + 1

    actions_list = relation_data.get("actions", [])
    action_str = ", ".join(actions_list) if actions_list else None

    policy = {
        "id":          new_id,
        "sentence":    relation_data["sentence"],
        "subject":     relation_data.get("subject"),
        "action":      action_str,
        "object":      relation_data.get("object"),
        "attributes":  relation_data.get("attributes", []),
        "environment": _format_environment(relation_data.get("environment", []))
    }

    dataset["policies"].append(policy)
    save_dataset(dataset)
    print(f"[OK] Policy #{new_id} added to dataset.")