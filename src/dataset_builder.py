import json
import os

# ===================================================================
# dataset_builder.py
# Quản lý đọc/ghi dataset JSON
# Cập nhật: hỗ trợ attributes với category + data_type (Alohaly 2019)
# ===================================================================

BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH  = os.path.join(BASE_DIR, "dataset", "policy_dataset.json")


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


def add_policy(relation_data):
    """
    Thêm một policy vào dataset.
    relation_data chứa: sentence, subject, action, object, attributes
    Mỗi attribute có: name, value, category, dep, data_type
    """
    dataset = load_dataset()

    # Tránh trùng câu
    for policy in dataset["policies"]:
        if policy["sentence"] == relation_data["sentence"]:
            print("[INFO] Policy already exists — skipped.")
            return

    new_id = len(dataset["policies"]) + 1

    policy = {
        "id":         new_id,
        "sentence":   relation_data["sentence"],
        "subject":    relation_data.get("subject"),
        "action":     relation_data.get("action"),
        "object":     relation_data.get("object"),
        "attributes": relation_data.get("attributes", [])
    }

    dataset["policies"].append(policy)
    save_dataset(dataset)
    print(f"[OK] Policy #{new_id} added to dataset.")