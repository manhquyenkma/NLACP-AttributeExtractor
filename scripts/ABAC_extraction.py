#!/usr/bin/env python
"""
ABAC_extraction.py — STEP 2: Environment Filling + Attribute Post-Processing

Doc policy_dataset.json (da duoc validate o Step 1) va:
  2A. Chay env_extractor -> dien truong "environment" theo format moi
  2B. Chay short_name, namespace, data_type cho "attributes"
      (loai bo cac gia tri trung voi environment)
  2C. (Tuy chon) Chay attribute_cluster + namespace_hierarchy

Usage:
    python ABAC_extraction.py
"""
import json
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.extraction.env_extractor import extract_env_attributes
from nlacp.extraction.short_name_suggester import suggest_short_names
from nlacp.normalization.namespace_assigner import assign_namespaces
from nlacp.normalization.category_identifier import identify_categories
from nlacp.normalization.data_type_infer import annotate_attributes_with_type

DATASET_DIR  = os.path.join(PROJECT_ROOT, "dataset")
from nlacp.paths import POLICY_DATASET_PATH as POLICY_PATH


# =====================================================================
#  2A: Environment Extraction & Formatting
# =====================================================================

# --- Spatial classification helpers ---
NETWORK_HINTS = {"network", "vpn", "intranet", "internet", "internal", "external", "remote"}
DEVICE_HINTS  = {"workstation", "device", "terminal", "laptop", "system", "portal", "platform"}


def _classify_spatial_type(value_text):
    """Phan loai spatial thanh spatial_physical, spatial_network, spatial_device."""
    val = value_text.lower()
    if any(h in val for h in NETWORK_HINTS):
        return "spatial_network"
    if any(h in val for h in DEVICE_HINTS):
        return "spatial_device"
    return "spatial_physical"


def _format_env_entry(raw_env):
    """Chuyen tu env_extractor format sang format moi cua policy_dataset."""
    value   = raw_env.get("value", "")
    trigger = raw_env.get("trigger", "")
    cat     = raw_env.get("category", "")
    subcat  = raw_env.get("subcategory", raw_env.get("sub_category", ""))

    # Parse value into parts
    parts = value.split()
    preposition = trigger if trigger and not trigger.startswith("NER:") else (parts[0] if parts else "")

    # head = danh tu chinh (tu cuoi cung la noun)
    # modifier = tu bo nghia (neu co)
    content_parts = [p for p in parts if p.lower() not in {
        "a", "an", "the", preposition.lower()
    }]

    head     = content_parts[-1] if content_parts else value
    modifier = content_parts[0] if len(content_parts) > 1 else None
    if modifier and modifier.lower() == head.lower():
        modifier = None

    # Xac dinh type
    if cat == "temporal" or subcat in ("relative", "absolute", "recurring", "event",
                                       "ner_detected", "business_hours"):
        from nlacp.paths import NS_ENV_TIME
        env_type  = "temporal"
        data_type = "time"
        ns_prefix = NS_ENV_TIME
    else:
        from nlacp.paths import NS_ENV_LOC
        env_type  = _classify_spatial_type(value)
        data_type = "location"
        ns_prefix = NS_ENV_LOC

    # Normalized name
    normalized = "_".join(p.lower() for p in content_parts) if content_parts else head.lower()

    return {
        "type":        env_type,
        "preposition": preposition,
        "head":        head,
        "modifier":    modifier,
        "full_value":  value,
        "normalized":  normalized,
        "namespace":   f"{ns_prefix}:{normalized}",
        "data_type":   data_type
    }


def fill_environment(policy):
    """Trich xuat va dien environment cho mot policy."""
    sentence = policy.get("sentence", "")
    raw_envs = extract_env_attributes(sentence)

    formatted = []
    for env in raw_envs:
        formatted.append(_format_env_entry(env))

    policy["environment"] = formatted
    return formatted


# =====================================================================
#  2B: Attribute Post-Processing (loai bo env overlap)
# =====================================================================

def _get_env_tokens(env_list):
    """Lay tap hop cac tu (token) thuoc ve environment de loai khoi attributes."""
    tokens = set()
    for env in env_list:
        head = (env.get("head") or "").lower()
        modifier = (env.get("modifier") or "").lower()
        prep = (env.get("preposition") or "").lower()
        if head:
            tokens.add(head)
        if modifier:
            tokens.add(modifier)
        if prep:
            tokens.add(prep)
        # Them cac tu trong full_value
        for word in (env.get("full_value") or "").lower().split():
            if word not in {"a", "an", "the"}:
                tokens.add(word)
    return tokens


def fill_attributes(policy):
    """Xu ly attributes tu validated pairs:
    1. Loai env overlap
    2. Classify SA/OA
    3. Short name, namespace, data_type
    """
    assert "environment" in policy, "Call fill_environment() first"
    env_tokens = _get_env_tokens(policy.get("environment", []))
    
    obj_name = (policy.get("object") or "").lower()
    sub_name = (policy.get("subject") or "").lower()
    
    # Lay validated pairs va chuyen thanh attribute dicts
    raw_attrs = []
    for pair in policy.get("relation_pairs", []):
        val = pair[0].lower()
        if val == obj_name:
            cat = "object"
        elif val == sub_name:
            cat = "subject"
        else:
            cat = "unclassified"
            
        raw_attrs.append({
            "name":     pair[1],
            "value":    pair[0],
            "category": cat
        })

    # Loc bo cac attribute ma name hoac value trung voi env token
    clean_attrs = []
    for attr in raw_attrs:
        name  = (attr.get("name") or "").lower()
        value = (attr.get("value") or "").lower()
        if name in env_tokens or (value in env_tokens and value not in [sub_name, obj_name]):
            continue
        if name in {"during", "between", "after", "before", "within",
                     "throughout", "until", "from", "via", "through",
                     "using", "at", "on", "inside", "outside"}:
            continue
        clean_attrs.append(attr)

    # [Module 4] Classify pairs con lai thanh SA/OA
    obj_name = policy.get("object", "")
    clean_attrs = identify_categories(clean_attrs, policy.get("sentence", ""), obj_name)

    # Chay Module 2: Short Name Suggestion
    clean_attrs = suggest_short_names(clean_attrs)

    # Chay Module 3: Namespace Assignment
    subject = policy.get("subject")
    obj     = policy.get("object")
    clean_attrs = assign_namespaces(clean_attrs, subject, obj)

    # Chay Module 5: Data Type Inference
    clean_attrs = annotate_attributes_with_type(clean_attrs)

    policy["attributes"] = clean_attrs


# =====================================================================
#  MAIN
# =====================================================================

def main():
    print("\n" + "=" * 60)
    print("  ABAC EXTRACTION — STEP 2")
    print("  S/A/O Extraction + Environment + Attribute Post-Processing")
    print("=" * 60)

    if not os.path.exists(POLICY_PATH):
        print(f"\n[ERROR] {POLICY_PATH} khong ton tai.")
        print("        Chay 'python scripts/data_processing.py' truoc (Step 1).")
        return

    with open(POLICY_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    policies = data.get("policies", [])
    if not policies:
        print("[WARN] policy_dataset.json rong.")
        return

    print(f"\n[INFO] Dang xu ly {len(policies)} policy(ies)...\n")

    # Import extract_relations de lay S/A/O tu sentence
    from nlacp.extraction.relation_candidate import extract_relations, parse_sentence

    for policy in policies:
        sentence = policy["sentence"]
        print(f"  Policy #{policy['id']}: {sentence[:60]}...")

        # 2A: Trich xuat S/A/O tu sentence neu policy chua co san hoac fallback thi parse (de tuong thich code cu)
        if not policy.get("subject") or not policy.get("object"):
            tokens = parse_sentence(sentence)
            rel = extract_relations(sentence, tokens)
            if rel.get("subject"): policy["subject"] = rel["subject"]
            if rel.get("actions"): policy["actions"] = rel["actions"]
            if rel.get("object"):  policy["object"]  = rel["object"]
            
        print(f"    Subject: {policy.get('subject')}  |  Actions: {policy.get('actions')}  |  Object: {policy.get('object')}")

        # 2B: Fill environment
        envs = fill_environment(policy)
        print(f"    Environment: {len(envs)} entry(ies)")
        for e in envs:
            print(f"      [{e['type']}] {e['preposition']} {e.get('modifier', '')} {e['head']}"
                  f" -> {e['namespace']}")

        # 2C: Fill attributes (tu validated pairs, loai env overlap)
        fill_attributes(policy)
        attrs = policy.get("attributes", [])
        print(f"    Attributes:  {len(attrs)} entry(ies)")
        for a in attrs:
            print(f"      [{a.get('category','?')}] {a.get('name','')} -> {a.get('value','')}"
                  f"  ns={a.get('namespace','?')}")
        print()

    # Ghi lai policy_dataset.json da enriched (tai cho)
    with open(POLICY_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

    print(f"[OK] Saved enriched policies to {POLICY_PATH}")

    # 2D: Chay mining (optional)
    print("\n[INFO] Chay attribute clustering...")
    try:
        from nlacp.mining.attribute_cluster import main as run_clustering
        run_clustering()
    except Exception as e:
        print(f"  [WARN] Clustering failed: {e}")

    print("\n[INFO] Chay namespace hierarchy...")
    try:
        from nlacp.mining.namespace_hierarchy import main as run_hierarchy
        run_hierarchy()
    except Exception as e:
        print(f"  [WARN] Hierarchy failed: {e}")

    print("\n" + "=" * 60)
    print("  STEP 2 HOAN TAT!")
    print(f"  -> policy_dataset.json: {POLICY_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    main()
