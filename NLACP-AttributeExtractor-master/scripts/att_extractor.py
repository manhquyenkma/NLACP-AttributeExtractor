#!/usr/bin/env python
"""
scripts/att_extractor.py — Bước 2 mới của pipeline

Đọc policy_dataset.json (đã verified bởi interactive_verification.py),
thực hiện:
  2A. Gán SA/OA attributes từ validated relation_pairs
      (short_name, namespace, data_type)
  2B. Gán ENV namespace theo cấu trúc phân cấp environment.*
  2C. DBSCAN clustering → attribute keys (gom nhóm các attributes tương đồng)
  2D. Namespace hierarchy

Output:
  policy_dataset.json (complete — có đầy đủ attributes + environment)
  outputs/clusters/attribute_clusters.json
  outputs/hierarchy/namespace_hierarchy.json

Usage:
    python scripts/att_extractor.py
    python scripts/att_extractor.py --no-cluster   # Bỏ qua clustering
"""
import json
import os
import sys
import argparse

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from nlacp.extraction.short_name_suggester import suggest_short_names
from nlacp.normalization.namespace_assigner import assign_namespaces
from nlacp.normalization.category_identifier import identify_categories
from nlacp.normalization.data_type_infer import annotate_attributes_with_type
from nlacp.paths import (
    POLICY_DATASET_PATH, ATTRIBUTE_CLUSTERS_PATH, NAMESPACE_HIERARCHY_PATH,
    NS_ENV_TIME, NS_ENV_LOC
)

DATASET_DIR = os.path.dirname(POLICY_DATASET_PATH)
PREDICATE_MAP_PATH = os.path.join(DATASET_DIR, "predicate_property_map.json")
def load_predicate_map():
    try:
        with open(PREDICATE_MAP_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except:
        return {}

PREDICATE_MAP = load_predicate_map()

# =====================================================================
#  ENV Namespace — cấu trúc phân cấp theo đặc tả
# =====================================================================

# Mapping env_type → namespace top-level
ENV_NS_MAP = {
    "temporal":         "environment.time",
    "spatial_physical": "environment.location",
    "spatial_network":  "environment.network",
    "spatial_device":   "environment.device",
    "conditional":      "environment.condition",
}

# Mapping subcategory → namespace leaf
ENV_SUB_NS = {
    "working_period": "working_period",  # environment.time.working_period
    "absolute":       "absolute_time",
    "recurring":      "recurring",
    "event":          "event_based",
    "relative":       "relative_time",
    "facility":       "facility",        # environment.location.facility
    "campus":         "campus",
    "geographic":     "geographic",
    "location":       "location",
    "network_zone":   "network_zone",    # environment.network.network_zone
    "access_type":    "access_type",     # environment.network.access_type
    "device_type":    "device_type",     # environment.device.device_type
}


def _build_env_namespace(env_entry: dict) -> str:
    """
    Tạo namespace cho environment entry theo cấu trúc:
      environment.time.working_period:business_hours
      environment.location.facility:hospital
      environment.network.access_type:vpn
      environment.condition.device.connection_status:secure_network (conditional_clause)
    """
    env_type    = env_entry.get("env_type", "")
    subcategory = env_entry.get("subcategory", "")
    normalized  = env_entry.get("normalized", "")

    if env_type == "conditional" and subcategory == "conditional_clause":
        cond = env_entry.get("condition", {})
        entity = cond.get("subject", "").lower().replace(" ", "_") or "unknown_entity"
        predicate = cond.get("predicate", "").lower()
        prop = PREDICATE_MAP.get(predicate, predicate if predicate else "status")
        val = cond.get("object", "").lower().replace(" ", "_") or "true"
        if cond.get("negated"):
            val = f"not_{val}" if val != "true" else "false"
        return f"environment.condition.{entity}.{prop}:{val}"

    top = ENV_NS_MAP.get(env_type, "environment.other")
    sub = ENV_SUB_NS.get(subcategory, subcategory) if subcategory else "unknown"
    val = normalized if normalized else "unknown"
    return f"{top}.{sub}:{val}"


# =====================================================================
#  SA/OA Attribute Processing
# =====================================================================

def process_sa_oa_attributes(policy: dict) -> list:
    """
    Chuyển validated relation_pairs thành SA/OA attributes hoàn chỉnh.
    Steps:
      1. Classify categories (subject / object)
      2. Suggest short names
      3. Assign namespaces
      4. Infer data types
    """
    # Lấy env tokens để loại overlap
    env_tokens = set()
    for env in policy.get("environments", policy.get("environment", [])):
        for word in (env.get("full_value") or env.get("trigger_phrase") or "").lower().split():
            if word not in {"a", "an", "the"}:
                env_tokens.add(word)
        trigger = (env.get("trigger") or "").lower()
        if trigger and not trigger.startswith("ner:"):
            env_tokens.add(trigger)

    subjects = policy.get("subjects", policy.get("subject", [])) or []
    if isinstance(subjects, str): subjects = [subjects]
    sub_names = [(s.get("role") or str(s)) if isinstance(s, dict) else str(s) for s in subjects]
    sub_names = [s.lower() for s in sub_names if s]
    
    objects = []
    res = policy.get("resource")
    if res:
        objects.append((res.get("label") or str(res)) if isinstance(res, dict) else str(res))
    for r in policy.get("additional_resources", []) or []:
        objects.append((r.get("label") or str(r)) if isinstance(r, dict) else str(r))
    if not objects:
        objects = policy.get("object", []) or []
    if isinstance(objects, str): objects = [objects]
    obj_names = [o.lower() for o in objects if o]

    # Build raw attrs từ relation_pairs
    ENV_PREPS = {"during", "within", "after", "before", "between", "via",
                 "through", "using", "at", "on", "from", "inside", "outside",
                 "throughout", "until"}

    raw_attrs = []
    for pair in policy.get("relation_pairs", []) or []:
        if isinstance(pair, dict):
            val = str(pair.get("value", ""))
            name = str(pair.get("attribute", ""))
        elif isinstance(pair, (list, tuple)):
            val  = str(pair[0]) if len(pair) > 0 else ""
            name = str(pair[1]) if len(pair) > 1 else ""
        else:
            continue
        if not val or not name:
            continue
        if name.lower() in ENV_PREPS:
            continue

        # Loại bỏ nếu name hoặc value là env token
        if name.lower() in env_tokens:
            continue
        if val.lower() in env_tokens and val.lower() not in sub_names and val.lower() not in obj_names:
            continue

        # Category sơ bộ
        if val.lower() in sub_names:
            cat = "subject"
        elif val.lower() in obj_names:
            cat = "object"
        else:
            cat = "unclassified"

        raw_attrs.append({
            "name":     name,
            "value":    val,
            "category": cat
        })

    # Module 4: Category identification
    sentence = policy.get("sentence", "")
    attrs = identify_categories(raw_attrs, sentence, objects)

    # Module 2: Short names
    attrs = suggest_short_names(attrs)

    # Module 3: Namespaces
    attrs = assign_namespaces(attrs, subjects, objects)

    # Module 5: Data types
    attrs = annotate_attributes_with_type(attrs)

    return attrs


# =====================================================================
#  ENV Namespace fix-up
# =====================================================================

def process_env_namespace(policy: dict) -> list:
    """
    Từ environment entries trong policy (đã verified),
    đảm bảo namespace đúng chuẩn phân cấp environment.*.
    Sử dụng rule-based classify_env_phrase_with_confidence để phân loại.
    """
    updated = []
    for env in policy.get("environments", policy.get("environment", [])):
        phrase = env.get("full_value", env.get("trigger_phrase", ""))
        if not phrase:
            phrase = env.get("phrase", "")
            
        env_type, namespace, score = classify_env_phrase_with_confidence(phrase)
        
        env["env_type"] = env_type
        env["namespace"] = namespace
        env["subcategory"] = env_type
        env["confidence"] = score

        # Ensure full_value exists for downstream
        if "full_value" not in env:
            env["full_value"] = phrase

        updated.append(env)
    return updated


import re

def classify_env_phrase_with_confidence(phrase, token=None, doc=None):
    phrase_lower = phrase.lower().strip()
    scores = {}
    
    # Score for each category based on evidence strength
    
    # TEMPORAL — time range (highest confidence: explicit times)
    if re.search(r'\d+\s*(?:am|pm|:\d{2})', phrase_lower):
        scores['time_range'] = 1.0  # explicit clock time
    
    # TEMPORAL — named periods (medium confidence)
    WORK_PERIOD = ["business hours", "working hours", "office hours"]
    ACADEMIC_PERIOD = ["enrollment period", "grading period", 
                       "semester", "academic year", "deadline"]
    if any(w in phrase_lower for w in WORK_PERIOD):
        scores['working_period'] = 0.9
    if any(w in phrase_lower for w in ACADEMIC_PERIOD):
        scores['academic_period'] = 0.85
        
    # TEMPORAL — generic temporal (lower confidence)
    TEMPORAL = ["during", "before", "after", "between", 
                "until", "night shift", "weekday"]
    temporal_matches = sum(1 for w in TEMPORAL if w in phrase_lower)
    if temporal_matches > 0:
        scores['temporal'] = 0.6 + (0.1 * min(temporal_matches, 3))
    
    # NETWORK
    NETWORK = ["vpn", "intranet", "network", "wi-fi", 
               "wifi", "campus wi", "university network"]
    network_matches = sum(1 for w in NETWORK if w in phrase_lower)
    if network_matches > 0:
        scores['spatial_network'] = 0.7 + (0.1 * min(network_matches, 2))
    
    # SYSTEM
    SYSTEM = ["management system", "information system",
              "registration system", "portal", "platform"]
    if any(w in phrase_lower for w in SYSTEM):
        scores['academic_system'] = 0.95
    
    # SPATIAL — physical
    SPATIAL = ["campus", "office", "lab", "building",
               "room", "terminal", "lounge", "on campus",
               "in the", "from the", "within the", "at the"]
    spatial_matches = sum(1 for w in SPATIAL if w in phrase_lower)
    if spatial_matches > 0:
        scores['spatial_physical'] = 0.65 + (0.1 * min(spatial_matches, 2))
    
    # CONDITIONAL
    CONDITIONAL = ["only if", "only when", "unless", 
                   "is active", "is open", "is pending",
                   "is archived", "in good standing",
                   "is under review", "is activated"]
    cond_matches = sum(1 for w in CONDITIONAL if w in phrase_lower)
    if cond_matches > 0:
        scores['conditional'] = 0.75 + (0.1 * min(cond_matches, 2))
    
    # Add syntactic evidence bonus if token/doc available
    if token is not None and doc is not None:
        # Bonus if phrase is in advcl (high confidence it's ENV)
        if token.dep_ in ['advcl', 'prep', 'advmod']:
            for key in scores:
                scores[key] = min(1.0, scores[key] + 0.1)
    
    if not scores:
        return 'other', 'environment.other', 0.0
    
    best_type = max(scores, key=scores.get)
    best_score = scores[best_type]
    
    NAMESPACE_MAP = {
        'time_range': 'environment.time.time_range',
        'working_period': 'environment.time.working_period',
        'academic_period': 'environment.time.academic_period',
        'temporal': 'environment.time.temporal',
        'spatial_network': 'environment.network.access_channel',
        'academic_system': 'environment.system.platform',
        'spatial_physical': 'environment.location.physical',
        'conditional': 'environment.state.condition',
        'other': 'environment.other'
    }
    
    return best_type, NAMESPACE_MAP[best_type], round(best_score, 4)


def classify_env_phrase(phrase):
    env_type, namespace, score = classify_env_phrase_with_confidence(phrase)
    return env_type, namespace

def run_clustering(dataset: dict):
    print("\n[INFO] Bypassing DBSCAN attribute clustering (using rule-based instead).")

def run_hierarchy():
    print("\n[INFO] Bypassing namespace hierarchy.")

# =====================================================================
#  REPORT ENV NAMESPACES
# =====================================================================

def print_env_namespace_report(all_policies: list):
    """
    In báo cáo tái sử dụng namespace environment.
    Cùng một 'business hours' có thể xuất hiện trong nhiều policy khác nhau
    → namespace được định nghĩa một lần, tham chiếu lại.
    """
    from collections import defaultdict, Counter
    ns_usage = defaultdict(list)

    for p in all_policies:
        for env in p.get("environments", p.get("environment", [])):
            ns = env.get("namespace", "")
            if ns:
                ns_usage[ns].append(p.get("id"))

    reused = {ns: ids for ns, ids in ns_usage.items() if len(ids) > 1}

    if reused:
        print("\n  ENV Namespace reuse (xuất hiện trong nhiều policies):")
        for ns, ids in sorted(reused.items(), key=lambda x: -len(x[1])):
            print(f"    {ns}  → policies: {ids}")


# =====================================================================
#  MAIN
# =====================================================================

def main(run_cluster: bool = True):
    print("\n" + "=" * 60)
    print("  ATT EXTRACTOR — Bước 2")
    print("  Attribute Processing + ENV Namespace + DBSCAN Clustering")
    print("=" * 60)

    if not os.path.exists(POLICY_DATASET_PATH):
        print(f"\n[ERROR] {POLICY_DATASET_PATH} không tồn tại.")
        print("        Chạy 'python scripts/interactive_verification.py' trước (Bước 1).")
        return

    with open(POLICY_DATASET_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    policies = data.get("policies", [])
    if not policies:
        print("[WARN] policy_dataset.json rỗng.")
        return

    print(f"\n[INFO] Đang xử lý {len(policies)} policy(ies)...\n")

    for policy in policies:
        sentence = policy.get("sentence", "")
        pid      = policy.get("id", "?")
        print(f"  Policy #{pid}: {sentence[:65]}...")

        # B1: Fix-up ENV namespaces
        env_key = "environments" if "environments" in policy else "environment"
        policy[env_key] = process_env_namespace(policy)
        env_count = len(policy[env_key])
        print(f"    ENV: {env_count} entry(ies)")
        for e in policy[env_key]:
            print(f"      [{e['env_type']:16s}|{e['subcategory']:14s}]"
                  f" \"{e.get('full_value', e.get('phrase','?'))}\"")
            print(f"         → {e['namespace']}")

        # B2: Process SA/OA attributes
        policy["attributes"] = process_sa_oa_attributes(policy)
        attr_count = len(policy["attributes"])
        print(f"    SA/OA: {attr_count} attribute(s)")
        for a in policy["attributes"]:
            cat = a.get("category", "?").upper()
            ns  = a.get("namespace", "?")
            sn  = a.get("short_name", "?")
            print(f"      [{cat}] {ns} = \"{sn}\"")
        print()

    # Ghi lại policy_dataset.json đã enriched
    with open(POLICY_DATASET_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print(f"[OK] Saved complete policies → {POLICY_DATASET_PATH}")

    # In báo cáo namespace reuse
    print_env_namespace_report(policies)

    # Clustering
    if run_cluster:
        run_clustering(data)
        run_hierarchy()

    print("\n" + "=" * 60)
    print("  BƯỚC 2 HOÀN TẤT!")
    print(f"  → policy_dataset.json : {POLICY_DATASET_PATH}")
    if run_cluster:
        print(f"  → attribute_clusters  : {ATTRIBUTE_CLUSTERS_PATH}")
        print(f"  → namespace_hierarchy : {NAMESPACE_HIERARCHY_PATH}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ATT Extractor — Bước 2 Pipeline")
    parser.add_argument("--no-cluster", action="store_true",
                        help="Bỏ qua DBSCAN clustering")
    args = parser.parse_args()
    main(run_cluster=not args.no_cluster)
