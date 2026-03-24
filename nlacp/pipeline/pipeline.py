from nlacp.extraction.relation_candidate import parse_sentence, extract_relations
from nlacp.extraction.env_extractor import extract_env_attributes
from nlacp.extraction.short_name_suggester import suggest_short_names
from nlacp.normalization.namespace_assigner import assign_namespaces
from nlacp.normalization.category_identifier import identify_categories
from nlacp.io.dataset_builder import add_policy
from nlacp.paths import NS_ENV_TIME, NS_ENV_LOC

try:
    from nlacp.normalization.data_type_infer import annotate_attributes_with_type
    HAS_DTYPE = True
except ImportError:
    HAS_DTYPE = False

# ===================================================================
# pipeline.py
# Full ABAC Pipeline (Alohaly et al. 2019)
#
# Pipeline:
#   Module 1: Attribute Extraction (subject/action/object + env)
#   Module 2: Suggesting Attributes' Short Names
#   Module 3: Assigning Attributes to Namespaces Hierarchically
#   Module 4: Identifying the Category of an Attribute
#   Module 5: Identifying Data Type
#   Module 6: Policy Representation
# ===================================================================

def _cnn_filter(attributes):
    """
    Placeholder cho CNN filter. 
    Trong tương lai, gọi model Predict để lọc attributes.
    Hiện tại pass-through toàn bộ attrs.
    """
    return attributes

# --- Spatial classification helpers (dùng cho single-pass pipeline) ---
_NETWORK_HINTS = {"network", "vpn", "intranet", "internet", "internal", "external", "remote"}
_DEVICE_HINTS  = {"workstation", "device", "terminal", "laptop", "system", "portal", "platform"}

def _format_env_for_pipeline(raw_env):
    """Chuyển output từ env_extractor sang format chuẩn policy_dataset."""
    value   = raw_env.get("value", "")
    trigger = raw_env.get("trigger", "")
    cat     = raw_env.get("category", "")
    subcat  = raw_env.get("subcategory", raw_env.get("sub_category", ""))

    parts = value.split()
    preposition = trigger if trigger and not trigger.startswith("NER:") else (parts[0] if parts else "")
    content_parts = [p for p in parts if p.lower() not in {"a", "an", "the", preposition.lower()}]

    head     = content_parts[-1] if content_parts else value
    modifier = content_parts[0] if len(content_parts) > 1 else None
    if modifier and modifier.lower() == head.lower():
        modifier = None

    if cat == "temporal" or subcat in ("relative", "absolute", "recurring", "event",
                                       "ner_detected", "business_hours"):
        env_type  = "temporal"
        data_type = "time"
        ns_prefix = NS_ENV_TIME
    else:
        val = value.lower()
        if any(h in val for h in _NETWORK_HINTS):
            env_type = "spatial_network"
        elif any(h in val for h in _DEVICE_HINTS):
            env_type = "spatial_device"
        else:
            env_type = "spatial_physical"
        data_type = "location"
        ns_prefix = NS_ENV_LOC

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

def process_sentence(sentence: str, save: bool = False) -> dict:
    """
    Xử lý một câu policy theo đúng chuẩn Alohaly 2019:
      1. Extract ALL candidate pairs (không phân loại)
      2. Extract environment attributes → tách env pairs ra
      3. Loại bỏ env overlap khỏi candidate pairs
      4. Classify pairs còn lại → Subject/Object attributes
      5. Short name → Namespace → Data type
    """
    # [Module 1] Extract ALL candidate pairs + S/A/O metadata
    tokens   = parse_sentence(sentence)
    relation = extract_relations(sentence, tokens)
    
    # [Module 1b] Extract environment attributes
    env_attrs = extract_env_attributes(sentence)

    # Lấy tập token thuộc environment để loại khỏi pairs
    env_tokens = set()
    for env in env_attrs:
        val = env.get("value", "")
        for word in val.lower().split():
            if word not in {"a", "an", "the"}:
                env_tokens.add(word)
        trigger = (env.get("trigger") or "").lower()
        if trigger and not trigger.startswith("ner:"):
            env_tokens.add(trigger)

    # [Module 2] CNN filter (hiện pass-through)
    all_pairs = _cnn_filter(relation["attributes"])

    # Loại bỏ pairs mà CẢ name VÀ value đều là env token
    sa_oa_pairs = []
    for attr in all_pairs:
        name  = (attr.get("name") or "").lower()
        value = (attr.get("value") or "").lower()
        rel_sub = (relation.get("subject") or "").lower()
        rel_obj = (relation.get("object") or "").lower()
        if name in env_tokens or (value in env_tokens and value not in [rel_sub, rel_obj]):
            continue
        # Bỏ qua preposition tokens (during, within, ...)
        if name in {"during", "between", "after", "before", "within",
                     "throughout", "until", "from", "via", "through",
                     "using", "at", "on", "inside", "outside"}:
            continue
        sa_oa_pairs.append(attr)

    # [Module 4] Category Identification cho SA/OA pairs
    attrs_mod4 = identify_categories(sa_oa_pairs, sentence, relation.get("object", ""))

    # [Module 2] Suggest short names
    attrs_mod2 = suggest_short_names(attrs_mod4)

    # [Module 3] Assign namespaces
    attrs_mod3 = assign_namespaces(attrs_mod2, relation.get("subject"), relation.get("object"))

    # [Module 5] Annotate data type
    if HAS_DTYPE:
        final_attrs = annotate_attributes_with_type(attrs_mod3)
    else:
        final_attrs = []
        for a in attrs_mod3:
            a["data_type"] = "string"
            final_attrs.append(a)

    # [Module 6] Tách riêng SA/OA và gắn env đã format
    sa_oa = [a for a in final_attrs if a.get("category") in ("subject", "object")]
    
    # Format env attrs theo chuẩn policy_dataset
    formatted_env = []
    for env in env_attrs:
        formatted_env.append(_format_env_for_pipeline(env))

    relation["attributes"]  = sa_oa
    relation["environment"] = formatted_env
    
    # Lưu vào DB nếu có save=True
    if save:
        add_policy(relation)
    return relation


def main():
    print("\n" + "="*55)
    print("  ABAC Policy NLP Extraction")
    print("  (Alohaly et al. 2019 — Module 1)")
    print("="*55)
    print("\nType 'exit' to stop\n")
    print("Ví dụ câu đầu vào:")
    print("  An on-call senior nurse may change the list of approved lab procedures.")
    print("  A junior lab technician can request follow-up lab procedures.")
    print("  Managers in the finance department can approve expense reports.\n")

    while True:
        try:
            sentence = input("Enter policy sentence: ").strip()
        except (EOFError, KeyboardInterrupt):
            break

        if not sentence:
            continue

        if sentence.lower() == "exit":
            break

        result = process_sentence(sentence)

        print("\n--- Extracted ABAC Policy (6 Modules) ---")
        print(f"  Subject  : {result.get('subject')}")
        print(f"  Actions  : {', '.join(result.get('actions', []))}")
        print(f"  Object   : {result.get('object')}")

        # Environment — nằm trong result["environment"], không phải attributes
        env_list = result.get("environment", [])
        if env_list:
            print(f"  Environment ({len(env_list)}):")
            for e in env_list:
                # canonical format: có 'namespace' và 'full_value'
                if "namespace" in e:
                    print(f"    [{e.get('type','?')}] {e.get('full_value','?')} "
                          f"-> {e.get('namespace','?')}")
                else:
                    # raw format fallback
                    print(f"    {e.get('value', e)}")
        else:
            print("  Environment: (none detected)")

        # SA/OA Attributes — nằm trong result["attributes"]
        sa_oa = result.get("attributes", [])
        if sa_oa:
            print(f"  Attributes ({len(sa_oa)}):")
            for a in sa_oa:
                print(f"    [{a.get('category','?').upper()}] "
                      f"{a.get('namespace','?')} = \"{a.get('short_name','?')}\"")
        else:
            print("  Attributes: (none detected)")
        print()


if __name__ == "__main__":
    main()