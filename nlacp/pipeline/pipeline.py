from nlacp.extraction.relation_candidate import parse_sentence, extract_relations
from nlacp.extraction.env_extractor import extract_env_attributes
from nlacp.extraction.short_name_suggester import suggest_short_names
from nlacp.normalization.namespace_assigner import assign_namespaces
from nlacp.normalization.category_identifier import identify_categories
from nlacp.io.dataset_builder import add_policy

try:
    from nlacp.normalization.data_type_infer import annotate_attributes_with_type
    HAS_DTYPE = True
except ImportError:
    HAS_DTYPE = False

# ===================================================================
# nlp_engine.py
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
    Hiện tại pass-through toàn bộ SA/OA attrs.
    """
    return attributes

def process_sentence(sentence):
    """
    Xử lý một câu policy qua 6 Module để xuất ra ABAC policy với kiến trúc chuẩn.
    """
    # [Module 1] Extract core attributes & relations
    tokens   = parse_sentence(sentence)
    relation = extract_relations(sentence, tokens)
    
    # [Module 1b] Extract environment attributes (Layer 1-3 hybrid)
    env_attrs = extract_env_attributes(sentence)

    # Bước 2: CNN filter cho SA/OA (nếu mô hình đã train, hiện pass-through)
    sa_oa_attrs = _cnn_filter(relation["attributes"])

    # Gộp lại để chạy qua các module xử lý tên và namespace
    all_raw_attrs = sa_oa_attrs + env_attrs

    # Bước 3: Đảm bảo category đúng (TRƯỚC namespace)
    attrs_mod4 = identify_categories(all_raw_attrs, sentence)

    # Bước 4: Tạo short_name ròi gán namespace
    attrs_mod2 = suggest_short_names(attrs_mod4)
    attrs_mod3 = assign_namespaces(attrs_mod2, relation.get("subject"), relation.get("object"))

    # Bước 5: Annotate data type
    if HAS_DTYPE:
        final_attrs = annotate_attributes_with_type(attrs_mod3)
    else:
        final_attrs = []
        for a in attrs_mod3:
            a["data_type"] = "string"
            final_attrs.append(a)

    # Bước 6: Tách riêng SA/OA và Environment
    sa_oa = [a for a in final_attrs if a.get("category") in ("subject", "object")]
    ea    = [a for a in final_attrs if a.get("category") == "environment"]

    relation["attributes"]  = sa_oa
    relation["environment"] = ea
    
    # Lưu vào DB (format mới)
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
        print(f"Subject:     {result.get('subject')}")
        print(f"Action:      {', '.join(result.get('actions', []))}")
        print(f"Object:      {result.get('object')}")
        
        env_strings = []
        other_attrs = []
        if result["attributes"]:
            for attr in result["attributes"]:
                cat = attr.get("category", "")
                ns = attr.get("namespace", "?")
                val = attr.get("short_name", "?")
                dtype = attr.get("data_type", "String")
                formatted = f"{ns} = \"{val}\" (Type: {dtype})"
                
                if cat == "environment":
                    env_strings.append(formatted)
                else:
                    other_attrs.append(f"[{cat.upper()}] {formatted}")
                    
        if env_strings:
            print(f"Environment: {', '.join(env_strings)}")
        else:
            print("Environment: (none detected)")
            
        if other_attrs:
            print("Other Attributes:")
            for a in other_attrs:
                print(f"  {a}")
        else:
            print("Other Attributes: (none detected)")
        print()


if __name__ == "__main__":
    main()