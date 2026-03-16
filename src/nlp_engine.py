import spacy
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from relation_candidate import parse_sentence, extract_relations
from env_extractor import extract_env_attributes
from short_name_suggester import suggest_short_names
from namespace_assigner import assign_namespaces
from category_identifier import identify_categories
from dataset_builder import add_policy

try:
    from data_type_infer import annotate_attributes_with_type
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

def process_sentence(sentence):
    """
    Xử lý một câu policy qua 6 Module để xuất ra ABAC policy.
    """
    # [Module 1] Extract core attributes & relations
    tokens   = parse_sentence(sentence)
    relation = extract_relations(sentence, tokens)
    
    # [Module 1] Extract environment attributes (luật nâng cao)
    env_attrs = extract_env_attributes(sentence)
    
    # Gộp tất cả attribute thô lại
    all_raw_attrs = relation.get("attributes", []) + env_attrs

    # [Module 2] Suggest Short Names
    attrs_mod2 = suggest_short_names(all_raw_attrs)
    
    # [Module 3] Assign Namespaces
    attrs_mod3 = assign_namespaces(attrs_mod2, relation.get("subject"), relation.get("object"))
    
    # [Module 4] Identify Categories
    attrs_mod4 = identify_categories(attrs_mod3, sentence)

    # [Module 5] Annotate data type
    if HAS_DTYPE:
        attrs_mod5 = annotate_attributes_with_type(attrs_mod4)
    else:
        # Fallback if module 5 missing
        attrs_mod5 = []
        for a in attrs_mod4:
            a["data_type"] = "String"
            attrs_mod5.append(a)

    relation["attributes"] = attrs_mod5
    
    # Lưu vào DB (tuỳ chọn)
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