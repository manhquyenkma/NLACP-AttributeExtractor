# Module 4: Identifying the Category of an Attribute
# The paper suggests classifying parsed attributes into Target Categories
# (Subject, Action, Object, Environment). 
# Our pipeline already handles some categorization via dependency parsing (subject/object)
# and Regex/NER (temporal/spatial). This module ensures any orphaned attribute is caught.

def identify_categories(attributes, sentence):
    """
    Refines and verifies the 'category' tag of extracted attributes.
    If it's missing, it guesses based on the namespace or contextual clues.
    """
    categorized = []
    
    # Action verbs are already extracted into the 'actions' key in Mod 1, 
    # but some attributes might actually be action-modifiers (e.g. "view remotely").
    
    for attr in attributes:
        cat = attr.get("category", "")
        ns = attr.get("namespace", "")
        
        # If Category is known and strong, keep it
        if cat in ["subject", "object", "temporal", "spatial", "action"]:
            pass 
        else:
            # Fallback deduction based on namespace
            if ns.startswith("subject:"):
                attr["category"] = "subject"
            elif ns.startswith("object:"):
                attr["category"] = "object"
            elif ns.startswith("environment:"):
                # Usually we want generic 'environment' in final ABAC output
                attr["category"] = "environment"
            else:
                attr["category"] = "context" # generic fallback
                
        # Remap temporal/spatial to the broader 'environment' category for standard ABAC
        if attr["category"] in ["temporal", "spatial"]:
            attr["sub_category"] = attr["category"]
            attr["category"] = "environment"
            
        categorized.append(attr)
        
    return categorized

if __name__ == "__main__":
    sample_attrs = [
        {"namespace": "subject:role:senior_nurse", "category": "subject"},
        {"namespace": "environment:time:business_hour", "category": "temporal"},
        {"namespace": "unknown:approved", "category": ""}
    ]
    
    print("Testing Module 4: Category Identification")
    results = identify_categories(sample_attrs, "A senior nurse views records during business hours.")
    for r in results:
        print(f"Namespace: '{r['namespace']}' -> Final Category: '{r['category']}'")
