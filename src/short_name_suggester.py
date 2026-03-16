import sys
import os
import re
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    pass # Assume loaded in pipeline if failed

# Module 2: Suggesting attributes' short names
# Steps: 
# (1) Construct the value space of an attribute 
# (2) Assign a key reference to the attribute value space

def standardize_value(value_str):
    """
    Cleans up the value string by lemmatizing and removing stop words
    so that variations like "lab procedures" and "the lab procedure"
    map to the same short name key.
    """
    if not value_str:
        return ""
        
    doc = nlp(value_str.lower())
    clean_tokens = []
    
    for token in doc:
        # Ignore common determiners and stop words that don't add meaning
        if token.is_stop or token.is_punct or token.pos_ == "DET":
            continue
        clean_tokens.append(token.lemma_)
        
    short_name = "_".join(clean_tokens)
    
    # Fallback if everything was stripped
    if not short_name:
        short_name = re.sub(r'\W+', '_', value_str.lower()).strip('_')
        
    return short_name

def suggest_short_names(attributes):
    """
    Iterates through a list of extracted attributes and assigns
    a 'short_name' suitable for policy construction (Module 2).
    """
    processed_attrs = []
    
    for attr in attributes:
        # attr is typically a dict from Module 1
        name = attr.get("name", "")
        value = attr.get("value", "")
        
        # In the context of the paper, the short name is often 
        # based on the modifier/value relationship.
        # e.g., "senior" modifying "nurse" -> "senior_nurse" or just "senior"
        
        # Construct the key reference
        if value:
            combined = f"{name} {value}" if name and name not in value else str(value)
            short_name = standardize_value(combined)
        elif name:
            short_name = standardize_value(name)
        else:
            short_name = "unknown"
            
        attr["short_name"] = short_name
        processed_attrs.append(attr)
        
    return processed_attrs

if __name__ == "__main__":
    sample_attrs = [
        {"name": "senior", "value": "nurse"},
        {"name": "lab", "value": "procedures"},
        {"name": "the full", "value": "health record"},
        {"category": "temporal", "value": "during business hours"},
        {"category": "spatial", "value": "within the VACT intranet"}
    ]
    
    print("Testing Module 2: Short Name Suggestions")
    results = suggest_short_names(sample_attrs)
    for r in results:
        print(f"Value: '{r.get('name', '')} {r.get('value', '')}' -> Short Name: '{r['short_name']}'")
