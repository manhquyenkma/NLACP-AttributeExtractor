import sys
import os
import re
import spacy

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    nlp = None # Assume loaded in pipeline if failed

# Module 2: Suggesting attributes' short names
# Steps: 
# (1) Construct the value space of an attribute 
# (2) Assign a key reference to the attribute value space

# FIX 2: Trigger words phải giữ nguyên — không được strip khỏi env attrs
ENV_TRIGGER_WORDS = {
    "during", "between", "after", "before", "within",
    "throughout", "until", "from", "at", "inside",
    "outside", "via", "through", "using", "on"
}

def standardize_value(value_str, preserve_triggers=False):
    """
    Cleans up the value string by lemmatizing and removing stop words
    so that variations like "lab procedures" and "the lab procedure"
    map to the same short name key.
    
    FIX 2: If preserve_triggers=True, keep ENV_TRIGGER_WORDS even if they
    are stop words (e.g. "during" in "during business hours").
    """
    if not value_str:
        return ""
        
    if nlp is None:
        return re.sub(r'\W+', '_', value_str.lower()).strip('_')
        
    doc = nlp(value_str.lower())
    clean_tokens = []
    
    for token in doc:
        # FIX 2: giữ trigger words cho env attrs
        if preserve_triggers and token.text in ENV_TRIGGER_WORDS:
            clean_tokens.append(token.text)
            continue
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
        name  = attr.get("name", "")
        value = attr.get("value", "")
        cat   = attr.get("category", "")

        # FIX 2: detect env attrs — không có name hoặc category là temporal/spatial
        is_env = cat in ("temporal", "spatial") or not name
        if is_env:
            # Giữ trigger words để bảo toàn semantic context
            short_name = standardize_value(value, preserve_triggers=True)
        elif value:
            combined   = f"{name} {value}" if name and name not in value else str(value)
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
