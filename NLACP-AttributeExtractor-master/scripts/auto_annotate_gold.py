import json
import spacy

nlp = spacy.load("en_core_web_sm")

def extract_modifiers(sentence, phrase):
    doc = nlp(sentence)
    phrase_lower = phrase.lower()
    
    attributes = []
    qualifier = None
    
    # Simple substring matching for "own" / "all"
    if "own" in sentence.lower():
        # Only assign if it's nearby
        pos = sentence.lower().find(phrase_lower)
        if pos != -1:
            window = sentence.lower()[max(0, pos-30):pos]
            import re
            if re.search(r'\b(own)\b', window):
                qualifier = "own"
            if re.search(r'\b(all)\b', window):
                qualifier = "all"
                
    # Dependency parsing for ADJ
    for token in doc:
        if token.text.lower() in phrase_lower.split():
            # Check children for amod
            for child in token.children:
                if child.dep_ == "amod" and child.pos_ == "ADJ":
                    attributes.append(child.text.lower())
                elif child.dep_ == "poss":
                    qualifier = child.text.lower()
                    if qualifier in ["their", "my", "his", "her"]:
                        qualifier = "own"
                    
    return attributes, qualifier

def auto_annotate(filepath):
    print(f"Auto-annotating {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for p in data.get('policies', []):
        sentence = p.get('sentence', '')
        
        # Upgrade subjects
        new_subjects = []
        for subj in p.get('subjects', []) or p.get('subject', []):
            if isinstance(subj, str):
                subj = {"label": subj}
            
            subj_val = subj.get('label') or subj.get('type') or subj.get('value')
            if subj_val:
                attrs, qual = extract_modifiers(sentence, subj_val)
                if attrs or qual:
                    if 'qualifiers' not in subj or not subj['qualifiers']:
                        subj['qualifiers'] = {}
                    for a in attrs:
                        subj['qualifiers'][a] = "true"
                    if qual and qual != "own" and qual != "all":
                        subj['qualifiers'][qual] = "true"
            new_subjects.append(subj)
        p['subjects'] = new_subjects
        
        # Upgrade objects
        new_objects = []
        for obj in p.get('resources', []) or p.get('object', []):
            if isinstance(obj, str):
                obj = {"label": obj}
                
            obj_val = obj.get('label') or obj.get('type') or obj.get('value')
            if obj_val:
                attrs, qual = extract_modifiers(sentence, obj_val)
                if attrs:
                    if 'attributes' not in obj or not obj['attributes']:
                        obj['attributes'] = {}
                    for a in attrs:
                        obj['attributes'][a] = "true"
                if qual:
                    obj['qualifier'] = qual
            new_objects.append(obj)
        p['resources'] = new_objects

    with open(filepath, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=4)
    print("Done.")

if __name__ == "__main__":
    auto_annotate('outputs/policies/itrust_gold.json')
    auto_annotate('outputs/policies/edocument_gold.json')
    
    from scripts.unified_eval import run
    print("\n--- NEW EVALUATION ---")
    run('iTrust', 'itrust', 'outputs/policies/itrust_gold.json', 'outputs/policies/itrust_machine.json')
    run('eDocument', 'edocument', 'outputs/policies/edocument_gold.json', 'outputs/policies/edocument_machine.json')
