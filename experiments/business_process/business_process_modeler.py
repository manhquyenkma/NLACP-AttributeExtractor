import os
import spacy
import json

try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    pass

# ===================================================================
# Block 1: Model business process
#
# Input: High-level requirement specifications (text)
# Output: Namespaces of actions/activities (ABAC structure)
#
# Process:
# 1. Parse natural language requirements.
# 2. Identify the core business process or subsystem.
# 3. Extract actions and their corresponding objects.
# 4. Format into a structured namespace (e.g., Process:Action:Object).
# ===================================================================

def extract_namespaces_from_specs(text):
    """
    Parses a single high-level specification sentence and generates 
    namespaces representing the action/activity.
    """
    doc = nlp(text)
    
    # 1. Identify the subsystem/process (look for subject compounds like "management system", "registration module")
    process_name = "System"
    for chunk in doc.noun_chunks:
        if "system" in chunk.text.lower() or "module" in chunk.text.lower() or "process" in chunk.text.lower() or "subsystem" in chunk.text.lower() or "portal" in chunk.text.lower():
            # Clean up determiners
            clean_tokens = [t.text for t in chunk if t.pos_ not in ["DET", "PUNCT"]]
            process_name = "_".join(clean_tokens).capitalize()
            break
            
    # 2. Extract Key Verbs (Activities) and their direct objects
    namespaces = []
    
    for token in doc:
        if token.pos_ == "VERB" and token.dep_ != "aux":
            action = token.lemma_.capitalize()
            
            # Find the direct object associated with this verb
            obj = "Resource"
            for child in token.children:
                if child.dep_ in ["dobj", "pobj", "attr"]:
                    # Get the full noun chunk for the object
                    for chunk in doc.noun_chunks:
                        if child in chunk:
                            clean_tokens = [t.lemma_ for t in chunk if t.pos_ not in ["DET", "PRON", "PUNCT"]]
                            if clean_tokens:
                                obj = "_".join(clean_tokens).capitalize()
                            break
                    break
                    
            # Generate the formal namespace
            # Definition: Namespace of Activity = ProcessName:Action:Object
            namespace = f"{process_name}:{action}:{obj}"
            
            # Filter out generic verbs that aren't business activities (like "allows", "requires", "enables", "permits")
            if action.lower() not in ["allow", "require", "enable", "permit", "need"]:
                namespaces.append({
                    "process": process_name,
                    "action": action,
                    "object": obj,
                    "activity_namespace": namespace,
                    "original_sentence": text.strip()
                })
                
    return namespaces

def process_specifications_file(filepath):
    """
    Reads a file of high-level requirements and models the business processes.
    """
    if not os.path.exists(filepath):
        print(f"Error: File not found - {filepath}")
        return []
        
    all_namespaces = []
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip() and not line.strip().startswith("#"):
                ns = extract_namespaces_from_specs(line)
                all_namespaces.extend(ns)
                
    return all_namespaces

if __name__ == "__main__":
    sample_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw", "high_level_specs.txt")
    print(f"Reading High-Level Requirements from: {sample_path}\n")
    
    results = process_specifications_file(sample_path)
    
    print("--- Generated Namespaces of Actions/Activities (Block 1) ---\n")
    for r in results:
        print(f"Spec:      {r['original_sentence']}")
        print(f"Namespace: {r['activity_namespace']}\n")
        
    # Optionally save to JSON for Block 3
    out_path = os.path.join(os.path.dirname(sample_path), "modeled_business_processes.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)
