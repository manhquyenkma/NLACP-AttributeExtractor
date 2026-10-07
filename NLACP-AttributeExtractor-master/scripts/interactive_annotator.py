import json
import os
import sys

def annotate_dataset(filepath):
    print(f"Loading dataset from {filepath}...")
    with open(filepath, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    policies = data.get('policies', [])
    updated_count = 0
    
    for i, policy in enumerate(policies):
        sentence = policy.get('sentence', '')
        
        # Check if we need to annotate subjects
        subjects = policy.get('subjects', [])
        objects = policy.get('resources', [])
        
        needs_annotation = False
        
        # For simplicity, we just prompt for every subject and object
        print(f"\n[{i+1}/{len(policies)}] SENTENCE: {sentence}")
        
        # Annotate Subjects
        for subj in subjects:
            if 'qualifiers' not in subj or not subj['qualifiers']:
                subj_name = subj.get('label') or subj.get('type') or subj.get('value')
                print(f"  Subject: '{subj_name}'")
                qual = input(f"    Enter qualifiers for Subject (comma separated, or press Enter to skip): ").strip()
                if qual:
                    # Parse comma separated like: civilian, senior
                    subj['qualifiers'] = {q.strip(): "true" for q in qual.split(',')}
                    updated_count += 1
        
        # Annotate Objects
        for obj in objects:
            if 'attributes' not in obj or not obj['attributes']:
                obj_name = obj.get('label') or obj.get('type') or obj.get('value')
                print(f"  Object: '{obj_name}'")
                
                # Check for existing qualifier (like "own" or "all")
                existing_qualifier = obj.get('qualifier')
                if existing_qualifier:
                    print(f"    (Current qualifier: {existing_qualifier})")
                    
                attr = input(f"    Enter attributes for Object (comma separated, e.g. 'comprehensive', or press Enter to skip): ").strip()
                if attr:
                    # Update attributes dictionary
                    if 'attributes' not in obj or obj['attributes'] is None:
                        obj['attributes'] = {}
                    
                    for a in attr.split(','):
                        a = a.strip()
                        obj['attributes'][a] = "true"
                    updated_count += 1
                
                qual_input = input(f"    Enter qualifier (e.g. 'own', 'all', 'patient\\'s') or Enter to skip/keep: ").strip()
                if qual_input:
                    obj['qualifier'] = qual_input
                    updated_count += 1
                    
        # Save after every sentence so progress isn't lost
        if updated_count > 0:
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=4)
                
        # Ask to continue
        cmd = input("Press Enter to continue, or 'q' to quit: ")
        if cmd.strip().lower() == 'q':
            print("Progress saved. Exiting.")
            sys.exit(0)
            
    print(f"\nDone! Made {updated_count} updates to {filepath}.")

if __name__ == "__main__":
    print("Select dataset to annotate:")
    print("1. iTrust (outputs/policies/itrust_gold.json)")
    print("2. eDocument (outputs/policies/edocument_gold.json)")
    choice = input("Enter 1 or 2: ")
    
    if choice == '1':
        annotate_dataset('outputs/policies/itrust_gold.json')
    elif choice == '2':
        annotate_dataset('outputs/policies/edocument_gold.json')
    else:
        print("Invalid choice.")
