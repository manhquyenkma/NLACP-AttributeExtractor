# Module 3: Assigning Attributes to Namespaces Hierarchically
# Sub-tasks:
# (1) Computes inheritance 
# (2) Computes assigned attributes

def assign_namespaces(attributes, subject_name, object_name):
    """
    Takes a list of standard attributes (with short_names) and assigns
    them to hierarchical namespaces (e.g., subject:role:senior_nurse).
    """
    namespace_attrs = []
    
    # Very simplified inheritance / knowledge base mapping for demonstration.
    # In a real system, you'd use a taxonomy DB or WordNet.
    role_keywords = {"nurse", "technician", "manager", "staff", "student", "officer", "administrator", "librarian", "user"}
    dept_keywords = {"department", "hospital", "clinic", "lab", "ward", "office"}
    resource_keywords = {"record", "procedure", "report", "paper", "data", "log", "material", "file", "grade", "submission"}
    
    for attr in attributes:
        short_name = attr.get("short_name", "")
        cat_guess = attr.get("category", "")
        
        namespace = ""
        
        # Heuristics for Sub-attributes (Subject)
        if cat_guess == "subject" or (subject_name and short_name in subject_name.lower()):
            if any(k in short_name for k in role_keywords):
                namespace = f"subject:role:{short_name}"
            elif any(k in short_name for k in dept_keywords):
                namespace = f"subject:department:{short_name}"
            else:
                namespace = f"subject:group:{short_name}"
                
        # Heuristics for Obj-attributes (Object)
        elif cat_guess == "object" or (object_name and short_name in object_name.lower()):
            if any(k in short_name for k in resource_keywords):
                namespace = f"object:type:{short_name}"
            else:
                namespace = f"object:prop:{short_name}"
                
        # Heuristics for Environment
        elif cat_guess == "temporal":
            namespace = f"environment:time:{short_name}"
        elif cat_guess == "spatial":
            if "device" in short_name or "workstation" in short_name or "platform" in short_name:
                namespace = f"environment:device:{short_name}"
            elif "network" in short_name or "vpn" in short_name or "intranet" in short_name:
                namespace = f"environment:network:{short_name}"
            else:
                namespace = f"environment:location:{short_name}"
        else:
            namespace = f"unknown:{short_name}"
            
        attr["namespace"] = namespace
        namespace_attrs.append(attr)
        
    return namespace_attrs

if __name__ == "__main__":
    sample_attrs = [
        {"short_name": "senior_nurse", "category": "subject"},
        {"short_name": "finance_department", "category": "subject"},
        {"short_name": "health_record", "category": "object"},
        {"short_name": "business_hour", "category": "temporal"},
        {"short_name": "vact_intranet", "category": "spatial"}
    ]
    
    print("Testing Module 3: Namespace Assignment")
    results = assign_namespaces(sample_attrs, "senior nurse", "health record")
    for r in results:
        print(f"Short Name: '{r['short_name']}' -> Namespace: '{r['namespace']}'")
