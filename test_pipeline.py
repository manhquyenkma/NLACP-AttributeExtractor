import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))
from nlp_engine import process_sentence

def main():
    sentence = "A senior nurse can read and write medical records during business hours within the hospital."
    result = process_sentence(sentence)
    
    with open("pipeline_demo.txt", "w", encoding="utf-8") as f:
        f.write(f"INPUT: {sentence}\n\n")
        f.write("--- Extracted ABAC Policy (6 Modules) ---\n")
        f.write(f"Subject:  {result.get('subject')}\n")
        f.write(f"Actions:  {result.get('actions')}\n")
        f.write(f"Object:   {result.get('object')}\n")
        
        env_strings = []
        other_attrs = []
        if result.get("attributes"):
            for attr in result.get("attributes", []):
                cat = attr.get("category", "")
                ns = attr.get("namespace", "?")
                val = attr.get("short_name", "?")
                dtype = attr.get("data_type", "String")
                formatted = f"{ns} = \"{val}\""
                
                if cat == "environment":
                    env_strings.append(formatted)
                else:
                    other_attrs.append(f"[{cat.upper()}] {formatted}")
                    
        if env_strings:
            f.write(f"Environment: {', '.join(env_strings)}\n")
        else:
            f.write("Environment: (none detected)\n")
            
        if other_attrs:
            f.write("Other Attributes:\n")
            for a in other_attrs:
                f.write(f"  {a}\n")
        else:
            f.write("Other Attributes: (none detected)\n")

if __name__ == "__main__":
    main()
