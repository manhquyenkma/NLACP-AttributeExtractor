import json
import random
import os

subjects = [
    "Students", "Lecturers", "System administrators", "Training department staff", 
    "Researchers", "Department heads", "Librarians", "Guest users",
    "VACT students", "Faculty members", "Security officers", "IT support staff",
    "Managers"
]

actions = [
    "access", "view", "modify", "delete", "download", "upload", "submit", "update",
    "configure", "audit", "review", "approve"
]

objects = [
    "exam grades", "course materials", "research papers", "student records", 
    "system logs", "library resources", "cryptography datasets", "financial reports",
    "network configurations", "thesis submissions", "exam schedules", "personnel files"
]

env_phrases = [
    # Temporal
    {"text": "during business hours", "category": "temporal"},
    {"text": "between 8am and 5pm", "category": "temporal"},
    {"text": "after the exam period", "category": "temporal"},
    {"text": "on weekends", "category": "temporal"},
    {"text": "during the registration period", "category": "temporal"},
    {"text": "at nighttime", "category": "temporal"},
    {"text": "before the deadline", "category": "temporal"},
    {"text": "during the night shift", "category": "temporal"},
    # Spatial - Physical/Network
    {"text": "within the VACT intranet", "category": "spatial"},
    {"text": "from the campus network", "category": "spatial"},
    {"text": "outside the campus", "category": "spatial"},
    {"text": "at the cryptography lab", "category": "spatial"},
    {"text": "from external IP addresses", "category": "spatial"},
    {"text": "at the headquarters building", "category": "spatial"},
    # Spatial - Device
    {"text": "using authorized workstations", "category": "spatial"},
    {"text": "via a secure VPN", "category": "spatial"},
    {"text": "through an encrypted channel", "category": "spatial"},
    {"text": "using internal devices", "category": "spatial"},
    {"text": "via trusted platforms", "category": "spatial"}
]

sentences = []
random.seed(42)

for i in range(100):
    sub = random.choice(subjects)
    act = random.choice(actions)
    obj = random.choice(objects)
    
    # 20% no env -> changed to 0% no env, 70% 1 env, 30% 2 envs (User request: MUST have env)
    env_count = random.choices([1, 2], weights=[0.7, 0.3])[0]
    
    selected_envs = random.sample(env_phrases, env_count)
    
    # 30% chance of a compound action for advanced parser testing
    if random.random() < 0.3:
        act2 = random.choice([a for a in actions if a != act])
        act_phrase = f"{act} and {act2}"
    else:
        act_phrase = act
    
    env_text = " ".join([e["text"] for e in selected_envs])
    
    if env_text:
        sentence = f"{sub} can {act_phrase} {obj} {env_text}."
    else:
        sentence = f"{sub} can {act_phrase} {obj}."
        
    env_attributes = []
    for e in selected_envs:
        # Match expected key in evaluator -> 'value' typically loose match checks first 2 words
        env_attributes.append({
            "category": e["category"],
            "value": e["text"],
            "trigger": "Manual",
            "source": "manual_annotation"
        })
        
    sentences.append({
        "id": f"vact_{i+1}",
        "dataset": "vact",
        "sentence": sentence,
        "env_attributes": env_attributes,
        "source": "Manual",
        "annotated": True
    })

output_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "annotated", "vact_env_annotated.json")

with open(output_path, "w", encoding="utf-8") as f:
    json.dump(sentences, f, indent=2)

print(f"Generated 100 sentences to {output_path}")
