"""
Canonical Gold Schema Converter
Normalizes VACT, iTrust, and eDocument Gold Standards into a single unified ABAC schema.

Unified Schema per Policy:
{
    "id": int,
    "sentence": str,
    "subjects": [
        {"role": str, "entity_type": str, "qualifiers": dict}
    ],
    "actions": [
        {"verb": str, "operation": str}
    ],
    "resources": [
        {"label": str, "qualifier": str, "attributes": dict}
    ],
    "environments": [
        {"trigger_phrase": str, "env_type": str}
    ]
}
"""

import json
import os
import re

PREPS = {"in", "on", "at", "during", "from", "within", "between", "for", "if", "only_if", "only_when", "even_if", "only"}

def clean_prep(s: str) -> str:
    if not isinstance(s, str):
        return ""
    words = s.strip().split()
    while words and words[0].lower() in PREPS:
        words = words[1:]
    return " ".join(words)

def canonicalize_vact(raw_policies):
    canonical = []
    for p in raw_policies:
        # Subjects
        subjs = []
        raw_s = p.get("subjects", []) or p.get("subject", [])
        if isinstance(raw_s, str): raw_s = [{"role": raw_s}]
        for s in raw_s:
            if isinstance(s, dict):
                role = s.get("role") or s.get("entity_type") or s.get("label") or ""
                subjs.append({
                    "role": role,
                    "entity_type": s.get("entity_type", "user"),
                    "qualifiers": s.get("qualifiers", {})
                })
            elif isinstance(s, str) and s.strip():
                subjs.append({"role": s.strip(), "entity_type": "user", "qualifiers": {}})

        # Actions
        acts = []
        for a in p.get("actions", []):
            if isinstance(a, dict):
                acts.append({"verb": a.get("verb", ""), "operation": a.get("operation", "")})
            elif isinstance(a, str) and a.strip():
                acts.append({"verb": a.strip(), "operation": ""})

        # Resources
        res = []
        raw_r = p.get("resources", []) or [p.get("resource", {})]
        if isinstance(raw_r, dict): raw_r = [raw_r]
        for r in raw_r:
            if isinstance(r, dict) and r.get("label"):
                res.append({
                    "label": r.get("label", ""),
                    "qualifier": r.get("qualifier", ""),
                    "attributes": r.get("attributes", {})
                })
            elif isinstance(r, str) and r.strip():
                res.append({"label": r.strip(), "qualifier": "", "attributes": {}})

        # Environments
        envs = []
        for e in p.get("environments", []):
            if isinstance(e, dict):
                phrase = clean_prep(e.get("trigger_phrase", ""))
                if phrase:
                    envs.append({"trigger_phrase": phrase, "env_type": e.get("env_type", "general")})

        canonical.append({
            "id": p["id"],
            "sentence": p["sentence"],
            "subjects": subjs,
            "actions": acts,
            "resources": res,
            "environments": envs
        })
    return canonical

def canonicalize_itrust(raw_policies):
    canonical = []
    for p in raw_policies:
        # Subjects
        subjs = []
        raw_s = p.get("subject", [])
        if isinstance(raw_s, str): raw_s = [raw_s]
        for s in raw_s:
            if isinstance(s, str) and s.strip():
                subjs.append({"role": s.strip(), "entity_type": "user", "qualifiers": {}})

        # Actions
        acts = []
        for a in p.get("actions", []):
            if isinstance(a, str) and a.strip():
                acts.append({"verb": a.strip(), "operation": a.strip()})

        # Resources
        res = []
        raw_o = p.get("object", [])
        if isinstance(raw_o, str): raw_o = [raw_o]
        for o in raw_o:
            if isinstance(o, str) and o.strip():
                res.append({"label": o.strip(), "qualifier": "", "attributes": {}})

        # Environments
        envs = []
        for e in p.get("environment", []):
            if isinstance(e, dict):
                phrase = clean_prep(e.get("full_value") or e.get("phrase", ""))
                if phrase:
                    envs.append({"trigger_phrase": phrase, "env_type": e.get("env_type", "general")})
            elif isinstance(e, str) and e.strip():
                phrase = clean_prep(e.strip())
                if phrase:
                    envs.append({"trigger_phrase": phrase, "env_type": "general"})

        canonical.append({
            "id": p["id"],
            "sentence": p["sentence"],
            "subjects": subjs,
            "actions": acts,
            "resources": res,
            "environments": envs
        })
    return canonical

def canonicalize_edoc(raw_policies):
    canonical = []
    for p in raw_policies:
        # Subjects
        subjs = []
        raw_s = p.get("subjects", []) or p.get("subject", [])
        if isinstance(raw_s, str): raw_s = [raw_s]
        for s in raw_s:
            if isinstance(s, str) and s.strip():
                subjs.append({"role": s.strip(), "entity_type": "user", "qualifiers": {}})

        # Actions
        acts = []
        for a in p.get("actions", []):
            if isinstance(a, dict):
                acts.append({"verb": a.get("verb", ""), "operation": a.get("operation", "")})
            elif isinstance(a, str) and a.strip():
                acts.append({"verb": a.strip(), "operation": ""})

        # Resources
        res = []
        raw_o = p.get("object", [])
        if isinstance(raw_o, str): raw_o = [raw_o]
        for o in raw_o:
            if isinstance(o, str) and o.strip():
                res.append({"label": o.strip(), "qualifier": "", "attributes": {}})

        # Environments
        envs = []
        raw_e = p.get("environment", [])
        if isinstance(raw_e, str): raw_e = [raw_e]
        for e in raw_e:
            if isinstance(e, str) and e.strip():
                phrase = clean_prep(e.strip())
                if phrase:
                    envs.append({"trigger_phrase": phrase, "env_type": "general"})

        canonical.append({
            "id": p["id"],
            "sentence": p["sentence"],
            "subjects": subjs,
            "actions": acts,
            "resources": res,
            "environments": envs
        })
    return canonical

def convert_all():
    os.makedirs("outputs/policies/canonical", exist_ok=True)
    
    # 1. VACT
    vact_raw = json.load(open("outputs/policies/vact_gold.json", encoding="utf-8"))
    vact_canon = canonicalize_vact(vact_raw.get("policies", vact_raw))
    with open("outputs/policies/canonical/vact_gold_canonical.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": "VACT", "policies": vact_canon}, f, indent=2, ensure_ascii=False)
    print(f"VACT canonicalized: {len(vact_canon)} policies")

    # 2. iTrust
    itrust_raw = json.load(open("outputs/policies/itrust_gold.json", encoding="utf-8"))
    itrust_canon = canonicalize_itrust(itrust_raw.get("policies", itrust_raw))
    with open("outputs/policies/canonical/itrust_gold_canonical.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": "iTrust", "policies": itrust_canon}, f, indent=2, ensure_ascii=False)
    print(f"iTrust canonicalized: {len(itrust_canon)} policies")

    # 3. eDocument
    edoc_raw = json.load(open("outputs/policies/edocument_gold.json", encoding="utf-8"))
    edoc_canon = canonicalize_edoc(edoc_raw.get("policies", edoc_raw))
    with open("outputs/policies/canonical/edocument_gold_canonical.json", "w", encoding="utf-8") as f:
        json.dump({"dataset": "eDocument", "policies": edoc_canon}, f, indent=2, ensure_ascii=False)
    print(f"eDocument canonicalized: {len(edoc_canon)} policies")

if __name__ == "__main__":
    convert_all()
