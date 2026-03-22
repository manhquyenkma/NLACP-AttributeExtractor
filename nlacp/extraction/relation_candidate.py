import spacy
import json
import os

# ===================================================================
# relation_candidate.py
# Module 1: Attribute Extraction (Alohaly et al. 2019)
#
# Top-5 Dependency Patterns (theo FIX 1 — Roadmap):
#   Pattern 1: nsubj + amod      → "senior nurse"
#   Pattern 2: nsubj + compound  → "lab technician"
#   Pattern 3: nsubj + prep      → "nurse at hospital"
#   Pattern 4: dobj + amod       → "approved records"
#   Pattern 5: pobj + amod       → "approved procedures"
# ===================================================================

from nlacp.utils.nlp_utils import get_spacy_model
nlp = get_spacy_model()

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj"}
OBJECT_DEPS  = {"dobj", "pobj", "attr"}
ATTR_DEPS    = {"amod", "compound", "acl", "prep", "nummod"}

STOPWORDS = {"a", "an", "the", "his", "her", "its", "their",
             "list", "full", "all", "this", "that",
             "in", "at", "of", "on", "by", "to", "for", "with",
             "from", "up", "into", "as", "about", "over", "under"}

ENV_PREPS = {"during", "between", "after", "before", "within",
             "throughout", "until", "from", "via", "through"}

def parse_sentence(sentence):
    """Tokenize + POS tag + dependency parse."""
    doc = nlp(sentence)
    tokens = []
    for token in doc:
        tokens.append({
            "text":     token.text,
            "lemma":    token.lemma_,
            "pos":      token.pos_,
            "dep":      token.dep_,
            "head":     token.head.text,
            "ent_type": token.ent_type_
        })
    return tokens


def extract_relations(sentence, tokens):
    """
    Trích xuất subject, action, object và attributes từ
    dependency tree theo Top-5 patterns của bài báo.

    Mỗi attribute có:
        name     — modifier text
        value    — element nó bổ nghĩa
        category — "subject" hoặc "object"
        dep      — dependency relation dùng để tìm ra nó
    """
    doc = nlp(sentence)

    subject  = None
    raw_actions = []
    obj_dobj = None   # direct object — ưu tiên cao nhất
    obj_pobj = None   # prepositional object — dùng nếu không có dobj
    obj      = None
    attributes = []

    # ── Tìm subject, action(s), object ──
    for token in doc:
        if token.dep_ in SUBJECT_DEPS:
            subject = token.text

        # Bắt root text cho action, và các liên từ nối với nó (conj)
        if token.dep_ == "ROOT" and token.pos_ in ("VERB", "NOUN"):
            raw_actions.append(token.lemma_)
            for child in token.children:
                if child.dep_ == "conj" and child.pos_ in ("VERB", "NOUN"):
                    raw_actions.append(child.lemma_)
        elif token.pos_ == "VERB" and token.dep_ != "aux":
            if token.lemma_ not in raw_actions:
               raw_actions.append(token.lemma_)

        # FIX 1a: ưu tiên dobj > pobj, bỏ qua pobj của env prep
        if token.dep_ == "dobj":
            obj_dobj = token.text
        elif token.dep_ in ("pobj", "attr"):
            # Chỉ nhận pobj nếu head KHÔNG phải prep của environment
            if token.head.dep_ == "prep" and token.head.text.lower() in ENV_PREPS:
                continue
            if obj_pobj is None:  # keep first valid pobj
                obj_pobj = token.text

    # FIX 7: Handle "light noun" object coreference pattern
    LIGHT_NOUNS = {"list", "set", "group", "collection", "series", "range", "array", "type", "kind", "class"}
    if obj_dobj and obj_dobj.lower() in LIGHT_NOUNS:
        for token in doc:
            if token.text == obj_dobj and token.dep_ == "dobj":
                for child in token.children:
                    if child.dep_ == "prep" and child.text.lower() == "of":
                        for pobj in child.children:
                            if pobj.dep_ == "pobj":
                                obj_dobj = pobj.text

    obj = obj_dobj or obj_pobj   # dobj luôn thắng pobj

    # Map raw actions to CRUD
    crud_map = {
        "read": "Read", "view": "Read", "access": "Read", "see": "Read", "audit": "Read", "get": "Read",
        "write": "Write",
        "create": "Create", "make": "Create", "add": "Create", "insert": "Create", "upload": "Create",
        "update": "Update", "modify": "Update", "change": "Update", "edit": "Update", "approve": "Update", "request": "Update",
        "delete": "Delete", "remove": "Delete", "destroy": "Delete", "drop": "Delete"
    }

    actions = []
    for act in raw_actions:
        mapped = crud_map.get(act.lower(), act)
        if mapped not in actions:
            actions.append(mapped)

    # ── Attributes: duyệt qua token, lấy children theo ATTR_DEPS ──
    for token in doc:
        if token.dep_ in SUBJECT_DEPS:
            category = "subject"
        elif token.dep_ in OBJECT_DEPS:
            if token.dep_ in ("pobj",) and token.head.text.lower() in ENV_PREPS:
                continue
            category = "object"
        else:
            continue

        for child in token.children:
            if child.dep_ in ATTR_DEPS:
                name = child.text.lower()
                if name in STOPWORDS:
                    continue
                # FIX 1b: bỏ qua prep trỏ đến temporal/spatial phrase
                if child.dep_ == "prep" and name in ENV_PREPS:
                    continue
                attributes.append({
                    "name":     child.text,
                    "value":    token.text,
                    "category": category,
                    "dep":      child.dep_
                })

    # Loại bỏ trùng lặp (name + category)
    seen   = set()
    unique = []
    for attr in attributes:
        key = (attr["name"].lower(), attr["category"])
        if key not in seen:
            seen.add(key)
            unique.append(attr)

    return {
        "sentence":   sentence,
        "subject":    subject,
        "actions":    actions,
        "object":     obj,
        "attributes": unique
    }


def _guess_category(element, relation):
    """Đoán xem element thuộc về subject hay object."""
    subj = (relation.get("subject") or "").lower()
    obj  = (relation.get("object")  or "").lower()
    elem = element.lower()
    if elem and elem in subj:
        return "subject"
    if elem and elem in obj:
        return "object"
    return "unknown"


def generate_candidates(sentence):
    """
    Sinh tất cả candidate pairs (element, modifier) — đúng + sai.
    Positive: các cặp dep thực sự từ parse tree (từ extract_relations).
    Negative: các cặp không có dep relation.
    Output định dạng dictionary cho pipeline/CNN.
    """
    doc = nlp(sentence)
    tokens = [ { "text": t.text, "lemma": t.lemma_, "pos": t.pos_, "dep": t.dep_, "head": t.head.text, "ent_type": t.ent_type_ } for t in doc ]
    relation = extract_relations(sentence, tokens)
    
    positives = set()
    for attr in relation["attributes"]:
        positives.add((attr["value"].lower(), attr["name"].lower()))
        
    nouns = [t.text for t in doc if t.pos_ in ("NOUN", "PROPN") and not t.is_stop]
    mods  = [t.text for t in doc if t.pos_ in ("ADJ", "NOUN", "VERB") and not t.is_stop] # VERB (participles)
    
    candidates = []
    seen_pairs = set()
    for n in nouns:
        for m in mods:
            if n.lower() == m.lower() or m in STOPWORDS:
                continue
            pair_key = (n.lower(), m.lower())
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            
            is_pos = pair_key in positives
            candidates.append({
                "element":  n,
                "modifier": m,
                "valid":    is_pos,
                "category": _guess_category(n, relation)
            })
            
    return {
        "sentence":   sentence,
        "subject":    relation["subject"],
        "actions":    relation["actions"],
        "object":     relation["object"],
        "candidates": candidates,
        # Giữ lại raw attributes để fallback pipeline cũ
        "attributes": relation["attributes"]
    }


if __name__ == "__main__":
    tests = [
        "An on-call senior nurse may change the list of approved lab procedures.",
        "A junior lab technician can request follow-up lab procedures.",
        "Managers in the finance department can approve expense reports.",
        "Students enrolled in the course can access lecture materials.",
        "A registered patient may view his full health record.",
    ]
    for s in tests:
        tokens = parse_sentence(s)
        result = extract_relations(s, tokens)
        print(f"INPUT:   {s}")
        print(f"Subject: {result['subject']} | Actions: {result['actions']} | Object: {result['object']}")
        for attr in result["attributes"]:
            print(f"  [{attr['category']}] {attr['name']!r} → {attr['value']!r}  (dep:{attr['dep']})")
        print()