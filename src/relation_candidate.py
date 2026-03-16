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

nlp = spacy.load("en_core_web_sm")

SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj"}
OBJECT_DEPS  = {"dobj", "pobj", "attr"}
ATTR_DEPS    = {"amod", "compound", "acl", "prep", "nummod"}

STOPWORDS = {"a", "an", "the", "his", "her", "its", "their",
             "list", "full", "all", "this", "that",
             # prepositions thường gặp (loại khi dep=prep)
             "in", "at", "of", "on", "by", "to", "for", "with",
             "from", "up", "into", "as", "about", "over", "under"}


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

    subject = None
    raw_actions = []
    obj     = None
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
            # Nếu không có ROOT verb rõ ràng, bắt thêm các động từ chính khác
            if token.lemma_ not in raw_actions:
               raw_actions.append(token.lemma_)

        if token.dep_ in OBJECT_DEPS:
            obj = token.text

    # Map raw actions to CRUD
    crud_map = {
        "read": "Read", "view": "Read", "access": "Read", "see": "Read", "audit": "Read",
        "write": "Create", "create": "Create", "make": "Create", "add": "Create", "insert": "Create",
        "update": "Update", "modify": "Update", "change": "Update", "edit": "Update",
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
            category = "object"
        else:
            continue

        for child in token.children:
            if child.dep_ in ATTR_DEPS:
                name = child.text.lower()
                if name in STOPWORDS:
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