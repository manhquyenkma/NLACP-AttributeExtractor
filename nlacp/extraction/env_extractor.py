"""
env_extractor.py — Module mới: Env-Att Extraction
Hybrid approach: Rule-based + spaCy NER

Scope (17 ngày):
  T1. TEMPORAL  — during/between/after + time NP
  T2. SPATIAL   — from/at/within + location NP, device/channel

Chạy: python src/env_extractor.py
"""
import spacy

from nlacp.utils.nlp_utils import get_spacy_model
nlp = get_spacy_model()

# ─── TEMPORAL config ─────────────────────────────────────────────
TEMPORAL_PREPS  = {"during", "between", "after", "before",
                   "within", "throughout", "until", "at", "on"}
TEMPORAL_HINTS  = {"hours", "hour", "shift", "night", "morning",
                   "evening", "day", "weekday", "weekend",
                   "period", "schedule", "time", "pm", "am",
                   "daytime", "nighttime", "business", "working",
                   "deadline", "month", "year", "session", "semester",
                   "date", "duration"}

# ─── SPATIAL config ──────────────────────────────────────────────
SPATIAL_PREPS   = {"from", "at", "within", "inside", "outside",
                   "through", "via", "on"}
SPATIAL_HINTS   = {"network", "ward", "department", "hospital",
                   "building", "floor", "site", "premises",
                   "location", "intranet", "vpn", "system",
                   "workstation", "device", "terminal",
                   "internal", "external", "remote", "local",
                   "secure", "trusted", "encrypted",
                   "campus", "lab", "office", "room", "clinic",
                   "headquarters"}

# ─── DEVICE/CHANNEL pattern (Layer 3) ────────────────────────────
# "using" / "via" thường bị spaCy parse là advcl/acl chứ không phải prep
# Nên cần pattern riêng: tìm token có text trong DEVICE_TRIGGERS
DEVICE_TRIGGERS = {"using", "via", "through"}
DEVICE_HINTS    = {"workstation", "device", "terminal", "laptop",
                   "system", "portal", "platform", "vpn",
                   "connection", "interface", "channel",
                   "network",  "intranet",  "console",
                   "browser",  "app",       "client",
                   "trusted",  "secure",    "encrypted",
                   "managed",  "approved",  "authorized"}

# ─── thứ tự sự kiện (action sequences — không phải env-att) ──────
EVENT_WORDS     = {"meeting", "conference", "session",
                   "training", "reviewing", "submitting",
                   "processing", "approval", "audit"}

# ─── người (person nouns → subject-att, không phải spatial) ──────
PERSON_NOUNS    = {"nurse", "doctor", "staff", "manager",
                   "student", "user", "physician", "technician",
                   "administrator", "reviewer", "employee"}


def extract_env_attributes(sentence):
    """
    Trích xuất env-att từ câu NLACP.
    Trả về list of dicts:
      { category, subcategory, value, trigger, method }
    method: "rule" | "ner" | "hybrid"
    """
    doc     = nlp(sentence)
    results = []

    # ── Layer 1: Rule-based ──────────────────────────────────────
    for token in doc:
        tl = token.text.lower()

        # TEMPORAL: prep + time NP
        if tl in TEMPORAL_PREPS and token.dep_ == "prep":
            np = _get_noun_phrase(token)
            if np and _has_hint(np, TEMPORAL_HINTS):
                results.append({
                    "category":    "temporal",
                    "subcategory": _classify_temporal(np),
                    "value":       f"{token.text} {np}",
                    "trigger":     token.text,
                    "method":      "rule"
                })

        # SPATIAL: prep + location/device NP
        if tl in SPATIAL_PREPS and token.dep_ == "prep":
            np = _get_noun_phrase(token)
            if np and (_has_hint(np, SPATIAL_HINTS) or
                       _is_location_ner(doc, np)):
                results.append({
                    "category":    "spatial",
                    "subcategory": _classify_spatial(np),
                    "value":       f"{token.text} {np}",
                    "trigger":     token.text,
                    "method":      "rule"
                })

    # ── Layer 3: Device/Channel pattern ──────────────────────────
    # "using trusted workstations" → spaCy dep=advcl/acl (not prep)
    for token in doc:
        tl = token.text.lower()
        if tl in DEVICE_TRIGGERS:
            # Lấy tất cả noun/adj con của token này
            np = _get_device_phrase(token)
            if np and _has_hint(np, DEVICE_HINTS):
                results.append({
                    "category":    "spatial",
                    "subcategory": "device",
                    "value":       f"{token.text} {np}",
                    "trigger":     token.text,
                    "method":      "rule"
                })

    # ── Layer 2: spaCy NER (hybrid boost) ────────────────────────
    for ent in doc.ents:
        # Temporal entities
        if ent.label_ in ("TIME", "DATE"):
            # Kiểm tra đã capture bằng rule chưa
            already = any(
                ent.text.lower() in r["value"].lower()
                for r in results if r["category"] == "temporal"
            )
            if not already:
                results.append({
                    "category":    "temporal",
                    "subcategory": "ner_detected",
                    "value":       ent.text,
                    "trigger":     "NER:" + ent.label_,
                    "method":      "ner"
                })

        # Spatial entities (GPE, LOC, FAC)
        if ent.label_ in ("GPE", "LOC", "FAC", "ORG"):
            already = any(
                ent.text.lower() in r["value"].lower()
                for r in results if r["category"] == "spatial"
            )
            if not already:
                results.append({
                    "category":    "spatial",
                    "subcategory": "ner_" + ent.label_.lower(),
                    "value":       ent.text,
                    "trigger":     "NER:" + ent.label_,
                    "method":      "ner"
                })

    # ── Loại false positives ─────────────────────────────────────
    results = _filter_false_positives(doc, results)
    # ── Deduplicate ───────────────────────────────────────────────
    results = _deduplicate(results)
    return results


def _get_noun_phrase(prep_token):
    """Lấy noun phrase sau giới từ — dừng tại prep boundary tiếp theo.
    
    Chi tiết bug cũ: child.subtree lấy toàn bộ cây con của noun head,
    khiến "during business hours" nuốt luôn "within the hospital" khi
    spaCy gắn within-clause vào subtree của "hours".
    Fix: dừng traversal ngay khi gặp token có dep_=="prep" và token đó
    không phải chính child đang xét.
    """
    for child in prep_token.children:
        if child.pos_ in ("NOUN", "PROPN", "ADJ", "NUM"):
            tokens = []
            for t in child.subtree:
                # Dừng khi gặp prep mới (vd: "within" trong subtree của "hours")
                if t.dep_ == "prep" and t != child:
                    break
                # Dừng tại dấu câu
                if t.is_punct:
                    break
                tokens.append(t.text)
            return " ".join(tokens)
    return ""


def _has_hint(text, hints):
    tl = text.lower()
    return any(h in tl for h in hints)


def _get_device_phrase(trigger_token):
    """Lấy phrase của device trigger (using/via/through + NP).
    Xử lý cả trường hợp dep=advcl, acl, prep.
    Dừng tại prep boundary để tránh overreach."""
    parts = []
    for child in trigger_token.children:
        if child.pos_ in ("NOUN", "PROPN", "ADJ"):
            tokens = []
            for t in child.subtree:
                if t.dep_ == "prep" and t != child:
                    break
                if t.is_punct:
                    break
                tokens.append(t.text)
            if tokens:
                parts.append(" ".join(tokens))
    if parts:
        return " ".join(parts)
    # Nếu không có con trực tiếp, tìm trong head children
    head = trigger_token.head
    for child in head.children:
        if child != trigger_token and child.pos_ in ("NOUN", "PROPN"):
            tokens = []
            for t in child.subtree:
                if t.dep_ == "prep" and t != child:
                    break
                if t.is_punct:
                    break
                tokens.append(t.text)
            if tokens:
                return " ".join(tokens)
    return ""


def _is_location_ner(doc, np_text):
    for ent in doc.ents:
        if ent.text in np_text and ent.label_ in ("GPE", "LOC", "FAC"):
            return True
    return False


def _classify_temporal(np_text):
    tl = np_text.lower()
    if any(c.isdigit() for c in np_text):
        return "absolute"     # "8am", "5pm"
    if any(w in tl for w in ("emergency", "code", "situation")):
        return "event"
    if any(w in tl for w in ("weekday", "weekend", "monday", "daily")):
        return "recurring"
    return "relative"         # "business hours", "night shift"


def _classify_spatial(np_text):
    tl = np_text.lower()
    if any(w in tl for w in ("network", "vpn", "intranet", "internet",
                              "internal", "external", "remote")):
        return "network"
    if any(w in tl for w in ("workstation", "device", "terminal",
                              "laptop", "system", "portal")):
        return "device"
    return "physical"         # "Ward A", "hospital"


def _filter_false_positives(doc, results):
    """Loại bỏ false positives phổ biến."""
    filtered = []
    for r in results:
        val = r["value"].lower()
        trig = r.get("trigger", "").lower()

        # "from nurses/doctors ..." → subject-att
        if trig == "from" and any(p in val for p in PERSON_NOUNS):
            continue

        # "at meeting/conference" → event sequence
        if trig == "at" and any(e in val for e in EVENT_WORDS):
            continue

        # "after reviewing/submitting" → action sequence (VERB sau prep)
        after_words = {"reviewing", "submitting", "approving",
                       "processing", "completing", "receiving"}
        if trig == "after" and any(w in val for w in after_words):
            continue

        filtered.append(r)
    return filtered


def _deduplicate(results):
    seen = set()
    out  = []
    for r in results:
        key = (r["category"], r["value"].lower()[:30])
        if key not in seen:
            seen.add(key)
            out.append(r)
    return out


if __name__ == "__main__":
    import json

    tests = [
        "A doctor can view patient records during business hours.",
        "Nurses from the hospital network can update patient charts.",
        "A senior nurse on night shift may change approved lab procedures.",
        "Administrators using trusted workstations can modify system settings.",
        "Managers accessing from internal VPN can approve expense reports.",
        "A registered patient may view his full health record.",          # no env-att
        "Physicians within the ICU can override standard protocols.",
        "Staff can access data only between 8am and 5pm on weekdays.",
        # BOUNDARY BUG test cases
        "A senior nurse can view medical records during business hours within the hospital.",
        "Staff may access records at night in the lab.",
        "Users can submit forms on weekends.",
        "Managers can access records within the campus.",
    ]

    print("\n" + "="*60)
    print("  Env-Att Extractor Test (Hybrid Rule + NER)")
    print("="*60)

    for s in tests:
        attrs = extract_env_attributes(s)
        print(f"\nInput:  {s}")
        if attrs:
            for a in attrs:
                print(f"  [{a['category']:8s}/{a['subcategory']:12s}] "
                      f"\"{a['value']}\"  (trigger: {a['trigger']}, {a['method']})")
        else:
            print("  (no env-att detected)")
