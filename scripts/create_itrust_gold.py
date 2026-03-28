#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
create_itrust_gold.py — Auto-generate gold annotation for iTrust_gold.txt
==========================================================================

Uses spaCy dependency parsing + the project's own extractors to build a
gold CSV for iTrust sentences, saved to:
  dataset/itrust_annotation_gold.csv

Columns match annotation_sheet.csv + policy_dataset layout:
  ID, Source, Sentence, subject, object, actions,
  temporal_gold, spatial_gold, temporal_final, spatial_final
"""

import sys, io, os, csv, re, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

import spacy
from nlacp.extraction.env_extractor import extract_env_attributes

nlp = spacy.load("en_core_web_sm")

ITRUST_TXT  = os.path.join(PROJECT_ROOT, "iTrust_gold.txt")
OUT_CSV     = os.path.join(PROJECT_ROOT, "dataset", "itrust_annotation_gold.csv")

MODAL_VERBS = {"can", "may", "shall", "must", "will", "should", "could", "would"}
ACCESS_VERBS = {
    "view","read","access","see","check","monitor","inspect","browse",
    "create","add","insert","upload","submit","enter","register","generate",
    "edit","update","modify","change","set","configure","assign","record",
    "delete","remove","clear","deactivate",
    "download","export","print",
    "approve","reject","grant","revoke","deny","enable","disable",
    "schedule","refer","report","review","audit","sort","filter","graph",
    "send","reply","receive","maintain","use","manage","search","select",
    "annotate","document","list","log","indicate","provide","choose",
    "designate","undesignate","reassign","report","obtain","apply",
}

def _norm(text: str) -> str:
    if not text:
        return ""
    text = text.strip()
    # remove leading articles
    text = re.sub(r"^(the|an?|their|its)\s+", "", text, flags=re.I)
    return text.strip()

def extract_subject(doc):
    """Find the grammatical subject of the main clause."""
    for tok in doc:
        if tok.dep_ in ("nsubj", "nsubjpass") and tok.head.dep_ in ("ROOT", "aux", "xcomp"):
            # Get the full noun phrase
            span = tok.subtree
            phrase = " ".join(t.text for t in span
                              if t.dep_ not in ("relcl", "acl", "prep")
                              and not any(a.dep_ in ("relcl","prep","acl") for a in t.ancestors
                                          if a in list(span) and a != tok))
            # Simpler: just the tok's noun chunk
            for chunk in doc.noun_chunks:
                if tok in chunk:
                    return _norm(chunk.text)
            return _norm(tok.text)
    # Fallback: first noun chunk
    chunks = list(doc.noun_chunks)
    if chunks:
        return _norm(chunks[0].text)
    return ""

def extract_object(doc):
    """Find the grammatical object of the main verb."""
    for tok in doc:
        if tok.dep_ in ("dobj", "attr", "pobj") and tok.pos_ in ("NOUN","PROPN","PRON"):
            # skip "ability", "list", "way" — meta nouns
            if tok.lemma_ in {"ability","way","access","right"}:
                continue
            for chunk in doc.noun_chunks:
                if tok in chunk:
                    return _norm(chunk.text)
            return _norm(tok.text)
    return ""

def extract_actions(doc):
    """Extract action verbs (after modals or as main verbs)."""
    actions = set()
    for tok in doc:
        if tok.pos_ == "VERB" and tok.lemma_.lower() in ACCESS_VERBS:
            actions.add(tok.lemma_.lower().capitalize())
        # Also catch verb chunks after modal
        if tok.lower_ in MODAL_VERBS:
            for child in tok.head.children:
                if child.pos_ == "VERB" and child.lemma_.lower() in ACCESS_VERBS:
                    actions.add(child.lemma_.lower().capitalize())
    # Also scan coordinated verbs
    for tok in doc:
        if tok.dep_ == "conj" and tok.pos_ == "VERB" and tok.lemma_.lower() in ACCESS_VERBS:
            actions.add(tok.lemma_.lower().capitalize())
    if not actions:
        # Root verb fallback
        for tok in doc:
            if tok.dep_ == "ROOT" and tok.pos_ == "VERB":
                actions.add(tok.lemma_.lower().capitalize())
    return sorted(actions)

def extract_context(sent: str):
    """Temporal and spatial context using env_extractor."""
    temporal, spatial = "", ""
    try:
        envs = extract_env_attributes(sent)
        for e in envs:
            t = e.get("type", "").lower()
            v = e.get("value", "").strip()
            if not v:
                continue
            if "time" in t or "temporal" in t:
                temporal = v if not temporal else temporal + "; " + v
            elif "location" in t or "spatial" in t:
                spatial = v if not spatial else spatial + "; " + v
    except Exception:
        pass
    return temporal, spatial


def main():
    # Load sentences
    with open(ITRUST_TXT, encoding="utf-8") as f:
        sentences = [ln.strip() for ln in f if ln.strip()]

    print(f"[INFO] Processing {len(sentences)} iTrust sentences...")

    rows = []
    for i, sent in enumerate(sentences):
        sid = f"IT{i+1:04d}"
        doc = nlp(sent)

        subj    = extract_subject(doc)
        obj     = extract_object(doc)
        acts    = extract_actions(doc)
        temporal, spatial = extract_context(sent)

        rows.append({
            "ID"            : sid,
            "Source"        : "itrust",
            "Sentence"      : sent,
            "subject"       : subj,
            "object"        : obj,
            "actions"       : "|".join(acts),   # pipe-separated list
            "temporal_gold" : temporal,
            "spatial_gold"  : spatial,
            "temporal_final": temporal,
            "spatial_final" : spatial,
            "note"          : "auto-annotated",
            "annotator"     : "spacy+rule",
            "status"        : "auto",
        })

        if (i + 1) % 20 == 0:
            print(f"  ... {i+1}/{len(sentences)} done")

    # Write CSV
    fieldnames = ["ID","Source","Sentence","subject","object","actions",
                  "temporal_gold","spatial_gold","temporal_final","spatial_final",
                  "note","annotator","status"]
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[OK] Saved {len(rows)} rows → {OUT_CSV}")

    # Preview first 5
    print("\nPreview (first 5 rows):")
    for r in rows[:5]:
        print(f"  {r['ID']} | subj={r['subject']!r:30s} | obj={r['object']!r:30s} | acts={r['actions']!r:30s} | temp={r['temporal_final']!r}")


if __name__ == "__main__":
    main()
