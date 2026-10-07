from nlacp.utils.nlp_utils import get_spacy_model
"""
Resource Extractor for ABAC Policy v2.0

This module extracts and normalizes resource/object entities from policy sentences.
Resources can be: data, document, record, schedule, agreement, offering, registration, attribute.
"""

import spacy
import re
from typing import List, Dict, Optional, Tuple
from nlacp.validation.schema_validator import (
    Resource, EntityType, ResourceScope, ResourceAttributes, ResourceEntityType
)
from nlacp.extraction.env_extractor import PERSON_NOUNS, extract_env_attributes

nlp = None  # Lazy load

def get_spacy_model():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.blank("en")
    return nlp

# Resource dependency patterns
RESOURCE_DEPS = {"dobj", "pobj", "attr"}

# Resource type mapping (noun phrase to entity_type)
RESOURCE_TYPE_MAPPING = {
    # Data resources
    "score": ResourceEntityType.DATA,
    "scores": ResourceEntityType.DATA,
    "grade": ResourceEntityType.DATA,
    "grades": ResourceEntityType.DATA,
    "transcript": ResourceEntityType.DOCUMENT,
    "transcripts": ResourceEntityType.DOCUMENT,
    
    # Documents
    "roster": ResourceEntityType.DOCUMENT,
    "rosters": ResourceEntityType.DOCUMENT,
    "course roster": ResourceEntityType.DOCUMENT,
    "application": ResourceEntityType.DOCUMENT,
    "applications": ResourceEntityType.DOCUMENT,
    "exam": ResourceEntityType.DOCUMENT,
    "multiple choice exam": ResourceEntityType.DOCUMENT,
    "multiple-choice exam": ResourceEntityType.DOCUMENT,
    "exam details": ResourceEntityType.DOCUMENT,
    "material": ResourceEntityType.DOCUMENT,
    "learning material": ResourceEntityType.DOCUMENT,
    "learning materials": ResourceEntityType.DOCUMENT,
    "agreement": ResourceEntityType.AGREEMENT,
    "visiting lecturer agreement": ResourceEntityType.AGREEMENT,
    "contract": ResourceEntityType.DOCUMENT,
    
    # Schedules
    "class schedule": ResourceEntityType.SCHEDULE,
    "class schedules": ResourceEntityType.SCHEDULE,
    "teaching schedule": ResourceEntityType.SCHEDULE,
    "teaching schedules": ResourceEntityType.SCHEDULE,
    "visiting lecturer class": ResourceEntityType.SCHEDULE,
    "visiting lecturer classes": ResourceEntityType.SCHEDULE,
    "schedule": ResourceEntityType.SCHEDULE,
    "schedules": ResourceEntityType.SCHEDULE,
    "teaching hour": ResourceEntityType.SCHEDULE,
    "teaching hours": ResourceEntityType.SCHEDULE,
    
    # Offerings
    "course offering": ResourceEntityType.OFFERING,
    "course offerings": ResourceEntityType.OFFERING,
    
    # Registrations
    "course registration": ResourceEntityType.REGISTRATION,
    "course registrations": ResourceEntityType.REGISTRATION,
    "registration": ResourceEntityType.REGISTRATION,
    
    # Attributes
    "contract number": ResourceEntityType.ATTRIBUTE,
    "attribute": ResourceEntityType.ATTRIBUTE,
    "field": ResourceEntityType.ATTRIBUTE,
    
    # Records
    "record": ResourceEntityType.RECORD,
    "records": ResourceEntityType.RECORD,
    "student record": ResourceEntityType.RECORD,
    "medical record": ResourceEntityType.RECORD,
    
    # Medical & Admin
    "prescription": ResourceEntityType.DOCUMENT,
    "prescriptions": ResourceEntityType.DOCUMENT,
    "lab result": ResourceEntityType.DATA,
    "lab results": ResourceEntityType.DATA,
    "invoice": ResourceEntityType.DOCUMENT,
    "invoices": ResourceEntityType.DOCUMENT,
    "report": ResourceEntityType.DOCUMENT,
    "reports": ResourceEntityType.DOCUMENT,
    "account": ResourceEntityType.RECORD,
    "accounts": ResourceEntityType.RECORD,
    "ticket": ResourceEntityType.DOCUMENT,
    "tickets": ResourceEntityType.DOCUMENT,
    "system": ResourceEntityType.SYSTEM,
    "systems": ResourceEntityType.SYSTEM,
    
    # General documents & business resources
    "document": ResourceEntityType.DOCUMENT,
    "documents": ResourceEntityType.DOCUMENT,
    "file": ResourceEntityType.DOCUMENT,
    "files": ResourceEntityType.DOCUMENT,
    "policy": ResourceEntityType.DOCUMENT,
    "policies": ResourceEntityType.DOCUMENT,
    "note": ResourceEntityType.DOCUMENT,
    "notes": ResourceEntityType.DOCUMENT,
    "message": ResourceEntityType.DOCUMENT,
    "messages": ResourceEntityType.DOCUMENT,
    "notification": ResourceEntityType.DOCUMENT,
    "notifications": ResourceEntityType.DOCUMENT,
    "log": ResourceEntityType.RECORD,
    "logs": ResourceEntityType.RECORD,
    "audit_log": ResourceEntityType.RECORD,
    "audit_logs": ResourceEntityType.RECORD,
    "configuration": ResourceEntityType.DOCUMENT,
    "configurations": ResourceEntityType.DOCUMENT,
    "setting": ResourceEntityType.DOCUMENT,
    "settings": ResourceEntityType.DOCUMENT,
    "profile": ResourceEntityType.RECORD,
    "profiles": ResourceEntityType.RECORD,
    "credential": ResourceEntityType.DOCUMENT,
    "credentials": ResourceEntityType.DOCUMENT,
    "permission": ResourceEntityType.ATTRIBUTE,
    "permissions": ResourceEntityType.ATTRIBUTE,
    "right": ResourceEntityType.ATTRIBUTE,
    "rights": ResourceEntityType.ATTRIBUTE,
    "paycheck": ResourceEntityType.DOCUMENT,
    "paychecks": ResourceEntityType.DOCUMENT,
    "payroll": ResourceEntityType.DOCUMENT,
    "offer": ResourceEntityType.OFFERING,
    "offers": ResourceEntityType.OFFERING,
    "status": ResourceEntityType.ATTRIBUTE,
    "detail": ResourceEntityType.DATA,
    "details": ResourceEntityType.DATA,
    "information": ResourceEntityType.DATA,
    "data": ResourceEntityType.DATA,
    "content": ResourceEntityType.DATA,
    "contents": ResourceEntityType.DATA,
    "metadata": ResourceEntityType.DATA,
    "meta-data": ResourceEntityType.DATA,
}

# Resource label mapping (for normalization)
# NOTE: VACT Gold Standard is inconsistent with plurals. We map to singular forms
# for most resources because it matches the majority of annotations.
RESOURCE_LABEL_MAPPING = {
    "score": "score",
    "scores": "score",
    "grade": "grade",
    "grades": "grade",
    "final grades": "final_grades",
    "transcript": "transcript",
    "transcripts": "transcript",
    "roster": "roster",
    "rosters": "roster",
    "course roster": "course_roster",
    "application": "application",
    "applications": "application",
    "exam": "exam",
    "exams": "exam",
    "multiple choice exams": "multiple-choice_exams",
    "multiple-choice exams": "multiple-choice_exams",
    "multiple choice exam": "multiple-choice_exams",
    "multiple-choice exam": "multiple-choice_exams",
    "exam details": "exam_details",
    "material": "learning_material",
    "materials": "learning_material",
    "learning material": "learning_material",
    "learning materials": "learning_material",
    "agreement": "agreement",
    "agreements": "agreement",
    "visiting lecturer agreement": "visiting_lecturer_agreement",
    
    # Medical & Admin
    "prescription": "prescription",
    "prescriptions": "prescription",
    "lab result": "lab_result",
    "lab results": "lab_result",
    "invoice": "invoice",
    "invoices": "invoice",
    "report": "report",
    "reports": "report",
    "account": "account",
    "accounts": "account",
    "ticket": "ticket",
    "tickets": "ticket",
    "system": "system",
    "systems": "system",
    "visiting lecturer agreements": "visiting_lecturer_agreement",
    "lecturer agreement": "visiting_lecturer_agreement",
    "lecturer agreements": "visiting_lecturer_agreement",
    "contract": "contract",
    "contract number": "contract_number",
    "class schedule": "class_schedule",
    "class schedules": "class_schedule",
    "teaching schedule": "teaching_schedule",
    "teaching schedules": "teaching_schedule",
    "visiting lecturer class": "visiting_lecturer_class",
    "visiting lecturer classes": "visiting_lecturer_class",
    "schedule": "schedule",
    "schedules": "schedule",
    "teaching hour": "teaching_hours",
    "teaching hours": "teaching_hours",
    "hours": "teaching_hours",
    "course offering": "course_offering",
    "course offerings": "course_offering",
    "course registration": "course_registration",
    "course registrations": "course_registration",
    "course registration results": "course_registration_results",
    "registration": "registration",
    "record": "record",
    "records": "record",
    "student record": "student_record",
    "medical record": "medical_record",
    "instructor": "instructor",
    "instructors": "instructor",
    "visiting lecturer agreement information": "visiting_lecturer_agreement_information",
    "lecturer agreement information": "visiting_lecturer_agreement_information",
    "reason for a transcript revision": "reason_for_a_transcript_revision",
    "reason": "reason_for_a_transcript_revision",
    "drug interaction information": "interaction_information",
    "session timeout policies": "timeout_policy",
    "session timeout policy": "timeout_policy",
    "session timeout period": "timeout_period",
    "adverse event report": "event_report",
    "adverse event reports": "event_report",
    "meta-information of document": "document_meta_information",
    "meta-information of every document": "document_meta_information",
    "meta-information of documents": "document_meta_information",
    "meta-data of a document": "document_meta_data",
    "meta-data of document": "document_meta_data",
    "metadata of a document": "document_meta_data",
    "metadata of document": "document_meta_data",
    "content of documents": "document_content",
    "content of document": "document_content",
    "status of a sent banking note": "banking_note_status",
    "status of sent banking note": "banking_note_status",
    "status of a banking note": "banking_note_status",
    "status of banking note": "banking_note_status",
    "status of note": "banking_note_status",
}

# Scope-related keywords
SCOPE_QUALIFIERS = {
    "own": "own",
    "their": "own",
    "my": "own",
    "every": "every",
    "all": "all",
    "entire": "entire",
    "the": "the",
}

SCOPE_LEVEL_INDICATORS = {
    "that course": "course",
    "for that course": "course",
    "in that course": "course",
    "that department": "department",
    "for that department": "department",
    "in that department": "department",
    "own department": "department",
    "all courses": "all",
    "every course": "all",
    "enrolled courses": "course",
    "per year": "all",
}

# Sensitivity levels
SENSITIVITY_KEYWORDS = {
    "public": "public",
    "internal": "internal",
    "confidential": "confidential",
    "secret": "secret",
}

# Status keywords
STATUS_KEYWORDS = {
    "open": "open",
    "closed": "closed",
    "submitted": "submitted",
    "issued": "issued",
    "numbered": "numbered",
    "official": "official",
    "approved": "approved",
}

# Subject filter indicators
SUBJECT_FILTER_KEYWORDS = {
    "civilian": "civilian_student",
    "civilian student": "civilian_student",
    "civilian students": "civilian_student",
}

# Words in an "of" pobj subtree that signal a SCOPE qualifier, not a resource specifier.
# When these appear in the pobj of "X of Y", we should NOT follow "of" (Gold wants just X).
# IMPORTANT: Do NOT include 'lecturer' or 'visiting' — they can be resource descriptors
# (e.g., "details of visiting lecturer agreements" should follow "of").
OF_SCOPE_PERSON_WORDS = {
    "student", "students", "user", "users", 
    "applicant", "applicants", "coordinator", "chair",
    "staff", "faculty", "civilian", "retired",
    "course", "courses", "class", "classes",
}

# amod words to exclude from resource phrases (these are qualifiers, not resource descriptors)
# NOTE: Do NOT exclude 'suggested', 'entire', 'final', or 'preferred' — Gold preserves these
# We also exclude common action verbs that spaCy misparses as amod/compound (e.g. "delete exams")
AMOD_EXCLUDE = {
    "own", "all", "every", "issued", "registered", "new", "certain", "received", "sent",
    "upload", "download", "modify", "delete", "view", "assess", "export", "approve", "teach",
    "search", "send", "delegate", "create", "assign", "log", "read", "update"
}

def _is_person_type(token, person_nouns):
    return token.lemma_.lower() in person_nouns

def _is_env_prep(prep_token):
    """Return True if this preposition begins an environment/scope phrase, not a resource qualifier.
    Heuristic: the pobj subtree contains a system/location noun, a temporal indicator,
    or more than 3 words — all signs it is not tightly bound to the resource head.
    """
    ENV_TRIGGER_PREPS = {"in", "for", "on", "within", "between", "after", "before", "during", "at"}
    if prep_token.text.lower() not in ENV_TRIGGER_PREPS:
        return False
    pobj = next((c for c in prep_token.children if c.dep_ == "pobj"), None)
    if pobj is None:
        return False
    subtree_words = [t.text.lower() for t in pobj.subtree]
    # If pobj subtree is long (>3 content words), it's likely an environment phrase
    if len(subtree_words) > 3:
        return True
    # If pobj contains system/location nouns
    SYSTEM_WORDS = {"system", "office", "department", "application", "platform", "module", "year"}
    if set(subtree_words) & SYSTEM_WORDS:
        return True
    return False

def _get_full_object_span(dobj_token, doc) -> str:
    tokens_in_span = [dobj_token]

    # Recursive pre-head modifier collection (compound, amod, hyphen)
    def collect_modifiers(tok):
        for child in tok.children:
            if child.i < dobj_token.i:
                if child.dep_ in ("amod", "compound"):
                    if child.lemma_.lower() not in AMOD_EXCLUDE:
                        tokens_in_span.append(child)
                        collect_modifiers(child)
                elif child.dep_ == "punct" and child.text == "-":
                    tokens_in_span.append(child)
                    collect_modifiers(child)
    collect_modifiers(dobj_token)

    # Preposition phrases attached to dobj
    for child in dobj_token.children:
        if child.dep_ == "prep" and child.head == dobj_token:
            # Special case 1: "right to read / right to manage"
            if child.text.lower() in ("to",) and dobj_token.lemma_.lower() in ("right",):
                tokens_in_span.extend(list(child.subtree))
            # Special case 2: "of" for metadata/content or status/type
            elif child.text.lower() in ("of",):
                pobj = next((c for c in child.children if c.dep_ == "pobj"), None)
                if pobj is not None and not _is_person_type(pobj, PERSON_NOUNS) and not _is_env_prep(child):
                    CONTAINER_NOUNS = {
                        "content", "contents", "meta-information", "meta-data", "metadata",
                        "status", "type", "value", "detail", "details"
                    }
                    if dobj_token.lemma_.lower() in CONTAINER_NOUNS:
                        tokens_in_span.append(child)
                        tokens_in_span.append(pobj)
                        for gc in pobj.children:
                            if gc.dep_ in ("compound", "amod"):
                                tokens_in_span.append(gc)
            else:
                continue

    tokens_in_span = sorted(set(tokens_in_span), key=lambda t: t.i)
    return " ".join(t.text for t in tokens_in_span)

def _get_full_noun(token, exclude_indices=None, follow_prep_of=True) -> Optional[str]:
    """Get full noun phrase including compound modifiers and adjectives."""
    if not token:
        return None
    if exclude_indices is None:
        exclude_indices = set()
    
    tokens = [token]
    
    def recurse_modifiers(node):
        for child in node.children:
            if child.i in exclude_indices:
                continue
            # Accept compound modifiers: NOUN, PROPN, and VERB (gerunds like 'learning', 'teaching', 'visiting')
            if child.dep_ == "compound" and child.pos_ in ("NOUN", "PROPN", "VERB"):
                # Exclude action verbs misparsed as compound
                if child.text.lower() not in AMOD_EXCLUDE:
                    tokens.append(child)
                    recurse_modifiers(child)
            elif child.dep_ == "amod":
                if child.text.lower() not in AMOD_EXCLUDE and child.text.lower() not in ("that", "which", "more"):
                    tokens.append(child)
                    recurse_modifiers(child)
            elif child.dep_ == "punct" and child.text == "-":
                tokens.append(child)
                recurse_modifiers(child)
                
    recurse_modifiers(token)
    
    RESOURCE_FOLLOW_OF_WORDS = {
        "status", "content", "contents", "type", "value", "list",
        "meta-data", "metadata", "meta-information", "data", "information",
        "access"
    }
    
    # Prepositional Phrase Stripping: For resources, we typically don't include 'of X'
    # unless the noun is a metadata word like 'content' or 'status'.
    follow_prep = False
    if token.text.lower() in RESOURCE_FOLLOW_OF_WORDS:
        follow_prep = True
        
    if follow_prep:
        for child in token.children:
            if child.dep_ == "prep" and child.text.lower() in ("of", "to"):
                for gc in child.children:
                    if gc.dep_ == "pobj":
                        # Check if pobj subtree contains person/scope words
                        pobj_words = {t.text.lower() for t in gc.subtree}
                        if pobj_words & OF_SCOPE_PERSON_WORDS:
                            # This is a scope qualifier ("of civilian students"), skip
                            continue
                        
                        pobj_phrase_tokens = []
                        
                        def recurse_pobj(n):
                            if n.i not in exclude_indices:
                                pobj_phrase_tokens.append(n)
                            for c in n.children:
                                if c.dep_ in ("compound", "amod", "punct"):
                                    recurse_pobj(c)
                                    
                        recurse_pobj(gc)
                        if pobj_phrase_tokens:
                            tokens.append(child) # 'of' or 'to'
                            tokens.extend(pobj_phrase_tokens)
                            
    # Special handling for "right to X"
    if token.text.lower() == "right":
        for child in token.children:
            if child.dep_ in ("acl", "relcl"):
                if any(c.text.lower() == "to" for c in child.children):
                    tokens.append(child)
                    for c in child.children:
                        if c.dep_ in ("aux", "dobj", "compound", "prt"):
                            if c.dep_ == "dobj":
                                def get_dobj_subtree(n):
                                    if n.i not in exclude_indices:
                                        if n.dep_ == "amod" and n.text.lower() in AMOD_EXCLUDE:
                                            pass
                                        else:
                                            tokens.append(n)
                                    for gc in n.children:
                                        if gc.dep_ in ("compound", "amod", "det", "punct"):
                                            get_dobj_subtree(gc)
                                get_dobj_subtree(c)
                            else:
                                tokens.append(c)
    
    unique_tokens = []
    seen = set()
    for t in sorted(tokens, key=lambda x: x.i):
        if t.i not in seen:
            seen.add(t.i)
            unique_tokens.append(t)
            
    text = "".join([t.text + t.whitespace_ for t in unique_tokens]).strip()
    return text


def _is_in_advcl(token):
    """Check if token is inside an adverbial clause that starts with a mark (if/unless)."""
    curr = token
    while curr != curr.head:
        if curr.dep_ == "advcl":
            # Only consider it a conditional if it has a 'mark' child like 'if' or 'unless'
            if any(c.dep_ == "mark" for c in curr.children):
                return True
        curr = curr.head
    return False

def _get_anchor_tokens(doc):
    """Find the main resource anchor tokens (nsubjpass/nsubj > dobj > attr > pobj of verb > pobj),
    ignoring tokens inside adverbial clauses (e.g. conditional if/unless).
    """
    valid_tokens = [t for t in doc if not _is_in_advcl(t)]
    anchors = []
    
    for token in valid_tokens:
        if token.dep_ == "nsubjpass" and token.head.dep_ not in ("relcl", "acl"):
            anchors.append(token)
            
    if not anchors:
        for token in valid_tokens:
            if token.dep_ == "dobj" and token.head.dep_ not in ("relcl", "acl"):
                if token.text.lower() == "all":
                    continue
                anchors.append(token)
        for token in valid_tokens:
            if token.dep_ == "conj" and token.pos_ in ("NOUN", "PROPN") and token.head.pos_ == "VERB" and token.head.dep_ not in ("relcl", "acl"):
                anchors.append(token)
            
    if anchors:
        seen = set()
        unique = []
        for a in anchors:
            if a.i not in seen:
                seen.add(a.i)
                unique.append(a)
        return unique
        
    for token in valid_tokens:
        if token.dep_ == "nsubjpass":
            return [token]
    for token in valid_tokens:
        if token.dep_ == "attr":
            return [token]
    for token in valid_tokens:
        if token.dep_ == "pobj" and token.head.head.pos_ == "VERB":
            return [token]
    for token in valid_tokens:
        if token.dep_ == "pobj":
            if token.text.lower() == "system":
                continue
            return [token]
    for token in valid_tokens:
        if token.dep_ == "pobj":
            return [token]
            
    return []

def _extract_resource_tokens(doc) -> List:
    """Extract resource tokens using an anchor and its conjuncts."""
    anchors = _get_anchor_tokens(doc)
    if not anchors:
        return []
    
    # If anchor is a generic quantifier/container like "list of [X]", elevate [X] to resource!
    adjusted_anchors = []
    for tok in anchors:
        if tok.lemma_.lower() in ("list", "set", "batch") and tok.pos_ in ("NOUN", "PROPN"):
            of_prep = next((c for c in tok.children if c.dep_ == "prep" and c.text.lower() == "of"), None)
            if of_prep:
                pobj = next((c for c in of_prep.children if c.dep_ == "pobj"), None)
                if pobj and not _is_person_type(pobj, PERSON_NOUNS) and pobj.lemma_.lower() not in ("student", "civilian_student", "user", "patient"):
                    adjusted_anchors.append(pobj)
                    continue
        adjusted_anchors.append(tok)

    resource_tokens = []
    for anchor in adjusted_anchors:
        if anchor not in resource_tokens:
            resource_tokens.append(anchor)
        for conj in anchor.conjuncts:
            # Only add noun/propn conjuncts
            if conj.pos_ in ("NOUN", "PROPN", "PRON"):
                if conj not in resource_tokens:
                    resource_tokens.append(conj)
                
        # Fallback for spaCy parsing error where noun conjuncts are attached to the verb instead of the dobj
        if anchor.head and anchor.head.pos_ == "VERB":
            for child in anchor.head.children:
                if child.dep_ == "conj" and child.pos_ in ("NOUN", "PROPN", "PRON") and child.i > anchor.i:
                    if child not in resource_tokens:
                        resource_tokens.append(child)
                        for c in child.conjuncts:
                            if c.pos_ in ("NOUN", "PROPN", "PRON") and c not in resource_tokens:
                                resource_tokens.append(c)
                            
    return resource_tokens


def _identify_resource_type(resource_phrase: str) -> ResourceEntityType:
    """Identify resource entity type from phrase."""
    phrase_lower = resource_phrase.lower()
    
    # Exact matches first
    if phrase_lower in RESOURCE_TYPE_MAPPING:
        return RESOURCE_TYPE_MAPPING[phrase_lower]
    
    # Partial matches
    for key, entity_type in RESOURCE_TYPE_MAPPING.items():
        if key in phrase_lower:
            return entity_type
    
    # Default
    return ResourceEntityType.DOCUMENT


def _extract_resource_label(resource_phrase: str) -> str:
    """Extract normalized resource label exactly, without substring matching."""
    phrase_lower = resource_phrase.lower().strip()
    phrase_lower = phrase_lower.replace("'", "").replace("'", "")
    
    if phrase_lower in RESOURCE_LABEL_MAPPING:
        label = RESOURCE_LABEL_MAPPING[phrase_lower]
    else:
        label = phrase_lower.replace(" ", "_").replace("-", "-")

    # Fix 3A — Strip possessive prefix
    if label.startswith("own_"):
        label = label[4:]

    # Fix 3C — Truncate "of [person_noun]" appendages
    PERSON_SUFFIXES = [
        "of civilian students", "of_civilian_students",
        "of all civilian students", "of_all_civilian_students",
        "of registered patients", "of_registered_patients",
        "of assigned patients", "of_assigned_patients",
        "of all patients", "of_all_patients"
    ]
    for suffix in PERSON_SUFFIXES:
        if label.endswith(suffix):
            label = label[:-len(suffix)].strip("_ ")

    return label


def _extract_qualifier(sentence: str, resource_phrase: str) -> Optional[str]:
    """Extract resource qualifier (own, every, all, entire)."""
    sentence_lower = sentence.lower()
    phrase_pos = sentence_lower.find(resource_phrase.lower())
    
    if phrase_pos == -1:
        return None
    
    # Check window before resource
    window_start = max(0, phrase_pos - 50)
    window = sentence_lower[window_start:phrase_pos]
    
    # Check for qualifiers in order of priority using word boundaries
    if re.search(r'\bentire\b', window):
        return "entire"
    elif re.search(r'\bevery\b', window):
        return "every"
    elif re.search(r'\ball\b', window):
        return "all"
    elif re.search(r'\b(their own|my own|own)\b', window):
        return "own"
    elif re.search(r'\b(their)\b', window):
        return "their"
    elif re.search(r'\bthe\b', window):
        return "the"
        
    # Implicit 'own' rules based on context (catching what human annotators inferred)
    if re.search(r'which they (have taken|are taking)|for that (department|course)', sentence_lower):
        return "own"
    if "department’s courses" in sentence_lower or "department's courses" in sentence_lower:
        return "own"
        
    return None


def _extract_scope(sentence: str) -> Tuple[Optional[str], Optional[str]]:
    """
    Extract resource scope (level and ref).
    
    Returns:
        Tuple of (scope_level, scope_ref)
    """
    sentence_lower = sentence.lower()
    
    # Check scope indicators
    for indicator, level in SCOPE_LEVEL_INDICATORS.items():
        if indicator in sentence_lower:
            return level, indicator.replace("for ", "").replace("in ", "")
    
    # Check for "per year" indicator
    if "per year" in sentence_lower:
        return "all", "per_year"
    
    return None, None


def _extract_subject_filter(sentence: str) -> Optional[str]:
    """Extract subject filter (type of subject owning resource)."""
    sentence_lower = sentence.lower()
    
    for keyword, filter_name in SUBJECT_FILTER_KEYWORDS.items():
        if keyword in sentence_lower:
            return filter_name
    
    return None


def _extract_attributes(sentence: str) -> Dict:
    """Extract resource attributes (sensitivity, status)."""
    sentence_lower = sentence.lower()
    attributes = {}
    
    # Check sensitivity
    for keyword, sens_level in SENSITIVITY_KEYWORDS.items():
        if keyword in sentence_lower:
            attributes["sensitivity"] = sens_level
            break
    
    # Check status
    for keyword, status in STATUS_KEYWORDS.items():
        if keyword in sentence_lower:
            attributes["status"] = status
            break
    
    return attributes if attributes else None


def _is_descendant(child_token, ancestor_token):
    """Check if child_token is in the subtree of ancestor_token, ignoring conj links."""
    curr = child_token
    while curr != curr.head:
        if curr.dep_ == "conj":
            return False  # Conjuncts are separate entities, not descendants of the same resource
        if curr.head == ancestor_token:
            return True
        curr = curr.head
    return False

def extract_resources(sentence: str, doc=None) -> List[Resource]:
    """
    Extract multiple resources from a policy sentence.
    
    Args:
        sentence: Natural language policy sentence
        doc: spaCy Doc object (optional)
    
    Returns:
        List of Resource objects
    """
    if doc is None:
        doc = get_spacy_model()(sentence)
    
    # Extract environment attributes to filter out their tokens
    env_attrs = extract_env_attributes(sentence)
    env_tokens_lower = set()
    for e in env_attrs:
        val = e.get("full_value") or e.get("phrase") or e.get("trigger_phrase", "")
        for t in str(val).lower().split():
            env_tokens_lower.add(t)

    raw_resource_tokens = _extract_resource_tokens(doc)
    
    # Khử trùng lặp token theo subtree
    deduped_tokens = []
    for token in raw_resource_tokens:
        is_sub = False
        for other in raw_resource_tokens:
            if token != other and _is_descendant(token, other):
                is_sub = True
                break
        if not is_sub:
            deduped_tokens.append(token)
            
    # Loại bỏ token thuộc environment phrase
    filtered_tokens = []
    for rt in deduped_tokens:
        if rt.text.lower() not in env_tokens_lower:
            filtered_tokens.append(rt)
            
    # If environment swallowed ALL candidates, revert to original tokens
    if not filtered_tokens and deduped_tokens:
        filtered_tokens = deduped_tokens
    
    resources = []
    for token in filtered_tokens:
        resource_phrase = _get_full_object_span(token, doc)
        if not resource_phrase:
            resource_phrase = token.text
        
        # Identify type and label
        entity_type = _identify_resource_type(resource_phrase)
        label = _extract_resource_label(resource_phrase)
        
        # Extract qualifier
        qualifier = _extract_qualifier(sentence, resource_phrase)
        
        # Extract scope
        scope_level, scope_ref = _extract_scope(sentence)
        scope = ResourceScope(level=scope_level, ref=scope_ref) if scope_level or scope_ref else None
        
        # Extract subject filter
        subject_filter = _extract_subject_filter(sentence)
        
        # Extract attributes
        attributes_dict = _extract_attributes(sentence)
        attributes = ResourceAttributes(**attributes_dict) if attributes_dict else None
        
        # Build namespace
        namespace = f"resource.{entity_type.value}.{label}"
        
        # Ref tokens
        ref_tokens = [t.text for t in [token] + list(token.conjuncts)]
        
        resource = Resource(
            entity_type=entity_type,
            ref_tokens=ref_tokens,
            label=label,
            qualifier=qualifier,
            scope=scope,
            subject_filter=subject_filter,
            attributes=attributes,
            namespace=namespace
        )
        resources.append(resource)
        
    return resources

def extract_resource(sentence: str, doc=None) -> Optional[Resource]:
    """Deprecated: dùng extract_resources() để lấy đầy đủ. Giữ lại để tương thích ngược — trả resource đầu tiên."""
    resources = extract_resources(sentence, doc)
    return resources[0] if resources else None
