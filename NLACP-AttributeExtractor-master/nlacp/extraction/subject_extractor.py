"""
Subject Extractor for ABAC Policy v2.0

This module extracts and normalizes subject entities from policy sentences.
Subjects can be: user, role, group, system, agreement, schedule.
"""

import spacy
import re
from typing import List, Dict, Optional, Tuple
from nlacp.validation.schema_validator import Subject, SubjectQualifier, EntityType

nlp = None  # Lazy load

def _get_nlp():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.blank("en")
    return nlp

# Subject dependency patterns
SUBJECT_DEPS = {"nsubj", "nsubjpass", "csubj"}

# Subject role keywords
SUBJECT_ROLE_KEYWORDS = {
    # Education domain (VACT)
    "student": "student",
    "lecturer": "lecturer",
    "instructor": "instructor",
    "professor": "professor",
    "academic coordinator": "academic_coordinator",
    "coordinator": "academic_coordinator",
    "department chair": "department_chair",
    "chair": "department_chair",
    "head": "head",
    "office user": "office_user",
    "user": "user",
    "applicant": "applicant",
    "visiting lecturer": "visiting_lecturer",
    "teacher": "lecturer",
    
    # Medical domain (iTrust)
    "hcp": "hcp",
    "lhcp": "lhcp",
    "patient": "patient",
    "physician": "physician",
    "nurse": "nurse",
    "doctor": "doctor",
    "administrator": "administrator",
    "uap": "uap",
    "representative": "representative",
    "agent": "agent",
    "technician": "technician",
    "pharmacist": "pharmacist",
    "specialist": "specialist",
    "staff": "staff",
    
    # Financial / Corporate / Business (eDocument)
    "member": "member",
    "employee": "employee",
    "sender": "sender",
    "receiver": "receiver",
    "tenant": "tenant",
    "provider": "provider",
    "reseller": "reseller",
    "admin": "administrator",
    "helpdesk": "helpdesk",
    "supervisor": "supervisor",
    "supervisee": "supervisee",
    "manager": "manager",
    "director": "director",
    "officer": "officer",
    "auditor": "auditor",
    "accountant": "accountant",
    "secretary": "secretary",
    "customer": "customer",
    "department": "department",
    "office": "office",
    "team": "team",
    "personnel": "personnel",
    "clerk": "clerk",
    "analyst": "analyst",
    
    # System management
    "superuser": "superuser",
    "operator": "operator",
    "service": "service",
    "system": "system",
    "application": "application",
    "automated_service": "automated_service",
    "print_house": "print_house",
}

# Subject type qualifiers
SUBJECT_TYPE_QUALIFIERS = {
    "civilian": "civilian",
    "visiting": "visiting",
    "retired": "retired",
    "official": "official",
}

# Subject rank qualifiers
SUBJECT_RANK_QUALIFIERS = {
    "senior": "senior",
    "head": "head",
    "chair": "chair",
    "coordinator": "coordinator",
}

# Subject status qualifiers
SUBJECT_STATUS_QUALIFIERS = {
    "retired": "retired",
    "active": "active",
    "not_yet_retired": "not_yet_retired",
}

# Office/Department qualifiers
OFFICE_QUALIFIERS = {
    "academic_affairs": "academic_affairs",
    "examination": "examination",
    "finance": "finance",
    "educational_testing": "educational_testing",
    "board_of_directors": "board_of_directors",
}

# Negation indicators
NEGATION_TRIGGERS = {"not", "who are not", "who are no", "no longer"}

SECURITY_QUALIFIERS = {
    'authenticated', 'unauthenticated', 'registered',
    'unregistered', 'authorized', 'unauthorized',
    'verified', 'unverified', 'active', 'inactive',
    'senior', 'junior', 'external', 'internal',
    'private', 'public', 'certified', 'licensed',
    'assigned', 'responsible', 'approved', 'health', 'patient', 'project'
}

COMPOUND_DOMAINS = {
    'bank', 'office', 'audit', 'accounting', 'large_bank', 'large_bank_audit',
    'ict', 'human_resource', 'customer'
}

def _clean_word(w):
    w = w.lower().strip()
    if len(w) > 3 and w.endswith("ies"):
        return w[:-3] + "y"
    if len(w) > 3 and w.endswith("s") and not w.endswith("ss"):
        return w[:-1]
    return w


def _get_conjuncts(token) -> List:
    """Get conjuncts (and/or) of a token."""
    conjs = []
    for child in token.children:
        if child.dep_ == "conj" and child.pos_ in ("NOUN", "PROPN"):
            conjs.append(child)
            conjs.extend(_get_conjuncts(child))
    return conjs


def _get_full_subject_span(nsubj_token, doc=None) -> Optional[str]:
    """Get full noun phrase including amod, compound, and prepositional phrases."""
    if not nsubj_token:
        return None
        
    span_tokens = [nsubj_token]
    
    def collect_children(token):
        for child in token.children:
            if child.dep_ in {"amod", "compound", "nummod"}:
                span_tokens.append(child)
                collect_children(child)
            elif child.dep_ == "prep":
                if child.text.lower() in {"in", "on", "at", "during", "before", "after", "from", "to"}:
                    continue
                span_tokens.append(child)
                for pobj in child.children:
                    if pobj.dep_ in {"pobj", "pcomp"}:
                        span_tokens.append(pobj)
                        collect_children(pobj)

    collect_children(nsubj_token)
    span_tokens.sort(key=lambda x: x.i)
    text = " ".join([t.text for t in span_tokens])
    return text.strip()


def _extract_subject_tokens(doc) -> List:
    """Extract subject tokens from dependency tree."""
    subject_tokens = []
    root = None
    for t in doc:
        if t.dep_ == "ROOT":
            root = t
            break

    for token in doc:
        # Case 1: nsubj directly attached to ROOT
        if token.dep_ in SUBJECT_DEPS and root is not None and token.head == root:
            subject_tokens.append(token)
            subject_tokens.extend(_get_conjuncts(token))
        # Case 2: nsubj attached to an aux of ROOT (spaCy quirk: "can view received" -> ROOT=received, view=aux)
        elif token.dep_ in SUBJECT_DEPS and root is not None and token.head.dep_ in {"aux", "auxpass"} and token.head.head == root:
            subject_tokens.append(token)
            subject_tokens.extend(_get_conjuncts(token))

    # We determine if a token is a likely actor by checking our known roles
    def is_known_actor(token):
        return any(kw in token.lemma_.lower() for kw in SUBJECT_ROLE_KEYWORDS.keys())

    # Case 3 & 4 (NEW): Passive "by" agent and Possessive subjects
    # If the grammatical nsubj is NOT a known actor, but there is an agent or possessor
    # that IS a known actor, we assume the agent/possessor is the true ABAC subject.
    if subject_tokens and not any(is_known_actor(t) for t in subject_tokens):
        new_subjects = []
        for st in subject_tokens:
            # Check for possessor (Case 4: "Students' schedules")
            poss = next((c for c in st.children if c.dep_ == "poss"), None)
            if poss and is_known_actor(poss):
                new_subjects.append(poss)
                continue
            
            # Check for "by" agent modifying the nsubj directly or via participle acl (e.g., in "action done by User")
            agent_found = False
            for child in st.children:
                agent_node = child if child.dep_ == "agent" else next((gc for gc in child.children if gc.dep_ == "agent"), None)
                if agent_node:
                    pobj = next((c for c in agent_node.children if c.dep_ == "pobj"), None)
                    if pobj and is_known_actor(pobj):
                        new_subjects.append(pobj)
                        agent_found = True
                        break
            if agent_found:
                continue
                
            # Check for "by" agent modifying the root/passive verb or acl
            for token in doc:
                if token.dep_ == "agent" or (token.dep_ == "prep" and token.text.lower() == "by"):
                    if token.head.dep_ in SUBJECT_DEPS or token.head.dep_ in {"ROOT", "auxpass", "acl"}:
                        pobj = next((c for c in token.children if c.dep_ == "pobj"), None)
                        if pobj and is_known_actor(pobj):
                            new_subjects.append(pobj)
                            agent_found = True
                            break
            if not agent_found:
                # Fallback to the original grammatical subject if no actor agent/possessor is found
                new_subjects.append(st)
        
        # Extend conjuncts for any new subjects we found
        final_subjects = []
        for ns in new_subjects:
            final_subjects.append(ns)
            final_subjects.extend(_get_conjuncts(ns))
        subject_tokens = final_subjects

    return subject_tokens



def _detect_negation(sentence: str, subject_phrase: str) -> bool:
    """Detect if subject is negated (e.g., 'who are NOT')."""
    # Look for negation patterns around subject
    sentence_lower = sentence.lower()
    subject_lower = subject_phrase.lower()
    subject_pos = sentence_lower.find(subject_lower)
    
    if subject_pos == -1:
        return False
    
    # Check window around subject
    window_start = max(0, subject_pos - 50)
    window_end = min(len(sentence_lower), subject_pos + len(subject_lower) + 50)
    window = sentence_lower[window_start:window_end]
    
    for trigger in NEGATION_TRIGGERS:
        if trigger in window:
            return True
    return False


def _identify_role(subject_phrase: str) -> Tuple[Optional[str], Dict]:
    """
    Identify subject role and qualifiers from phrase.
    
    Returns:
        Tuple of (role, qualifiers_dict)
    """
    phrase_lower = subject_phrase.lower()
    
    qualifiers = {}
    role = None
    
    # Phase F3: Closed-class pronouns -> user (generic)
    PRONOUNS = {"he", "she", "they", "one", "someone", "no one"}
    if phrase_lower in PRONOUNS:
        role = "user"
    
    if not role:
        words = [_clean_word(w) for w in re.split(r'[^a-z0-9]+', phrase_lower) if w and w not in {"a", "an", "the", "personal", "public", "valid"}]
        
        # 1. Specific ABAC / VACT patterns
        if "head" in words and "of" in words:
            role = "head"
        elif "visiting" in words and "lecturer" in words and "agreement" in words:
            role = "visiting_lecturer_agreement"
        elif "application" in words and any(w.startswith("admin") for w in words):
            role = "application_admin"
        elif "authenticated" in words and "user" in words:
            role = "authenticated_user"
        elif "unregistered" in words and "sender" in words:
            role = "unregistered_sender"
        elif "patient" in words and "representative" in words:
            role = "patient_representative"
        elif "health" in words and "representative" in words:
            role = "health_representative"
        elif "health" in words and "agent" in words:
            role = "health_agent"
        elif "project" in words and "member" in words:
            role = "project_member"
        elif "head of " in phrase_lower:
            role = "head"
        elif words and words[0] == "member" and "of" in words:
            role = "_".join(words)
        elif words and "of" in words and any("service" in w for w in words):
            role = "_".join(words)
        elif " of " in phrase_lower:
            role = phrase_lower.replace(" ", "_")
        else:
            # Check if phrase ends in a known role and has security/domain modifiers
            KNOWN_ROLES = {"user", "sender", "receiver", "representative", "agent", "manager", "member", "department", "employee", "director", "officer", "auditor", "accountant", "clerk", "supervisor", "analyst", "tenant", "provider", "admin", "administrator"}
            matched_composite = False
            for i in range(len(words)-1, -1, -1):
                if words[i] in KNOWN_ROLES or words[i].startswith("admin"):
                    base = "admin" if words[i].startswith("admin") else words[i]
                    prefix_words = words[:i]
                    if any(pw in SECURITY_QUALIFIERS or pw in COMPOUND_DOMAINS for pw in prefix_words):
                        role = "_".join(prefix_words + [base])
                        matched_composite = True
                        break
            if not matched_composite:
                for keyword in sorted(SUBJECT_ROLE_KEYWORDS.keys(), key=len, reverse=True):
                    if keyword in phrase_lower:
                        role = SUBJECT_ROLE_KEYWORDS[keyword]
                        break
    
    # Extract type qualifiers
    for type_q in SUBJECT_TYPE_QUALIFIERS:
        if type_q in phrase_lower:
            qualifiers["type"] = type_q
    
    # Extract rank qualifiers
    for rank_q in SUBJECT_RANK_QUALIFIERS:
        if rank_q in phrase_lower:
            qualifiers["rank"] = rank_q
    
    # Extract status qualifiers
    for status_q in SUBJECT_STATUS_QUALIFIERS:
        if status_q in phrase_lower:
            qualifiers["status"] = status_q
    
    # Extract office qualifiers
    for office_q in OFFICE_QUALIFIERS:
        if office_q in phrase_lower:
            if "office" not in qualifiers:
                qualifiers["office"] = []
            qualifiers["office"].append(office_q)
    
    # Check for "own department" scope
    if "own" in phrase_lower and ("department" in phrase_lower or "department's" in phrase_lower):
        qualifiers["department"] = "own"
    elif "any" in phrase_lower and "department" in phrase_lower:
        qualifiers["department"] = "any"
    
    if not role:
        role = phrase_lower.replace(" ", "_")
    
    # Flatten office list if needed
    if "office" in qualifiers and len(qualifiers["office"]) == 1:
        qualifiers["office"] = qualifiers["office"][0]
    
    return role, qualifiers


def _determine_entity_type(role: str, qualifiers: Dict) -> EntityType:
    """Determine entity type based on role and qualifiers."""
    if role == "visiting_lecturer_agreement":
        return EntityType.AGREEMENT
    elif "schedule" in role.lower():
        return EntityType.SCHEDULE
    # Most subjects are users
    return EntityType.USER


def extract_subjects(sentence: str, doc=None) -> List[Subject]:
    """
    Extract all subjects from a policy sentence.
    
    Args:
        sentence: Natural language policy sentence
        doc: spaCy Doc object (optional, will parse if not provided)
    
    Returns:
        List of Subject objects
    """
    if doc is None:
        doc = _get_nlp()(sentence)
    
    subject_tokens = _extract_subject_tokens(doc)
    
    if not subject_tokens:
        return []
    
    subjects = []
    for token in subject_tokens:
        # Get full noun phrase
        subject_phrase = _get_full_subject_span(token, doc)
        if not subject_phrase:
            subject_phrase = token.text
        
        # Identify role and qualifiers
        role, qualifiers_dict = _identify_role(subject_phrase)
        
        # Detect negation
        negated = _detect_negation(sentence, subject_phrase)
        if negated:
            qualifiers_dict["negated"] = negated
        
        # Build qualifier object
        qualifiers = SubjectQualifier(**qualifiers_dict) if qualifiers_dict else None
        
        # Determine entity type
        entity_type = _determine_entity_type(role, qualifiers_dict)
        
        # Build namespace
        namespace = f"subject.{entity_type.value}.{role}"
        if qualifiers:
            if qualifiers.type:
                namespace += f".{qualifiers.type}"
        
        # Extract ref tokens (tokens that reference this subject)
        ref_tokens = [t.text for t in [token] + _get_conjuncts(token)]
        
        subject = Subject(
            entity_type=entity_type,
            ref_tokens=ref_tokens,
            role=role,
            qualifiers=qualifiers,
            namespace=namespace
        )
        
        subjects.append(subject)
    
    return subjects


def extract_subjects_with_logical_op(sentence: str, doc=None) -> Tuple[List[Subject], Optional[str]]:
    """
    Extract subjects with logical operator (AND/OR) if multiple.
    
    Returns:
        Tuple of (subjects_list, logical_operator)
    """
    subjects = extract_subjects(sentence, doc)
    
    if len(subjects) <= 1:
        return subjects, None
    
    # Detect logical operator
    sentence_lower = sentence.lower()
    logical_op = None
    
    if " and " in sentence_lower:
        logical_op = "AND"
    elif " or " in sentence_lower:
        logical_op = "OR"
    
    return subjects, logical_op
