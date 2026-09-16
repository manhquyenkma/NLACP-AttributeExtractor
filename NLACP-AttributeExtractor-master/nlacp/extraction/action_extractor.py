from nlacp.utils.nlp_utils import get_spacy_model
"""
Action Extractor for ABAC Policy v2.0

This module extracts and normalizes actions/operations from policy sentences.
Maps natural language verbs to standard ABAC operations (CRUD+).
"""

import spacy
from typing import List, Tuple, Optional, Dict
from nlacp.validation.schema_validator import Action, Operation

nlp = None  # Lazy load

def get_spacy_model():
    global nlp
    if nlp is None:
        try:
            nlp = spacy.load("en_core_web_sm")
        except OSError:
            nlp = spacy.blank("en")
    return nlp

# Verb to operation mapping
VERB_TO_OPERATION = {
    # Read operations
    "read": "read",
    "view": "read",
    
    # Create operations
    "create": "create",
    "add": "create",
    
    # Update operations
    "update": "update",
    "edit": "update",
    
    # Delete operations
    "delete": "delete",
    "remove": "delete",
}

from nlacp.extraction.env_extractor import extract_env_attributes

ACTION_BLACKLIST = {
    "contain", "consist", "remain",
    "become", "seem", "appear", "include", "involve",
    "require", "need", "mean", "represent", "indicate",
    "record", "consider"
}

# Negation indicators
NEGATION_WORDS = {"not", "cannot", "no", "never", "neither"}


def _extract_action_tokens(doc, sentence="") -> List:
    action_tokens = []

    env_attrs = extract_env_attributes(sentence) if sentence else []
    env_tokens_lower = set()
    for env in env_attrs:
        fv = env.get("full_value", "").lower()
        for w in fv.split():
            env_tokens_lower.add(w)

    SPACY_KNOWN_MISTAG_EXCEPTIONS = {
        "learn": {
            "reason": "gerund-as-verb mistagging trong compound noun 'learning + [materials|management]'",
            "condition": lambda t: any(c.lemma_.lower() in {"material", "management", "system"} for c in t.children)
        }
    }

    root = next((t for t in doc if t.dep_ == "ROOT"), None)
    if root is None:
        return []

    frontier = [root]

    # (1) ROOT
    if root.pos_ == "VERB" or (root.pos_ in ("NOUN", "AUX") and root.lemma_.lower() in VERB_TO_OPERATION):
        exclude = False
        if root.lemma_.lower() in ACTION_BLACKLIST:
            has_dobj = any(c.dep_ == "dobj" for c in root.children)
            if not has_dobj:
                exclude = True
        
        if root.text.lower() in env_tokens_lower:
            exclude = True

        if root.lemma_.lower() in SPACY_KNOWN_MISTAG_EXCEPTIONS:
            if SPACY_KNOWN_MISTAG_EXCEPTIONS[root.lemma_.lower()]["condition"](root):
                exclude = True
                
        if not exclude:
            action_tokens.append(root)
    else:
        return []

    frontier = [root]
    seen = {root}
    while frontier:
        current = frontier.pop()
        for child in current.children:
            if child.dep_ == "conj":
                if child.pos_ == "VERB" or (child.pos_ == "NOUN" and child.lemma_.lower() in VERB_TO_OPERATION):
                    exclude = False
                    if child.lemma_.lower() in ACTION_BLACKLIST:
                        has_dobj = any(c.dep_ == "dobj" for c in child.children)
                        if not has_dobj:
                            exclude = True
                    
                    if child.text.lower() in env_tokens_lower:
                        exclude = True

                    if child.lemma_.lower() in SPACY_KNOWN_MISTAG_EXCEPTIONS:
                        if SPACY_KNOWN_MISTAG_EXCEPTIONS[child.lemma_.lower()]["condition"](child):
                            exclude = True
                            
                    if not exclude:
                        if child not in seen:
                            action_tokens.append(child)
                            seen.add(child)
                            frontier.append(child)

    return action_tokens


def _get_negation_before(token, window_size=5) -> bool:
    start_idx = max(0, token.i - window_size)
    for i in range(start_idx, token.i):
        if token.doc[i].text.lower() in NEGATION_WORDS:
            return True
    return False


def extract_actions(sentence: str, doc=None) -> List[Action]:
    if doc is None:
        doc = get_spacy_model()(sentence)
    
    action_tokens = _extract_action_tokens(doc, sentence)
    
    actions = []
    for token in action_tokens:
        verb_lemma = token.lemma_.lower() if token.lemma_ else token.text.lower()
        
        # Apply CRUD synonym map
        mapped_verb = VERB_TO_OPERATION.get(verb_lemma)
        if not mapped_verb:
            mapped_verb = VERB_TO_OPERATION.get(token.text.lower())
            
        operation_verb = mapped_verb if mapped_verb else verb_lemma
        
        if verb_lemma == "have":
            dobj = next((c for c in token.children if c.dep_ == "dobj"), None)
            if dobj:
                dobj_lemma = dobj.lemma_.lower()
                # If object is person/role noun -> CREATE
                if dobj_lemma in ["instructor", "student", "user", "lecturer", "chair", "coordinator", "officer", "person", "representative"]:
                    operation_verb = "create"
                else:
                    # If object is thing -> READ
                    operation_verb = "read"
            else:
                continue # Default: skip "have"
        negated = _get_negation_before(token)
        
        action = Action(
            verb=verb_lemma,             # Keep the raw original verb
            operation=operation_verb.upper(), # Store the CRUD operation
            negated=negated
        )
        
        actions.append(action)
    
    seen = set()
    unique_actions = []
    for action in actions:
        key = (action.verb, action.operation, action.negated)
        if key not in seen:
            seen.add(key)
            unique_actions.append(action)
    
    return unique_actions


def extract_actions_with_logical_op(sentence: str, doc=None) -> Tuple[List[Action], Optional[str]]:
    """
    Extract actions with logical operator (AND/OR) if multiple.
    
    Returns:
        Tuple of (actions_list, logical_operator)
    """
    actions = extract_actions(sentence, doc)
    
    if len(actions) <= 1:
        return actions, None
    
    # Detect logical operator from sentence
    sentence_lower = sentence.lower()
    logical_op = None
    
    if " and " in sentence_lower:
        logical_op = "AND"
    elif " or " in sentence_lower:
        logical_op = "OR"
    
    return actions, logical_op


def suggest_operation_from_context(sentence: str, doc=None) -> Optional[Operation]:
    """
    If action extraction failed, try to infer operation from context/keywords.
    
    Useful for cases like: "An official visiting lecturer agreement must have a contract number"
    where implicit operations might be CREATE/UPDATE.
    """
    if doc is None:
        doc = nlp(sentence)
    
    sentence_lower = sentence.lower()
    
    # Context-based inference
    if "must have" in sentence_lower:
        return Operation.CREATE
    elif "can only" in sentence_lower or "can only be" in sentence_lower:
        return Operation.READ
    
    return None
