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
    # Read operations (CRUD: Read)
    "read": "read",
    "view": "read",
    "access": "read",
    "check": "read",
    "get": "read",
    "retrieve": "read",
    "see": "read",
    "query": "read",
    "list": "read",
    "show": "read",
    "display": "read",
    "monitor": "read",
    "review": "read",
    "inspect": "read",
    "examine": "read",
    "audit": "read",
    "track": "read",
    "watch": "read",
    "find": "read",
    "fetch": "read",
    "download": "read",
    "export": "read",
    "print": "read",
    "verify": "read",
    "browse": "read",
    "look": "read",
    "choose": "read",
    "search": "read",
    
    # Create operations (CRUD: Create)
    "create": "create",
    "submit": "create",
    "upload": "create",
    "publish": "create",
    "post": "create",
    "register": "create",
    "generate": "create",
    "issue": "create",
    "add": "create",
    "insert": "create",
    "install": "create",
    "deploy": "create",
    "request": "create",
    "log": "create",
    "store": "create",
    
    # Update operations (CRUD: Update)
    "change": "update",
    "update": "update",
    "modify": "update",
    "edit": "update",
    "configure": "update",
    "approve": "update",
    "reject": "update",
    "deny": "update",
    "revoke": "update",
    "grant": "update",
    "enable": "update",
    "disable": "update",
    "activate": "update",
    "deactivate": "update",
    "cancel": "update",
    "close": "update",
    "lock": "update",
    "unlock": "update",
    "transfer": "update",
    "move": "update",
    "delegate": "update",
    "forward": "update",
    "notify": "update",
    "bill": "update",
    "charge": "update",
    "process": "update",
    "handle": "update",
    "manage": "update",
    "maintain": "update",
    "reset": "update",
    "complete": "update",
    "allow": "update",
    
    # Delete operations (CRUD: Delete)
    "remove": "delete",
    "delete": "delete",
    "drop": "delete",
    "erase": "delete",
    "clear": "delete",
    "archive": "delete",
    "purge": "delete",
    "unregister": "delete",
    "deregister": "delete",
    
    # Extended & Domain-Specific Operations (CRUD+ for domain datasets like iTrust)
    "schedule": "schedule",
    "assign": "assign",
    "send": "send",
    "report": "report",
    "record": "record",
    "set": "set",
    "filter": "filter",
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
    env_spans = []
    if sentence:
        s_lower = sentence.lower()
        for env in env_attrs:
            fv = (env.get("full_value", "") or env.get("trigger_phrase", "") or "").lower().strip()
            if fv:
                start = 0
                while True:
                    idx = s_lower.find(fv, start)
                    if idx == -1:
                        break
                    env_spans.append((idx, idx + len(fv)))
                    start = idx + len(fv)

    def is_in_env_span(tok):
        for start_c, end_c in env_spans:
            if start_c <= tok.idx < end_c:
                return True
        return False

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
        
        if is_in_env_span(root):
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
                # Always add conj to frontier for deeper chain traversal
                # (e.g., create→delete→manage)
                if child not in seen:
                    seen.add(child)
                    frontier.append(child)
                
                # Accept VERB, or NOUN/ADJ whose lemma is a known action verb
                # (spaCy sometimes mistags coordinate verbs as ADJ/NOUN)
                is_action = (child.pos_ == "VERB" or 
                            (child.pos_ in ("NOUN", "ADJ") and child.lemma_.lower() in VERB_TO_OPERATION))
                if is_action:
                    exclude = False
                    if child.lemma_.lower() in ACTION_BLACKLIST:
                        has_dobj = any(c.dep_ == "dobj" for c in child.children)
                        if not has_dobj:
                            exclude = True
                    
                    if is_in_env_span(child):
                        exclude = True

                    if child.lemma_.lower() in SPACY_KNOWN_MISTAG_EXCEPTIONS:
                        if SPACY_KNOWN_MISTAG_EXCEPTIONS[child.lemma_.lower()]["condition"](child):
                            exclude = True
                            
                    if not exclude:
                        action_tokens.append(child)
                else:
                    # Also check if child has a compound child that is an action verb (e.g. "view or search invoices")
                    for sub in child.children:
                        if sub.dep_ in ("compound", "dep") and sub.lemma_.lower() in VERB_TO_OPERATION:
                            if not is_in_env_span(sub):
                                action_tokens.append(sub)

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
