#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
build_manual_gold.py  — Build gold annotation JSON from the manual listing.
Saves to:
  dataset/itrust_manual_gold.json
  dataset/vact_manual_gold.json
"""
import sys, io, os, json
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ─── iTrust gold ──────────────────────────────────────────────────────────────
# Format: {line: [values]}  (line is 1-indexed, matches iTrust_gold.txt)

ITRUST_SUBJECTS = {
    1:  ["health care personnel"],
    2:  ["HCP", "patient"],
    3:  ["HCP", "patient"],
    4:  ["HCP", "patient"],
    5:  ["HCP"],
    6:  ["HCP", "patient"],
    7:  ["HCP"],
    8:  ["DLHCP", "patient"],
    9:  ["HCP", "administrator"],
    10: ["LHCP", "administrator"],
    11: ["HCP", "administrator"],
    12: ["HCP", "administrator"],
    13: ["LHCP", "patient"],
    14: ["administrator"],
    15: ["administrator"],
    16: ["patient representative"],
    17: ["LHCP", "patient"],
    18: ["HCP", "patient"],
    19: ["HCP", "patient"],
    20: ["LHCP", "patient"],
    21: ["LHCP"],
    22: ["user"],
    23: ["LHCP", "patient"],
    24: ["LHCP"],
    25: ["user"],
    26: ["HCP"],
    27: ["HCP"],
    28: ["HCP"],
    29: ["HCP"],
    30: ["LHCP"],
    31: ["patient"],
    32: ["administrator"],
    33: ["LHCP", "user"],
    34: ["LHCP", "patient"],
    35: ["LHCP", "patient"],
    36: ["patient"],
    37: ["LHCP"],
    38: ["user"],
    39: ["user"],
    40: ["user"],
    41: ["HCP", "LHCP", "patient"],
    42: ["HCP"],
    43: ["HCP"],
    44: ["LHCP", "UAP"],
    45: ["LHCP", "patient"],
    46: ["LHCP", "patient"],
    47: ["patient"],
    48: ["LHCP"],
    49: ["LHCP", "patient"],
    50: ["UAP"],
    51: ["patient representative"],
    52: ["patient"],
    53: ["patient"],
    54: ["public health agent"],
    55: ["public health agent"],
    56: ["public health agent"],
    57: ["administrator"],
    58: ["administrator"],
    59: ["administrator"],
    60: ["administrator"],
    61: ["user"],
    62: ["patient representative"],
    63: ["HCP", "patient"],
    64: ["HCP"],
    65: ["HCP", "patient"],
    66: ["LHCP", "patient"],
    67: ["LHCP", "patient"],
    68: ["patient"],
    69: ["patient"],
    70: ["patient representative"],
    71: ["LHCP"],
    72: ["user"],
    74: ["user"],
    75: ["user"],
    76: ["user"],
    77: ["patient"],
    78: ["patient"],
    79: ["public health agent"],
    81: ["administrator"],
    83: ["patient"],
    84: ["patient"],
}

ITRUST_OBJECTS = {
    1:  ["personal health information", "historical values", "immunizations", "office visit"],
    2:  ["diagnosis"],
    3:  ["height", "weight"],
    4:  ["height", "weight"],
    5:  ["immunizations", "lab procedure", "medical procedure"],
    6:  ["patient referral", "referral"],
    7:  ["office visit"],
    8:  ["personal representative"],
    9:  ["allowable immunizations"],
    10: ["allowable diagnoses"],
    11: ["allowable drugs"],
    12: ["allowable laboratory procedures"],
    13: ["chronic disease"],
    14: ["hospital entry", "hospital"],
    15: ["hospital ID", "hospital"],
    16: ["prescriptions"],
    17: ["prescriptions"],
    18: ["patient information"],
    19: ["security question", "password"],
    20: ["appointment"],
    21: ["appointment"],
    22: ["appointment"],
    23: ["patient report"],
    24: ["requested reports"],
    25: ["satisfaction survey"],
    26: ["office visit", "lab procedure"],
    27: ["lab procedure"],
    28: ["lab procedure", "lab technician"],
    29: ["lab procedure"],
    30: ["office visit"],
    31: ["diagnosis"],
    32: ["healthcare provider", "hospital"],
    34: ["message"],
    35: ["message"],
    36: ["message"],
    37: ["message"],
    38: ["message"],
    39: ["message"],
    40: ["message"],
    42: ["referral"],
    43: ["referral"],
    44: ["monitoring list"],
    45: ["monitoring list"],
    46: ["monitoring data types"],
    47: ["physiological measurements"],
    48: ["monitoring data types"],
    50: ["physiological measurements"],
    51: ["physiological measurements"],
    52: ["adverse event"],
    53: ["immunizations", "adverse event"],
    54: ["adverse event"],
    56: ["adverse event"],
    58: ["drug interaction"],
    59: ["drug interaction"],
    60: ["session timeout"],
    61: ["security question", "password"],
    62: ["demographic information"],
    63: ["demographic information", "patient MID"],
    64: ["demographic information"],
    65: ["photo"],
    68: ["personal health records"],
    69: ["access log"],
    70: ["prescriptions"],
    71: ["prescriptions"],
    72: ["appointment"],
    73: ["appointment"],
    74: ["message"],
    75: ["message"],
    76: ["message"],
    77: ["adverse event"],
    78: ["immunizations", "adverse event"],
    79: ["adverse event"],
    81: ["session timeout"],
    83: ["access log"],
    84: ["access log"],
}

ITRUST_CONTEXTS = {
    3:  ["over the last calendar years"],
    18: ["according to data format"],
    28: ["if not in testing state"],
    29: ["if status in transit or received"],
    39: ["by sender recipient or timestamp"],
    61: ["after answering security question"],
    67: ["by name specialty or location"],
    69: ["sorted by role and date"],
    70: ["sorted by start date"],
    71: ["sorted by start date"],
    72: ["equal to or later than current datetime"],
    73: ["soonest upcoming first"],
    74: ["by timestamp", "most recent first"],
    75: ["by timestamp", "ascending or descending order"],
    76: ["based on timestamp range"],
    77: ["in the last months"],
    78: ["in the last months"],
    79: ["within a given time period"],
    80: ["period of inactivity"],
    82: ["period of inactivity"],
    83: ["sorted by date", "most recent first"],
    84: ["most recent first"],
}

ITRUST_ACTIONS = {
    1:  ["enter", "edit"],
    2:  ["indicate"],
    3:  ["graph"],
    4:  ["graph", "choose"],
    5:  ["document"],
    6:  ["add"],
    7:  ["modify", "delete", "return"],
    8:  ["add", "delete"],
    9:  ["maintain", "update"],
    10: ["maintain"],
    11: ["maintain", "update"],
    12: ["maintain", "update"],
    14: ["add", "modify"],
    15: ["modify", "delete"],
    16: ["choose", "view"],
    17: ["view"],
    18: ["edit"],
    19: ["enter", "view"],
    20: ["schedule"],
    21: ["view"],
    22: ["view"],
    23: ["view"],
    24: ["view"],
    25: ["view"],
    26: ["create"],
    27: ["view"],
    28: ["reassign"],
    29: ["remove"],
    30: ["view"],
    31: ["view", "select"],
    32: ["assign"],
    33: ["enter"],
    34: ["send"],
    35: ["send"],
    36: ["reply"],
    37: ["reply"],
    38: ["read"],
    39: ["sort"],
    40: ["apply"],
    41: ["refer"],
    42: ["view"],
    43: ["view"],
    44: ["add", "delete"],
    45: ["delete"],
    46: ["configure"],
    47: ["report"],
    48: ["view"],
    49: ["select"],
    50: ["report"],
    51: ["report"],
    52: ["report"],
    53: ["report"],
    54: ["view"],
    55: ["view"],
    56: ["remove"],
    57: ["record"],
    58: ["view"],
    59: ["delete"],
    60: ["set"],
    61: ["change"],
    62: ["enter", "edit"],
    63: ["enter", "edit"],
    64: ["enter", "edit"],
    65: ["upload"],
    66: ["designate", "undesignate"],
    67: ["search"],
    68: ["view"],
    69: ["view"],
    70: ["view"],
    71: ["view"],
    72: ["view"],
    74: ["view"],
    75: ["sort"],
    76: ["filter"],
    77: ["report"],
    78: ["report"],
    79: ["view"],
    81: ["set"],
    83: ["view"],
    84: ["view"],
}

# ─── VACT KMA_ACP gold ────────────────────────────────────────────────────────

VACT_SUBJECTS = {i: [v] for i, v in enumerate([
    "Security officers","Security officers","Training department staff","IT support staff",
    "Managers","Lecturers","Managers","Librarians","IT support staff","Training department staff",
    "Training department staff","Training department staff","Managers","Faculty members",
    "Security officers","VACT students","Training department staff","System administrators",
    "Faculty members","Students","Managers","IT support staff","Security officers",
    "VACT students","Managers","Lecturers","VACT students","VACT students",
    "Security officers","Training department staff","Faculty members","Department heads",
    "IT support staff","Training department staff","Guest users","Managers","Librarians",
    "VACT students","Lecturers","Students","Researchers","Training department staff",
    "Guest users","Lecturers","Faculty members","Faculty members","Training department staff",
    "Managers","Lecturers","Lecturers","IT support staff","VACT students","IT support staff",
    "Training department staff","Lecturers","System administrators","Librarians",
    "Department heads","Managers","Department heads","Managers","Librarians","Lecturers",
    "Guest users","Librarians","Librarians","System administrators","Managers",
    "Training department staff","VACT students","Security officers","Researchers",
    "Researchers","System administrators","Researchers","Department heads","Managers",
    "Security officers","Training department staff","Faculty members","Guest users",
    "Managers","VACT students","VACT students","VACT students","Security officers",
    "IT support staff","Lecturers","Lecturers","Training department staff","Managers",
    "Librarians","Guest users","Guest users","Security officers","System administrators",
    "Department heads","Guest users","Managers","Faculty members",
], start=1)}

VACT_OBJECTS = {i: [v] for i, v in enumerate([
    "exam grades","network configurations","network configurations","cryptography datasets",
    "personnel files","cryptography datasets","personnel files","network configurations",
    "exam grades","cryptography datasets","system logs","financial reports","student records",
    "library resources","research papers","personnel files","network configurations",
    "research papers","network configurations","personnel files","library resources",
    "system logs","exam schedules","exam grades","student records","financial reports",
    "system logs","personnel files","library resources","course materials","exam grades",
    "network configurations","thesis submissions","course materials","exam grades",
    "student records","system logs","exam grades","research papers","cryptography datasets",
    "personnel files","thesis submissions","network configurations","course materials",
    "thesis submissions","network configurations","cryptography datasets","exam grades",
    "student records","library resources","thesis submissions","exam schedules",
    "research papers","exam schedules","cryptography datasets","system logs","exam grades",
    "network configurations","library resources","cryptography datasets","cryptography datasets",
    "personnel files","exam grades","system logs","system logs","network configurations",
    "system logs","cryptography datasets","exam grades","personnel files","thesis submissions",
    "system logs","research papers","thesis submissions","student records","financial reports",
    "research papers","library resources","research papers","financial reports",
    "student records","cryptography datasets","course materials","library resources",
    "research papers","network configurations","student records","research papers",
    "cryptography datasets","cryptography datasets","cryptography datasets","personnel files",
    "cryptography datasets","thesis submissions","course materials","system logs",
    "system logs","personnel files","exam grades","student records",
], start=1)}

VACT_CONTEXTS = {
    1:  ["during the night shift", "via trusted platforms"],
    2:  ["at the headquarters building"],
    3:  ["using internal devices"],
    4:  ["via trusted platforms"],
    5:  ["within the VACT intranet"],
    6:  ["at the cryptography lab"],
    7:  ["on weekends"],
    8:  ["at the cryptography lab"],
    9:  ["from the campus network"],
    10: ["at the cryptography lab"],
    11: ["after the exam period", "at nighttime"],
    12: ["using internal devices"],
    13: ["outside the campus", "from external IP addresses"],
    14: ["via a secure VPN"],
    15: ["during the night shift"],
    16: ["via trusted platforms"],
    17: ["between 8am and 5pm"],
    18: ["at the headquarters building", "after the exam period"],
    19: ["using internal devices"],
    20: ["using internal devices"],
    21: ["at the headquarters building"],
    22: ["at nighttime", "through an encrypted channel"],
    23: ["before the deadline"],
    24: ["via a secure VPN"],
    25: ["via trusted platforms"],
    26: ["using internal devices", "during the registration period"],
    27: ["at the headquarters building"],
    28: ["from the campus network"],
    29: ["through an encrypted channel"],
    30: ["via trusted platforms"],
    31: ["between 8am and 5pm"],
    32: ["via a secure VPN"],
    33: ["via a secure VPN"],
    34: ["at the cryptography lab"],
    35: ["on weekends"],
    36: ["using internal devices"],
    37: ["after the exam period"],
    38: ["using internal devices"],
    39: ["via a secure VPN"],
    40: ["from external IP addresses"],
    41: ["using internal devices", "via a secure VPN"],
    42: ["between 8am and 5pm", "outside the campus"],
    43: ["through an encrypted channel"],
    44: ["during the night shift"],
    45: ["after the exam period"],
    46: ["within the VACT intranet"],
    47: ["from the campus network"],
    48: ["via trusted platforms"],
    49: ["during the registration period"],
    50: ["using authorized workstations"],
    51: ["through an encrypted channel", "during business hours"],
    52: ["during the registration period"],
    53: ["before the deadline"],
    54: ["via a secure VPN", "through an encrypted channel"],
    55: ["between 8am and 5pm", "during business hours"],
    56: ["using authorized workstations"],
    57: ["during the registration period"],
    58: ["during the registration period"],
    59: ["during the night shift"],
    60: ["during the night shift", "during the registration period"],
    61: ["outside the campus"],
    62: ["within the VACT intranet", "at nighttime"],
    63: ["during the night shift", "before the deadline"],
    64: ["during the night shift", "via trusted platforms"],
    65: ["within the VACT intranet", "at the cryptography lab"],
    66: ["during business hours"],
    67: ["at the headquarters building"],
    68: ["through an encrypted channel"],
    69: ["through an encrypted channel", "during business hours"],
    70: ["at the cryptography lab", "before the deadline"],
    71: ["on weekends"],
    72: ["outside the campus"],
    73: ["from external IP addresses"],
    74: ["using internal devices"],
    75: ["via trusted platforms"],
    76: ["before the deadline"],
    77: ["from the campus network"],
    78: ["during the night shift"],
    79: ["during the night shift"],
    80: ["via trusted platforms"],
    81: ["during business hours"],
    82: ["through an encrypted channel"],
    83: ["using authorized workstations"],
    84: ["through an encrypted channel", "using authorized workstations"],
    85: ["via a secure VPN", "using authorized workstations"],
    86: ["during the night shift"],
    87: ["outside the campus"],
    88: ["during the registration period"],
    89: ["using authorized workstations"],
    90: ["via trusted platforms", "during business hours"],
    91: ["at the cryptography lab"],
    92: ["during the night shift", "via a secure VPN"],
    93: ["from external IP addresses"],
    94: ["from external IP addresses"],
    95: ["during the registration period"],
    96: ["before the deadline"],
    97: ["at the headquarters building", "within the VACT intranet"],
    98: ["at the cryptography lab"],
    99: ["during the night shift", "before the deadline"],
    100: ["on weekends"],
}

VACT_ACTIONS = {i: acts for i, acts in enumerate([
    ["view","modify"],["approve","view"],["delete","approve"],["configure","access"],
    ["modify","submit"],["view"],["access"],["view"],["view"],["view","submit"],
    ["review"],["modify","submit"],["access","download"],["approve"],["update"],
    ["download"],["modify"],["review"],["update"],["review"],["review","access"],
    ["approve"],["download","modify"],["configure","upload"],["download"],
    ["approve","update"],["modify"],["approve"],["review"],["delete"],
    ["delete","access"],["view","delete"],["audit"],["view"],["approve","upload"],
    ["view"],["modify"],["view"],["delete","update"],["modify","configure"],
    ["submit","download"],["access","review"],["configure","modify"],["audit"],
    ["delete"],["audit","upload"],["download"],["view"],["configure"],["delete"],
    ["download"],["download","view"],["configure"],["review","access"],["review"],
    ["review"],["configure"],["audit","upload"],["access"],["configure"],["modify"],
    ["review"],["submit"],["upload","delete"],["upload"],["review","download"],
    ["audit"],["upload","review"],["download"],["review"],["upload"],["configure"],
    ["configure"],["audit"],["download"],["update"],["approve"],["audit"],["delete"],
    ["view","update"],["submit"],["view"],["delete"],["audit"],["update","delete"],
    ["download","view"],["download"],["modify"],["submit"],["submit"],["audit","submit"],
    ["configure","submit"],["access"],["modify"],["access"],["access"],["submit"],
    ["access","modify"],["review"],["modify"],
], start=1)}


# ─── Assemble and save ────────────────────────────────────────────────────────

def build_dataset(subjects, objects, contexts, actions):
    """Build per-sentence gold dict from line-indexed dicts."""
    all_lines = sorted(set(
        list(subjects) + list(objects) + list(contexts) + list(actions)
    ))
    out = {}
    for ln in all_lines:
        out[ln] = {
            "subject": subjects.get(ln, []),
            "object":  objects.get(ln,  []),
            "context": contexts.get(ln, []),
            "actions": actions.get(ln,  []),
        }
    return out


if __name__ == "__main__":
    import csv

    itrust_gold = build_dataset(ITRUST_SUBJECTS, ITRUST_OBJECTS, ITRUST_CONTEXTS, ITRUST_ACTIONS)
    vact_gold   = build_dataset(VACT_SUBJECTS,   VACT_OBJECTS,   VACT_CONTEXTS,   VACT_ACTIONS)

    # Load sentences
    itrust_txt = os.path.join(PROJECT_ROOT, "iTrust_gold.txt")
    vact_txt   = os.path.join(PROJECT_ROOT, "VACT_ACP.txt")

    def load_txt(path):
        with open(path, encoding="utf-8") as f:
            return [ln.strip() for ln in f if ln.strip()]

    itrust_sents = load_txt(itrust_txt)
    vact_sents   = load_txt(vact_txt)

    # Merge sentence text into gold
    def attach_sentences(gold_by_line, sentences):
        out = {}
        for ln, data in gold_by_line.items():
            if 1 <= ln <= len(sentences):
                out[sentences[ln - 1]] = data
        return out

    itrust_final = attach_sentences(itrust_gold, itrust_sents)
    vact_final   = attach_sentences(vact_gold,   vact_sents)

    out_itrust = os.path.join(PROJECT_ROOT, "dataset", "itrust_manual_gold.json")
    out_vact   = os.path.join(PROJECT_ROOT, "dataset", "vact_manual_gold.json")

    with open(out_itrust, "w", encoding="utf-8") as f:
        json.dump(itrust_final, f, ensure_ascii=False, indent=2)
    with open(out_vact, "w", encoding="utf-8") as f:
        json.dump(vact_final, f, ensure_ascii=False, indent=2)

    # Summary
    def count_total(gold, key):
        return sum(len(v[key]) for v in gold.values())

    print("iTrust gold:")
    print(f"  Sentences with annotations: {len(itrust_final)}")
    print(f"  Subject  total: {count_total(itrust_final,'subject')}")
    print(f"  Object   total: {count_total(itrust_final,'object')}")
    print(f"  Context  total: {count_total(itrust_final,'context')}")
    print(f"  Actions  total: {count_total(itrust_final,'actions')}")
    print()
    print("VACT KMA_ACP gold:")
    print(f"  Sentences with annotations: {len(vact_final)}")
    print(f"  Subject  total: {count_total(vact_final,'subject')}")
    print(f"  Object   total: {count_total(vact_final,'object')}")
    print(f"  Context  total: {count_total(vact_final,'context')}")
    print(f"  Actions  total: {count_total(vact_final,'actions')}")
    print(f"\n[OK] Saved → {out_itrust}")
    print(f"[OK] Saved → {out_vact}")
