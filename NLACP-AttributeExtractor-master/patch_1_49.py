import json

PATCHES = {
    # Nhóm 1: Nhầm phòng ban (office) thành resource
    4: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["Users"], "role": "user", "qualifiers": {"type": None, "rank": None, "status": None, "office": "educational testing and quality assurance", "department": None, "negated": False}, "namespace": "subject.user.user"}],
        "resource": {"entity_type": "academic", "ref_tokens": ["scores"], "label": "scores", "qualifier": "the", "scope": None, "subject_filter": "civilian_student", "attributes": None, "namespace": "resource.academic.scores"}
    },
    5: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["Users"], "role": "user", "qualifiers": {"type": None, "rank": None, "status": None, "office": "academic affairs", "department": None, "negated": False}, "namespace": "subject.user.user"}],
        "resource": {"entity_type": "academic", "ref_tokens": ["scores"], "label": "scores", "qualifier": "the", "scope": None, "subject_filter": "civilian_student", "attributes": None, "namespace": "resource.academic.scores"}
    },
    10: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["Users"], "role": "user", "qualifiers": {"type": None, "rank": None, "status": None, "office": "educational testing and quality assurance", "department": None, "negated": False}, "namespace": "subject.user.user"}],
        "resource": {"entity_type": "academic", "ref_tokens": ["grades"], "label": "final_grades", "qualifier": "the", "scope": None, "subject_filter": "civilian_student", "attributes": None, "namespace": "resource.academic.final_grades"}
    },
    15: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["Users"], "role": "user", "qualifiers": {"type": None, "rank": None, "status": None, "office": "academic affairs", "department": None, "negated": False}, "namespace": "subject.user.user"}],
        "resource": {"entity_type": "document", "ref_tokens": ["rosters"], "label": "rosters", "qualifier": "all", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.document.rosters"}
    },
    16: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["Users"], "role": "user", "qualifiers": {"type": None, "rank": None, "status": None, "office": "academic affairs", "department": None, "negated": False}, "namespace": "subject.user.user"}],
        "resource": {"entity_type": "document", "ref_tokens": ["rosters"], "label": "course_rosters", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.document.course_rosters"}
    },
    17: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["Users"], "role": "user", "qualifiers": {"type": None, "rank": None, "status": None, "office": "academic affairs", "department": None, "negated": False}, "namespace": "subject.user.user"}],
        "resource": {"entity_type": "document", "ref_tokens": ["rosters"], "label": "course_rosters", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.document.course_rosters"}
    },
    20: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["Users"], "role": "user", "qualifiers": {"type": None, "rank": None, "status": None, "office": "academic affairs", "department": None, "negated": False}, "namespace": "subject.user.user"}],
        "resource": {"entity_type": "document", "ref_tokens": ["transcripts"], "label": "transcripts", "qualifier": "the", "scope": None, "subject_filter": "civilian_student", "attributes": None, "namespace": "resource.document.transcripts"}
    },
    21: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["Users"], "role": "user", "qualifiers": {"type": None, "rank": None, "status": None, "office": "educational testing and quality assurance", "department": None, "negated": False}, "namespace": "subject.user.user"}],
        "resource": {"entity_type": "document", "ref_tokens": ["transcripts"], "label": "transcripts", "qualifier": "the", "scope": None, "subject_filter": "civilian_student", "attributes": None, "namespace": "resource.document.transcripts"}
    },
    22: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["head"], "role": "head", "qualifiers": {"type": None, "rank": "head", "status": None, "office": "academic affairs", "department": None, "negated": False}, "namespace": "subject.user.head"}],
        "resource": {"entity_type": "document", "ref_tokens": ["transcript"], "label": "transcript", "qualifier": "a", "scope": None, "subject_filter": "civilian_student", "attributes": None, "namespace": "resource.document.transcript"}
    },
    25: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["Users"], "role": "user", "qualifiers": {"type": None, "rank": None, "status": None, "office": "academic affairs", "department": None, "negated": False}, "namespace": "subject.user.user"}],
        "resource": {"entity_type": "document", "ref_tokens": ["status"], "label": "status_of_applications", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.document.status_of_applications"}
    },
    35: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["head"], "role": "head", "qualifiers": {"type": None, "rank": "head", "status": None, "office": "academic affairs", "department": None, "negated": False}, "namespace": "subject.user.head"}],
        "resource": {"entity_type": "academic", "ref_tokens": ["exams"], "label": "multiple-choice_exams", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.academic.multiple-choice_exams"}
    },
    36: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["head"], "role": "head", "qualifiers": {"type": None, "rank": "head", "status": None, "office": "academic affairs", "department": None, "negated": False}, "namespace": "subject.user.head"}],
        "resource": {"entity_type": "academic", "ref_tokens": ["exams"], "label": "multiple-choice_exams", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.academic.multiple-choice_exams"}
    },
    37: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["head"], "role": "head", "qualifiers": {"type": None, "rank": "head", "status": None, "office": "educational testing and quality assurance", "department": None, "negated": False}, "namespace": "subject.user.head"}],
        "resource": {"entity_type": "academic", "ref_tokens": ["exams"], "label": "multiple-choice_exams", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.academic.multiple-choice_exams"}
    },
    38: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["head"], "role": "head", "qualifiers": {"type": None, "rank": "head", "status": None, "office": "educational testing and quality assurance", "department": None, "negated": False}, "namespace": "subject.user.head"}],
        "resource": {"entity_type": "academic", "ref_tokens": ["exams"], "label": "multiple-choice_exams", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.academic.multiple-choice_exams"}
    },
    47: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["head"], "role": "head", "qualifiers": {"type": None, "rank": "head", "status": None, "office": "academic affairs", "department": None, "negated": False}, "namespace": "subject.user.head"}],
        "resource": {"entity_type": "document", "ref_tokens": ["materials"], "label": "learning_materials", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.document.learning_materials"}
    },
    48: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["head"], "role": "head", "qualifiers": {"type": None, "rank": "head", "status": None, "office": "academic affairs", "department": None, "negated": False}, "namespace": "subject.user.head"}],
        "resource": {"entity_type": "document", "ref_tokens": ["materials"], "label": "learning_materials", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.document.learning_materials"}
    },
    49: {
        "subjects": [{"entity_type": "user", "ref_tokens": ["head"], "role": "head", "qualifiers": {"type": None, "rank": "head", "status": None, "office": "educational testing and quality assurance", "department": None, "negated": False}, "namespace": "subject.user.head"}],
        "resource": {"entity_type": "document", "ref_tokens": ["materials"], "label": "learning_materials", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.document.learning_materials"}
    },

    # Nhóm 2: Nhầm hệ thống thành resource
    27: {
        "resource": {"entity_type": "document", "ref_tokens": ["applications"], "label": "applications", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.document.applications"}
    },
    30: {
        "resource": {"entity_type": "academic", "ref_tokens": ["exams"], "label": "multiple-choice_exams", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.academic.multiple-choice_exams"}
    },
    41: {
        "resource": {"entity_type": "document", "ref_tokens": ["materials"], "label": "learning_materials", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.document.learning_materials"}
    },

    # Nhóm 3: Câu bị động và sửa nhãn phức hợp
    23: {
        "subjects": [{"entity_type": "system", "ref_tokens": [], "role": "system", "qualifiers": {"type": None, "rank": None, "status": None, "office": None, "department": None, "negated": False}, "namespace": "subject.system"}],
        "resource": {"entity_type": "document", "ref_tokens": ["reason"], "label": "reason_for_a_transcript_revision", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.document.reason"}
    },
    29: {"resource": {"entity_type": "academic", "ref_tokens": ["exams"], "label": "multiple-choice_exams", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.academic.multiple-choice_exams"}},
    31: {"resource": {"entity_type": "academic", "ref_tokens": ["exams"], "label": "multiple-choice_exams", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.academic.multiple-choice_exams"}},
    32: {"resource": {"entity_type": "academic", "ref_tokens": ["exams"], "label": "multiple-choice_exams", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.academic.multiple-choice_exams"}},
    33: {"resource": {"entity_type": "academic", "ref_tokens": ["exams"], "label": "multiple-choice_exams", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.academic.multiple-choice_exams"}},
    34: {"resource": {"entity_type": "academic", "ref_tokens": ["exams"], "label": "multiple-choice_exams", "qualifier": "the", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.academic.multiple-choice_exams"}}
}

path = 'outputs/policies/policy_dataset_gold.json'
with open(path, 'r', encoding='utf-8') as f:
    data = json.load(f)

for p in data['policies']:
    if p['id'] in PATCHES:
        for k, v in PATCHES[p['id']].items():
            p[k] = v

with open(path, 'w', encoding='utf-8') as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"Patched successfully {len(PATCHES)} policies!")
