import json

PATCHES = {
  68: {
    "authorization_decision": "permit",
    "policy_modality": "can",
    "subjects": [{"entity_type": "user", "ref_tokens": ["students"], "role": "student", "qualifiers": {"type": "civilian", "rank": None, "status": None, "office": None, "department": None, "negated": False}, "namespace": "subject.user.student.civilian"}],
    "actions": [{"verb": "allowed", "operation": "READ", "negated": False}],
    "resource": {"entity_type": "schedule", "ref_tokens": ["schedules"], "label": "class_schedule", "qualifier": None, "scope": None, "subject_filter": "civilian_student", "attributes": None, "namespace": "resource.schedule.class_schedule"},
    "environments": [{"id": "env_1", "env_type": "system", "trigger_phrase": "the academic information system", "trigger_word": "in", "systems": [{"label": "the academic information system", "namespace": "environment.device.device_type:academic_information_system"}], "time_range": {"start_time": "07:00", "end_time": "17:25"}}],
    "context": [],
    "relation_pairs": [
      {"entity": "subject", "rel_type": "attribute", "attribute": "type", "value": "civilian"},
      {"entity": "resource", "rel_type": "possession", "attribute": "subject_filter", "value": "civilian_student"}
    ]
  },
  69: {
    "authorization_decision": "permit",
    "policy_modality": "can",
    "subjects": [{"entity_type": "user", "ref_tokens": ["Lecturers"], "role": "lecturer", "qualifiers": {"type": None, "rank": None, "status": None, "office": None, "department": None, "negated": False}, "namespace": "subject.user.lecturer"}],
    "actions": [{"verb": "allowed", "operation": "READ", "negated": False}],
    "resource": {"entity_type": "schedule", "ref_tokens": ["schedules"], "label": "teaching_schedule", "qualifier": None, "scope": None, "subject_filter": "lecturer", "attributes": None, "namespace": "resource.schedule.teaching_schedule"},
    "environments": [{"id": "env_1", "env_type": "system", "trigger_phrase": "the academic information system", "trigger_word": "in", "systems": [{"label": "the academic information system", "namespace": "environment.device.device_type:academic_information_system"}], "time_range": {"start_time": "07:00", "end_time": "17:25"}}],
    "context": [],
    "relation_pairs": [
      {"entity": "resource", "rel_type": "possession", "attribute": "subject_filter", "value": "lecturer"}
    ]
  },
  72: {
    "authorization_decision": "permit",
    "policy_modality": "can",
    "subjects": [{"entity_type": "user", "ref_tokens": ["students"], "role": "student", "qualifiers": {"type": "civilian", "rank": None, "status": None, "office": None, "department": None, "negated": False}, "namespace": "subject.user.student.civilian"}],
    "actions": [{"verb": "retake", "operation": "UPDATE", "negated": False}],
    "resource": {"entity_type": "academic", "ref_tokens": ["course"], "label": "course", "qualifier": "a", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.academic.course"},
    "environments": [{"id": "env_1", "env_type": "system", "trigger_phrase": "the course registration system", "trigger_word": "in", "systems": [{"label": "the course registration system", "namespace": "environment.device.device_type:course_registration_system"}], "time_range": None}],
    "context": [{"id": "cond_1", "type": "relational", "trigger_phrase": "even if the number of registered civilian students has reached the prescribed capacity", "left_operand": "resource.enrolled_students", "operator": ">=", "right_operand": "resource.capacity", "value": None, "value_unit": None, "approver_entity": None, "approval_state": None, "formal_expression": "resource.enrolled_students >= resource.capacity", "negated": False}],
    "relation_pairs": [
      {"entity": "subject", "rel_type": "attribute", "attribute": "type", "value": "civilian"}
    ]
  },
  75: {
    "authorization_decision": "permit",
    "policy_modality": "can",
    "subjects": [{"entity_type": "academic", "ref_tokens": ["project"], "role": "graduation_project", "qualifiers": {"type": None, "rank": None, "status": None, "office": None, "department": None, "negated": False}, "namespace": "subject.academic.graduation_project"}],
    "actions": [{"verb": "have", "operation": "CREATE", "negated": False}],
    "resource": {"entity_type": "user", "ref_tokens": ["instructors"], "label": "instructor", "qualifier": None, "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.user.instructor"},
    "environments": [],
    "context": [{"id": "cond_1", "type": "limit", "trigger_phrase": "at most two", "left_operand": "resource.count", "operator": "<=", "right_operand": None, "value": "2", "value_unit": None, "approver_entity": None, "approval_state": None, "formal_expression": "resource.count <= 2", "negated": False}],
    "relation_pairs": []
  },
  92: {
    "authorization_decision": "permit",
    "policy_modality": "can",
    "subjects": [{"entity_type": "system", "ref_tokens": [], "role": "system", "qualifiers": {"type": None, "rank": None, "status": None, "office": None, "department": None, "negated": False}, "namespace": "subject.system"}],
    "actions": [{"verb": "considered valid", "operation": "UPDATE", "negated": False}],
    "resource": {"entity_type": "agreement", "ref_tokens": ["agreements"], "label": "visiting_lecturer_agreement", "qualifier": None, "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.agreement.visiting_lecturer_agreement"},
    "environments": [],
    "context": [{"id": "cond_1", "type": "approval", "trigger_phrase": "only after approval by the finance office", "left_operand": None, "operator": "APPROVED_BY", "right_operand": None, "value": None, "value_unit": None, "approver_entity": "finance_office", "approval_state": "approved", "formal_expression": "APPROVAL(finance_office) == TRUE", "negated": False}],
    "relation_pairs": []
  },
  95: {
    "authorization_decision": "deny",
    "policy_modality": "cannot",
    "subjects": [{"entity_type": "user", "ref_tokens": ["lecturer"], "role": "visiting_lecturer", "qualifiers": {"type": "visiting", "rank": None, "status": "not_retired", "office": None, "department": None, "negated": False}, "namespace": "subject.user.visiting_lecturer.visiting"}],
    "actions": [{"verb": "teach", "operation": "EXECUTE", "negated": True}],
    "resource": {"entity_type": "academic", "ref_tokens": ["hours"], "label": "teaching_hours", "qualifier": None, "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.academic.teaching_hours"},
    "environments": [],
    "context": [{"id": "cond_1", "type": "limit", "trigger_phrase": "more than 280 hours per year", "left_operand": "subject.teaching_hours_per_year", "operator": ">", "right_operand": None, "value": "280", "value_unit": "hours", "approver_entity": None, "approval_state": None, "formal_expression": "subject.teaching_hours_per_year > 280", "negated": False}],
    "relation_pairs": [
      {"entity": "subject", "rel_type": "attribute", "attribute": "type", "value": "visiting"},
      {"entity": "subject", "rel_type": "attribute", "attribute": "status", "value": "not_retired"}
    ]
  },
  96: {
    "authorization_decision": "deny",
    "policy_modality": "cannot",
    "subjects": [{"entity_type": "user", "ref_tokens": ["lecturer"], "role": "visiting_lecturer", "qualifiers": {"type": "visiting", "rank": None, "status": "retired", "office": None, "department": None, "negated": False}, "namespace": "subject.user.visiting_lecturer.visiting"}],
    "actions": [{"verb": "teach", "operation": "EXECUTE", "negated": True}],
    "resource": {"entity_type": "academic", "ref_tokens": ["hours"], "label": "teaching_hours", "qualifier": None, "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.academic.teaching_hours"},
    "environments": [],
    "context": [{"id": "cond_1", "type": "limit", "trigger_phrase": "more than 560 hours per year", "left_operand": "subject.teaching_hours_per_year", "operator": ">", "right_operand": None, "value": "560", "value_unit": "hours", "approver_entity": None, "approval_state": None, "formal_expression": "subject.teaching_hours_per_year > 560", "negated": False}],
    "relation_pairs": [
      {"entity": "subject", "rel_type": "attribute", "attribute": "type", "value": "visiting"},
      {"entity": "subject", "rel_type": "attribute", "attribute": "status", "value": "retired"}
    ]
  },
  99: {
    "authorization_decision": "permit",
    "policy_modality": "must",
    "subjects": [{"entity_type": "agreement", "ref_tokens": ["agreement"], "role": "visiting_lecturer_agreement", "qualifiers": {"type": "valid", "rank": None, "status": "valid", "office": None, "department": None, "negated": False}, "namespace": "subject.agreement.visiting_lecturer_agreement"}],
    "actions": [{"verb": "have", "operation": "UPDATE", "negated": False}],
    "resource": {"entity_type": "data", "ref_tokens": ["number"], "label": "contract_number", "qualifier": "a", "scope": None, "subject_filter": None, "attributes": None, "namespace": "resource.data.contract_number"},
    "environments": [{"id": "env_1", "env_type": "system", "trigger_phrase": "the visiting lecturer management system", "trigger_word": "in", "systems": [{"label": "the visiting lecturer management system", "namespace": "environment.device.device_type:visiting_lecturer_management_system"}], "time_range": None}],
    "context": [],
    "relation_pairs": [
      {"entity": "subject", "rel_type": "attribute", "attribute": "status", "value": "valid"}
    ]
  }
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

print("Patched successfully!")
