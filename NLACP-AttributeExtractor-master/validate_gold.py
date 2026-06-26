import json

with open('outputs/policies/policy_dataset_gold.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

errors = []
warnings = []

# Validate specific rules
forbidden_resources = [
    'affairs_office', 'assurance_office', 'academic_affairs', 'educational_testing',
    'learning_management_system', 'academic_information_system', 'visiting_lecturer_management_system'
]

fallback_ids = [68, 69, 72, 75, 92, 95, 96, 99]

for p in data['policies']:
    p_id = p['id']
    sentence = p['sentence'].lower()
    
    # 1. Check if required keys exist
    if 'subjects' not in p or 'resource' not in p:
        errors.append(f"ID {p_id}: Missing 'subjects' or 'resource' key.")
        continue

    # 2. Check fallback policies are populated
    if p_id in fallback_ids:
        if not p.get('subjects') and not p.get('resource'):
            errors.append(f"ID {p_id}: Fallback policy is empty!")

    # 3. Check for forbidden resource labels (offices or systems mistakenly set as resource)
    res = p.get('resource')
    if res:
        res_label = res.get('label')
        if res_label in forbidden_resources:
            errors.append(f"ID {p_id}: Resource label '{res_label}' should not be an office or system!")
            
    # 4. Check "Users in the [X] office" missing qualifiers
    if "users in the" in sentence and "office" in sentence:
        subjects = p.get('subjects', [])
        has_office = False
        for s in subjects:
            if s.get('qualifiers') and s['qualifiers'].get('office'):
                has_office = True
        if not has_office:
            warnings.append(f"ID {p_id}: Sentence mentions 'office' but Subject has no office qualifier.")

if not errors and not warnings:
    print("HOÀN HẢO! File Gold Standard đã hoàn toàn sạch sẽ không còn lỗi cấu trúc hay nhầm lẫn tài nguyên/hệ thống.")
else:
    if errors:
        print("PHÁT HIỆN LỖI:")
        for e in errors:
            print("-", e)
    if warnings:
        print("\nCẢNH BÁO (Nên xem lại):")
        for w in warnings:
            print("-", w)
