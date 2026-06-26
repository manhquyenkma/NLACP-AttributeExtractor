import json

with open('outputs/policies/policy_dataset_gold.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

with open('plan_generator.py', 'w', encoding='utf-8') as f:
    f.write('''import json
with open("outputs/policies/policy_dataset_gold.json", "r", encoding="utf-8") as f:
    data = json.load(f)

with open("C:/Users/PAV/.gemini/antigravity-ide/brain/83725689-f9ad-4adf-887e-6fc470a7f85e/implementation_plan.md", "w", encoding="utf-8") as out:
    out.write("# Sửa Subject và Object cho câu 50-100\\n\\n")
    out.write("Mục tiêu: Ánh xạ lại chính xác Subject và Object (Resource) theo đúng ngữ nghĩa của câu tiếng Anh cho các câu từ 50-100, khắc phục lỗi nhận nhầm của thuật toán (VD: nhận phòng ban làm tài nguyên).\\n\\n")
    out.write("## Proposed Changes\\n\\n")
    out.write("Dưới đây là danh sách chi tiết các sửa đổi dự kiến cho Subject và Resource của từng câu từ 50 đến 100. Vui lòng kiểm tra và duyệt:\\n\\n")
    
    for p in data["policies"][49:100]:
        out.write(f"### ID: {p['id']}\\n")
        out.write(f"**Sentence:** {p['sentence']}\\n")
        
        # Analyze current vs new
        out.write("**Mới (Proposed):**\\n")
        out.write("- **Subject:** (Cần điền)\\n")
        out.write("- **Resource:** (Cần điền)\\n\\n")
''')
