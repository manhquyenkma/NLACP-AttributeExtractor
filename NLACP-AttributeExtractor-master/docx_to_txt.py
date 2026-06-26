import docx

doc = docx.Document('VNICT2026_ABAC_NCKH.docx')
with open('VNICT2026_ABAC_NCKH.txt', 'w', encoding='utf-8') as f:
    for para in doc.paragraphs:
        if para.text.strip():
            f.write(para.text + '\n')
