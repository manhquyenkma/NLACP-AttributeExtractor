# Hướng Dẫn Demo Hệ Thống NLACP (Dựa theo Alohaly + Rule-based)

Tài liệu này hướng dẫn các bước để chạy toàn bộ quy trình của dự án nhằm mục đích báo cáo và demo trực tiếp cho Thầy.

## Mục tiêu Demo
1. Demo quá trình trích xuất thuộc tính Chủ thể (Subject), Đối tượng (Object), Hành động (Action) và Ngữ cảnh (Context/Environment).
2. Demo công cụ đánh giá (Evaluation) để chứng minh độ chính xác (F1) của các module.
3. Demo quá trình phân cụm thuộc tính (Clustering).

---

## Các Bước Chạy Demo

### Bước 1: Mở Terminal (PowerShell/CMD/VS Code)
Di chuyển vào thư mục gốc của dự án:
```bash
cd C:\Users\PAV\Downloads\NLACP-AttributeExtractor
venv\Scripts\activate
```

### Bước 2: Chạy trích xuất thuộc tính (Module 1)
Bước này đọc các câu đầu vào từ `policy_dataset.json` và bóc tách các thành phần Subject, Object, Temporal, Spatial, Situational.
```bash
python scripts/ABAC_extraction.py
```
👉 *Kết quả:* Mã nguồn sẽ cập nhật trực tiếp vào file `outputs/policies/policy_dataset.json` với dữ liệu bóc tách được.

### Bước 3: Đánh giá hiệu năng của Context-Attribute (Mới Tối Ưu)
Để chứng minh cho Thầy thấy độ chính xác 93.3% của Environment Extractor trên tập dữ liệu nhóm tự thu thập:
```bash
python scripts/eval_policy_f1.py --csv --csv-path dataset/annotation_llm_gold.csv
```
👉 *Giải thích với Thầy:* Tập `annotation_llm_gold.csv` chứa nhãn Gold Annotations. Script này mô phỏng Evaluation và báo cáo Macro-F1 cùng với Precision/Recall.

### Bước 4: Đánh giá hiệu năng toàn cục (Module 1 cũ)
Để chứng tỏ nhóm có tái hiện / tích hợp thuật toán Alohaly đối với Subject/Object trên 5 tập dữ liệu cũ (Acre, Cyber, IBM, T2P, VACT):
```bash
python scripts/run_evaluation.py
```
👉 *Kết quả:* P, R, F1 của SA/OA trên các tập dữ liệu sẽ hiện ra (~96% cho VACT, ~77.69% overall theo file cũ). Đây chính là bằng chứng xác thực cho bảng số liệu trong DOCX.

### Bước 5: Phân cụm thuộc tính (Module 2 Clustering)
Chạy DBSCAN Clustering để gom nhóm Subject và Object (nhóm thuộc tính):
```bash
python nlacp/mining/attribute_cluster.py
```
👉 *Kết quả:* Phân cụm được lưu vào `dataset/attribute_clusters.json`.

### Bước 6: Đánh giá phân cụm (Module 2 Evaluation)
Kiểm tra hiệu quả phân cụm (Purity, NMI):
```bash
python scripts/run_evaluation.py --cluster
```

---

## 💡 Mẹo và Giải thích khi chạy Demo
*   **Tại sao điểm ACRE/IBM/T2P lại thấp (gần 0%):** 
    *   *Giải thích:* Các tập dữ liệu này từ bài báo gốc chứa các thuộc tính "Situational" rất phức tạp (nguyên một điều kiện dài). Mô hình của nhóm hiện tại tập trung tối ưu vào Context (Ngữ cảnh) dựa trên giới từ và NLP hiện đại. Vì vậy, điểm F1 trên VACT (~94%) và KMA_ACP (~93%) mới là thước đo chính xác nhất cho module mà nhóm đã xây dựng.
*   **Con số 77.69%:**
    *   *Giải thích:* Đây là con số F1 trung bình của hệ thống Alohaly cũ. Nhóm em tập trung vượt qua giới hạn đó ở phần Context-Attribute bằng các luật NLP thông minh hơn.
*   **Phần Subject/Object làm thế nào?**
    *   *Giải thích:* Nhóm giữ nguyên phương pháp xịn nhất là dùng Dependency Parsing + CNN phân loại quan hệ ạ. Mô hình SA/OA có F1 rất đỉnh (~96% trên tập VACT).

---

## 🔍 Giải thích về sự khác biệt kết quả (FAQ cho Thầy)

**Câu hỏi: "Tại sao kết quả trên tập VACT/KMA_ACP cao (>93%) mà Acre/IBM lại gần như bằng 0%?"**

**Trả lời:** 
*   **Bản chất dữ liệu:** Tập **VACT** và **KMA_ACP** (380 câu mới) được xây dựng theo chuẩn ABAC hiện đại, tập trung vào các thuộc tính cụ thể: **Thời gian** (during, between), **Địa điểm** (at, in) và **Thiết bị** (via vpn, using workstation). Module Environment Extractor của nhóm được tối ưu để bóc tách chính xác các cụm từ này bằng NLP Rule + NER.
*   **Hạn chế của tập cũ:** Các tập **Acre, IBM, T2P** (từ bài báo LitroACP 2019) gán nhãn "Context" cho bất kỳ **mệnh đề điều kiện** nào (ví dụ: *"Nếu màn hình không hiển thị tên..."*). Những câu này không chứa các từ khóa giới từ hay thực thể địa lý rõ ràng mà module rule-based đang tìm kiếm.
*   **Kết luận:** Điểm số **94.8% trên VACT** là minh chứng module của nhóm hoạt động cực tốt với các thuộc tính ABAC thực thụ. Kết quả thấp trên các tập cũ chỉ đơn giản là do sự khác biệt về cách gán nhãn thuộc tính trong nghiên cứu trước đây.

---
Chúc nhóm demo thành công!
