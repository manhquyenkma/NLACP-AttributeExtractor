# NLACP-AttributeExtractor

## Mô tả Dự án
NLACP-AttributeExtractor là một hệ thống pipeline AI chuyên dụng được thiết kế để tự động trích xuất các chính sách "Attribute-Based Access Control" (ABAC - Kiểm soát truy cập dựa trên thuộc tính) từ các câu chính sách bằng ngôn ngữ tự nhiên (NLACPs). Hệ thống chuyển đổi các câu tiếng Anh phi cấu trúc thành định dạng JSON ABAC có cấu trúc mà máy tính có thể đọc được.

Kiến trúc mới được cập nhật sử dụng **Pipeline Lai 2 Bước (2-Step Hybrid Pipeline)** kết hợp giữa trích xuất NLP dựa trên luật (Rule-based) và xác thực Học máy tương tác (CNN) nhằm đảm bảo độ chính xác cao, đồng thời phân lập chính xác các Thuộc tính Môi trường (Environment) khỏi các Thuộc tính Chủ thể/Đối tượng (Subject/Object).

---

## Tính năng Chính
- **Kiến trúc Pipeline 2 Bước:** Tách biệt việc trích xuất/xác thực các ứng viên quan hệ (relation candidates) với bước hậu xử lý và gán nhãn thuộc tính.
- **Xác thực Human-in-the-Loop:** Giao diện Command Line tương tác cho phép người dùng xác nhận các cặp ứng viên quan hệ Subject-Attribute và Object-Attribute.
- **Trích xuất Môi trường (Environment) Nâng cao:** Phân lập và phân loại môi trường Không gian (Vật lý/Mạng/Thiết bị) và Thời gian (Tuyệt đối/Tương đối/Lặp lại).
- **Cấu trúc ABAC Tự động:** Tự động suy luận Tên viết tắt (Short Name), Không gian tên (Namespace) và Kiểu dữ liệu (Data Type) cho các thuộc tính.
- **NLP Fallback Mạnh mẽ:** Chuyển đổi linh hoạt và an toàn giữa các mô hình `spaCy` để đảm bảo hệ thống không bị lỗi.

---

## Công nghệ Sử dụng
- **Ngôn ngữ:** Python 3.8+
- **Framework NLP:** spaCy (`en_core_web_sm` / `en_core_web_md`)
- **Machine Learning:** scikit-learn, numpy (cho việc phân cụm GloVe embeddings & DBSCAN)
- **Định dạng Output:** JSON

---

## Cấu trúc Dự án
```text
NLACP-AttributeExtractor/
├── dataset/                        # Datasets (Input Annotations & Runtime JSON Outputs)
│   ├── relation_candidate.json     # Các cặp quan hệ đã trích xuất chờ xát thực
│   └── policy_dataset.json         # Danh sách policy ABAC cuối cùng
├── docs/                           # Tài liệu và báo cáo chính thức
├── nlacp/                          # Core Package (Mã nguồn lõi)
│   ├── extraction/                 # Các Module NLP trích xuất (S-A-O, Env, Candidates)
│   ├── normalization/              # Module Chuẩn hóa (Namespace & DataType)
│   ├── mining/                     # Module Khai phá (Clustering & Hierarchy)
│   ├── paths.py                    # File định nghĩa đường dẫn chuẩn
│   └── pipeline/                   # Pipeline nguyên bản (Single-pass)
├── scripts/                        # Các Script thực thi chính
│   ├── data_processing.py          # [BƯỚC 1] Trích xuất & Xác thực Tương tác
│   ├── ABAC_extraction.py          # [BƯỚC 2] Điền Env & Chuẩn hóa Thuộc tính
│   ├── run_pipeline.py             # Script chạy toàn bộ pipeline
│   ├── run_evaluation.py           # Công cụ đo đạc F1-score (thế chỗ các script eval cũ)
│   ├── eval_policy_f1.py           # Chuyên đánh giá mô đun Environment
│   └── mock_llm.py                 # (Optional) Mô phỏng LLM tạo text annotations
├── archive/                        # Các tệp kịch bản cũ và data tools chỉ chạy 1 lần
├── tests/                          # Unit và Integration Tests
├── README.md                       # Tài liệu mô tả dự án
└── Run.txt                         # Hướng dẫn chạy dự án ngắn gọn
```

---

## Hướng dẫn Cài đặt

1. **Clone repository và di chuyển vào thư mục dự án:**
   ```bash
   git clone <repository-url>
   cd NLACP-AttributeExtractor
   ```

2. **Tạo và kích hoạt môi trường ảo (virtual environment):**
   ```bash
   python -m venv venv
   # Trên Windows:
   venv\Scripts\activate
   # Trên macOS/Linux:
   source venv/bin/activate
   ```

3. **Cài đặt các thư viện lõi & Mô hình ngôn ngữ:**
   ```bash
   pip install spacy scikit-learn numpy pytest
   python -m spacy download en_core_web_md
   python -m spacy download en_core_web_sm
   ```

---

## Hướng dẫn Sử dụng & Lệnh Demo

### 1. Pipeline Tương tác 2 Bước
**Bước 1: Data Processing (Trích xuất + Xác thực Tương tác)**
Trích xuất Subject, Action, Object và các Ứng viên Quan hệ. Yêu cầu người dùng xác nhận (y/n) cho mỗi cặp thuộc tính.
```bash
python scripts/data_processing.py
```
*Ví dụ Input:* `A senior nurse can approve medical records during business hours.`
*Output:* Lưu các cặp được validate vào `dataset/policy_dataset.json`

**Bước 2: ABAC Extraction (Tiền xử lý & Điền Môi trường)**
Điền các thuộc tính Environment, dọn dẹp các từ trùng lặp và gán namespaces/kiểu dữ liệu.
```bash
python scripts/ABAC_extraction.py
```

### 2. Chạy Test Nhanh 1 Câu (Không tương tác)
```bash
python scripts/run_pipeline.py --sentence "Managers in the finance department can view expense reports over the VPN."
```

### 3. Đánh giá thuật toán (Evaluation)
Hệ thống đi kèm công cụ đánh giá F1-score để tự động đo đạc độ chính xác:

**Đánh giá Module 1 (Trích xuất Subject/Object trên tập dữ liệu chuẩn Alohaly):**
```bash
python scripts/run_evaluation.py
```

**Đánh giá Module 1 (Trích xuất Context-Environment trên tập policy_dataset):**
```bash
python scripts/eval_policy_f1.py --csv --csv-path dataset/annotation_llm_gold.csv
```

**Đánh giá Module 2 (Phân cụm Không gian giá trị - DBSCAN):**
Tính toán F1-score của các cụm phân tích (công thức $n_{ij}$, $n_i$, $n_j$ theo Alohaly 2019):
```bash
python scripts/run_evaluation.py --cluster
```
Kết quả được xuất ra log console hoặc file trong thư mục `outputs/logs/`.

---

## Lưu ý
- Đảm bảo mô hình `en_core_web_md` đã được cài đặt để pipeline phân loại quan hệ và nhóm thuộc tính hoạt động chính xác nhất. Hệ thống có cơ chế fallback sang `en_core_web_sm` nhưng độ chính xác của vector embeddings sẽ giảm.
- Các token chỉ môi trường (ví dụ: `during`, `business`, `hospital`) sẽ được lọc tự động khỏi danh sách thuộc tính Subject/Object ở Bước 2 để cấu trúc ABAC được gọn gàng và chuẩn xác.
