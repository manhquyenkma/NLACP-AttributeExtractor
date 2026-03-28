# Báo cáo Dự án: NLACP-AttributeExtractor

## 1. Lời nói đầu (Giới thiệu)
Mô hình Kiểm soát Truy cập dựa trên Thuộc tính (Attribute-Based Access Control - ABAC) là một cơ chế linh hoạt và chi tiết được sử dụng để bảo mật các hệ thống công nghệ hiện đại. Tuy nhiên, việc định nghĩa các chính sách ABAC một cách thủ công thường tốn nhiều thời gian và dễ xảy ra lỗi do con người. Các chính sách bằng Ngôn ngữ Tự nhiên (NLACPs) chứa đựng các quy tắc truy cập được viết bằng ngữ nghĩa tiếng Anh thông thường.
Dự án **NLACP-AttributeExtractor** được ra đời nhằm thu hẹp khoảng cách giữa chính sách do con người đọc và bộ quy tắc do máy tính thực thi. Dự án sử dụng quy trình kết hợp giữa Xử lý Ngôn ngữ Tự nhiên (NLP) và Học máy Sâu (Deep Learning) để tự động trích xuất các thuộc tính: Chủ thể (Subject), Hành động (Action), Đối tượng (Object) và Môi trường (Environment), sau đó tổ chức chúng thành định dạng JSON có cấu trúc rõ ràng.

## 2. Mục tiêu của dự án
1. **Trích xuất Tự động:** Phân tích câu tiếng Anh phi cấu trúc để nhận diện các thành phần ABAC quan trọng (Chủ thể, Hành động, Đối tượng, Môi trường).
2. **Xác thực Tương tác:** Cung cấp cơ chế "Human-in-the-loop" (HitL) cho phép người dùng phê duyệt hoặc từ chối các cặp thuộc tính ứng viên do máy đề xuất trước khi hậu xử lý.
3. **Xử lý Mạnh mẽ & Tối ưu:** Áp dụng pipeline 2-Bước để phân tách an toàn các nhóm từ chỉ môi trường, tránh tình trạng rò rỉ (leak) dữ liệu môi trường vào các thuộc tính Chủ thể hay Đối tượng thông thường.
4. **Chuẩn hóa Dữ liệu:** Đồng bộ gán Tên viết tắt (Short Names), Không gian tên (Namespaces) và Kiểu dữ liệu (Data Types, ví dụ: Số nguyên, Chuỗi, Thời gian, Địa điểm) cho mỗi thuộc tính được trích xuất.

## 3. Kiến trúc Hệ thống
Hệ thống tuân theo **Kiến trúc Pipeline Lai 2 Bước (2-Step Hybrid Pipeline Architecture)** được cấu trúc lại:

- **Bước 1: Xử lý Dữ liệu & Trích xuất Ứng viên (Data Processing)**
  Đọc văn bản đầu vào, phân tích quan hệ cú pháp (dependency tree) thông qua thư viện `spaCy` để chỉ ra các cặp thuộc tính quan hệ tiềm năng. Một giao diện Command-Line (CLI) sẽ yêu cầu người dùng xác minh từng cặp ứng viên nhằm loại bỏ các thông tin nhiễu.
- **Bước 2: Hậu xử lý Thuộc tính ABAC (ABAC Extraction)**
  Tập trung xử lý các cặp đã được xác thực để nội suy bối cảnh môi trường chính xác (Không gian/Vật lý, Không gian/Mạng, Thời gian) và chú thích các thuộc tính Chủ thể/Đối tượng hợp lệ. Kết quả cuối cùng được trả về file `policy_dataset.json`.

[Chèn Ảnh chụp Màn hình Giao diện / Sơ đồ Kiến trúc vào đây]

## 4. Giải thích các Module/Thành phần
Logic cốt lõi của hệ thống được đặt trong gói `nlacp/` và chia thành các module với nhiệm vụ riêng biệt:

- **Module Trích xuất (`nlacp/extraction`)**
  - `relation_candidate.py`: Ứng dụng phân tích quan hệ cú pháp để tìm ra các chuẩn ghép cặp "Chủ thể - Thuộc tính" (Subject-Attribute) và "Đối tượng - Thuộc tính" (Object-Attribute).
  - `env_extractor.py`: Xác định các giới từ (ví dụ: *during*, *within*) dùng để phân lập các yếu tố không gian/thời gian môi trường.
  - `short_name_suggester.py`: Loại bỏ giới từ stop words và định dạng lại các thuộc tính thành dạng tên viết gọn (`short_name`).

- **Module Chuẩn hóa (`nlacp/normalization`)**
  - `namespace_assigner.py`: Phân nhóm các thuộc tính dưới một cấu trúc namespace tiêu chuẩn (ví dụ: `subject:role:senior_nurse`).
  - `data_type_infer.py`: Suy luận giá trị của thuộc tính đại diện cho `string`, `datetime`, `location`, hay `integer`.

- **Module Khai phá (`nlacp/mining`)** *(Các bước nâng cao tùy chọn)*
  - `attribute_cluster.py`: Sử dụng thuật toán GloVe embeddings và luật DBSCAN để phân cụm các thuộc tính mang ý nghĩa tương đồng.
  - `namespace_hierarchy.py`: Xây dựng một cây kế thừa (cha-con) cho các quy tắc kiểm soát truy cập dựa trên những tập thuộc tính tập con (subsets).

## 5. Luồng dữ liệu / Quy trình hoạt động (Workflow)
1. **Đầu vào (Input):** Người dùng nhập một câu chính sách (VD: *"A senior nurse can approve records during business hours."*) thông qua script `scripts/data_processing.py`.
2. **Gắn thẻ NLP (NLP Tagging):** `spaCy` phân tích Từ loại (POS) và cấu trúc quan hệ ngữ pháp của câu.
3. **Xác thực Ứng viên (Candidate Validation):** Scripts đề xuất các cặp thuộc tính như `[nurse, senior]` và `[hours, business]`. Người dùng thao tác tương tác y/n để phê duyệt/từ chối.
4. **Lưu trữ Trung gian:** Các cặp được chọn sẽ lưu ở dạng log tạm thời trong `dataset/relation_candidate.json`.
5. **Tiền xử lý nâng cao (Enrichment):** Chạy lệnh `scripts/ABAC_extraction.py` để tách biệt phần môi trường (`during business hours`) và hoàn thành tài liệu ABAC JSON cấu trúc chuẩn mực tại `dataset/policy_dataset.json`.

[Chèn Ảnh chụp Màn hình Test Terminal / Workflow vào đây]

## 6. Các Thuật toán Cốt lõi
- **Khớp Cú pháp Phụ thuộc (Syntactic Dependency Matching):** Engine chạy bằng luật sẽ rà quét cây cú pháp để bắt các điểm kích hoạt như `amod` (bổ ngữ tính từ) hoặc `compound` (danh từ ghép) đã liên kết với Subject hoặc Object.
- **Logic Phân loại Môi trường:** Áp dụng bộ từ khóa gợi ý để phân loại bối cảnh không gian kỹ thuật (VD: "vpn", "internet" -> `spatial_network`; "workstation", "laptop" -> `spatial_device`; dự phòng -> `spatial_physical`).

## 7. Đánh giá Thuật toán (Evaluation)
Hệ thống được đánh giá độ chính xác thông qua 2 Module chính, bám sát các phương trình của Alohaly et al. (2019):

**1. Module 1 - Trích xuất Môi trường (Environment)**
Đánh giá trên bộ dữ liệu `vact_env_annotated.json` (100 câu), kết quả trích xuất:
- **Thời gian (Temporal):** Precision = 0.8958 | Recall = 0.9149 | F1-score = 0.9053
- **Không gian (Spatial):** Precision = 1.0000 | Recall = 1.0000 | F1-score = 1.0000
- **Tổng thể (Overall):** Precision = 0.9597 | Recall = 0.9675 | F1-score = 0.9636
*Kết luận:* Mức F1-score > 0.96 cho thấy module trích xuất hoạt động vô cùng ổn định và vượt mục tiêu thực nghiệm đề ra.

**2. Module 2 - Phân cụm Không gian giá trị (DBSCAN Clustering)**
Dựa trên thuật toán gán nhãn cụm (nhãn chiếm đa số) và công thức $n_{ij}$ trên tập `attribute_clusters.json`:
- **Average Precision:** 0.5000
- **Average Recall:** 0.7500
- **Average F1-score:** 0.5833
*Lưu ý:* Tập `policy_dataset.json` hiện tại khá nhỏ nên số cụm hình thành chưa nhiều. Để đạt F1-score cao, mô hình DBSCAN cần chạy trên bộ dữ liệu quy mô lớn (VD: LitroACP).

## 8. Kết luận
Dự án "NLACP-AttributeExtractor" đã thành công trong việc chuyển đổi các chính sách bằng ngôn ngữ tự nhiên tối nghĩa, phức tạp thành các hệ thuộc tính ABAC kiểm soát truy cập rõ ràng. Với việc cấu trúc lại theo kiến trúc Pipeline 2 bước, giải pháp này là sự tổng hòa giữa việc trích xuất xử lý bằng NLP tốc độ cao cùng với độ tin cậy tuyệt đối nhờ sự kiểm duyệt thực tế của con người (human-in-the-loop), mở ra khả năng mở rộng mạnh mẽ cho các hệ thống an ninh hiện đại.
