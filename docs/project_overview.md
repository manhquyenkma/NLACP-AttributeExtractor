# Báo Cáo Tổng Thể Dự Án Bóc Tách Thuộc Tính ABAC (NLACP)

## 1. Mục Tiêu Dự Án
- Tự động hóa quá trình trích xuất các thuộc tính ABAC (Attribute-based Access Control) từ các câu chính sách bảo mật viết bằng ngôn ngữ tự nhiên (Natural Language).
- Lấy nền tảng framework 5 module của bài báo gốc **Alohaly et al. (2019)**, đồng thời khắc phục các hạn chế mà bài báo chưa giải quyết được (cụ thể là trích xuất và phân nhóm thuộc tính môi trường - Environment Attributes).
- Hiển thị đầu ra chuẩn hóa với định dạng JSON, có cấu trúc linh hoạt để tích hợp mượt mà vào các hệ quản trị tiếp nhận hoặc hệ tầng kiểm soát truy cập trực tiếp.

## 2. Kết Quả và Yêu Cầu Đã Đạt Được
- **Cải thiện Dimension của Thuộc Tính**: Thay vì duy trì 5 chiều (dimension) phức tạp của bài báo, dự án đã tái cấu trúc lại thành 4 chiều chính thức (`name`, `value`, `category`, `data_type`, đi kèm `dep_relation`). Thiết kế này tinh gọn hơn nhiều, phục vụ tốt cho quá trình gán cấp bậc không gian (Namespace hierarchy).
- **Hoàn thiện Module Trích Xuất Tích Hợp (M1, M4, M5)**: Hoàn thành việc kết hợp Dependency Parsing (Trích xuất các yếu tố Subject, Object) với NER và Hybrid Rules (Trích xuất yếu tố Environment hiếm gặp). Tích hợp khả năng tự động luận kiểu dữ liệu (`String`, `Integer`, `Datetime`).
- **Xây Dựng Cơ Chế Cụm Thuộc Tính Tiến Tiến (M2)**: Đã áp dụng và chạy thành công thuật toán phân cụm không gian mật độ DBSCAN, nổi bật là có khả năng tự động tuning giá trị $\epsilon$ (Epsilon). Đã và đang chuyển vùng ánh xạ vector từ TF-IDF đơn thuần sang dạng từ vựng biểu diễn sâu sắc **GloVe 300d**.
- **Tạo Công Cụ Chuẩn Bị Dữ Liệu**: Tiên phong hoàn thiện Tool `annotate_helper.py` cung cấp UI/Terminal hỗ trợ việc người dùng gán nhãn thủ công (Annotation). Quy trình này tạo bước đệm huấn luyện mạng nơ-ron tích chập (CNN) cho bài toán tìm "Short name" ở giai đoạn sau.

## 3. Cấu Trúc Tổng Thể Các Module Và Cách Hoạt Động
Dự án được cấu trúc gọn nhẹ thành các module, được kết nối với nhau và chạy tuần tự thông qua tệp tin trung tâm `main.py`:

### **- Nhóm Module Khởi Tạo (Module 1, 4, 5): NLP Extraction + DataType Inference** 
- **Các File:** `src/nlp_engine.py`, `src/relation_candidate.py`, `src/env_extractor.py`, `src/data_type_infer.py`.
- **Chức năng:**  
  1. Module sẽ đọc từng câu khai báo chính sách đầu vào từ tệp raw.
  2. Bóc tách và định cấu trúc cú pháp của câu (Dependency Parsing) kết hợp với nhận diện thực thể (NER).
  3. Lấy ra chủ thể kích hoạt (Subject) và đối tượng thao tác (Object) qua các Modifier (ví dụ mẫu phụ thuộc amod, compound).
  4. Trích xuất thuộc tính cấu trúc ẩn là Environment Attributes thông qua bộ định vị trung gian (Hybrid Rule).
  5. Đánh giá và lưu trữ kiểu định dạng giá trị dữ liệu (Data Type).
- **Đầu ra:** Dữ liệu thuộc tính JSON gốc (`dataset/policy_dataset.json`).

### **- Nhóm Module Khai Phá (Module 2): Attribute Clustering**
- **Các File:** `mining/attribute_cluster.py` (và `attribute_cluster_glove.py` bản nâng cấp).
- **Chức năng:**
  1. Nhận diện và quy giải các vấn đề từ đồng nghĩa/đa nghĩa (ví dụ: `doctor`, `physician`, hay `medical staff` là một).
  2. Map toàn bộ các thuộc tính text dưới dạng vector TF-IDF hoặc GloVe Embeddings.
  3. Chạy thuật toán DBSCAN để nhóm các vector thuộc tính có khoảng cách mật độ gần nhau nhất thành "các cụm" (clusters) có ý nghĩa tương đương đại diện.
- **Đầu ra:** Tệp phân loại cụm (`dataset/attribute_clusters.json`).

### **- Nhóm Module Hậu Xử Lý (Module 3): Namespace Assignment & Hierarchy**
- **Các File:** `mining/namespace_hierarchy.py`.
- **Chức năng:** Dựa trên các cụm đã phân vùng, kết hợp với metadata hệ thống, tiến hành việc phân tầng kiến trúc quản ký quyền ABAC. Gán các attribute vào một không gian phân cấp (namespace hierarchy) chuẩn xác (ví dụ `subject:role:senior`, `object:medical_record`, `environment:time`).
- **Đầu ra:** Tệp không gian tên cấp bậc (`dataset/namespace_hierarchy.json`).
