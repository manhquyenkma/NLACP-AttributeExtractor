# Hướng Dẫn Chạy và Test Dự Án

## 1. Yêu Cầu Môi Trường Cài Đặt (Prerequisites)
Để khởi chạy được dự án, bạn cần đảm bảo hệ thống đã cài đặt Python (phiên bản 3.8 trở lên khuyên dùng). Cần cài đặt tất cả các thư viện thông qua các lệnh terminal (vào CMD/PowerShell) như sau:

```bash
# Cài đặt các thư viện lõi cho Machine Learning và NLP
pip install spacy scikit-learn nltk torch

# Tải xuống các pre-trained pipeline từ spacy
python -m spacy download en_core_web_sm
python -m spacy download en_core_web_md
```

---

## 2. Hướng Dẫn Chạy Toàn Bộ Dự Án (Full Pipeline)
Script trung tâm `main.py` là entry point quản lý và điều phối hoạt động của toàn bộ 5 module theo luồng bài báo Alohaly 2019. 

**Cách chạy:**
Mở terminal, điều hướng con trỏ đến thẳng thư mục gốc của dự án (`NLACP-AttributeExtractor`) và chạy hiệu lệnh:
```bash
python main.py
```

**Mô tả quá trình phần mềm chạy:**
1. Trình biên dịch sẽ gọi module trong nhánh `src/nlp_engine.py` đầu tiên, làm việc với các văn bản tự nhiên để tách ra tất cả các thuộc tính ABAC (Subject, Object, Environment, Data Type...).
2. Tiếp theo, trình biên dịch gọi `mining/attribute_cluster.py` để biểu diễn từ vựng vừa trích xuất qua Vector và chạy phân cụm DBSCAN.
3. Cuối cùng, trình biên dịch gọi `mining/namespace_hierarchy.py` để phân bổ cấu trúc phân tầng.

**Lưu ý sau khi chạy:** Toàn bộ kết quả thực thi sẽ được xuất ra và lưu theo cấu trúc tại các tệp tĩnh định dạng chuẩn tại thư mục `dataset/`:
- `dataset/policy_dataset.json`
- `dataset/attribute_clusters.json`
- `dataset/namespace_hierarchy.json`

---

## 3. Hướng Dẫn Test Nhanh (Test Riêng Lẻ Các Phần)

### a. Test Toàn Bộ Cấu Trúc Bằng Dữ Liệu Rút Gọn Mẫu
Nếu bạn chỉ muốn thử nghiệm với một vài câu đại diện (thay vì xử lý tập dữ liệu hàng nghìn câu) nhằm đảm bảo thư viện đã được cài đủ và code không có Bug nội tải:
```bash
python test_pipeline.py
```
**Giải thích:** Tệp này chạy mô phỏng ngầm Pipeline với 5 câu đầu mẫu trong nội tại code, và in log đầu ra trực tiếp lên terminal mà không tiến hành lưu vào ổ cứng dưới dạng tệp `dataset/`.

### b. Test Tiêu Điểm Cải Tiến: Cơ Chế Bắt Thuộc Tính Môi Trường (Env-Extractor)
Để theo dõi và hiểu cách thức hoạt động của công nghệ Hybrid Rule kết hợp NER bắt "Trạng thái Môi Trường", hãy chạy luồng riêng sau:
```bash
python src/env_extractor.py
```
**Giải thích:** File này đã được hardcode mảng dữ liệu văn bản là các tình huống chứa các cụm truy cập rất trừu tượng (vd: `during business hours`, `Nurses from the hospital network`). Nó sẽ console text (in) ra chi tiết quá trình bắt từng đoạn, Trigger nhận diện là gì, thuộc Category nào (Temporal hay Spatial) và bằng Phương Pháp (Method) xử lý nào (Rule hay NER).

---

## 4. Công Cụ Hỗ Trợ Gán Nhãn Tích Hợp (Data Annotation)
Để phục vụ tạo bộ Test/Train Corpus cho huấn luyện máy học (CNN Classification ở FIX 5), dự án trang bị riêng công cụ gán nhãn:
```bash
python annotate_helper.py
```
**Lưu ý:**
- Khởi chạy mã trên terminal, bạn sẽ tương tác vòng lặp bằng cách trả lời các prompt do ứng dụng cung cấp để định hình (gán nhãn) cú pháp thuộc tính.
- Chuỗi kết quả này được bọc lại và kết xuất lưu vào file `dataset/annotated_corpus.json`.
- Sau khi nhóm có đủ tập dữ liệu Annotated Corpus (khuyến nghị trên 200 câu), có thể sử dụng model CNN để huấn luyện bằng câu lệnh tham chiếu sau:
  ```bash
  python src/cnn_classifier.py --train --data dataset/annotated_corpus.json --type subject
  ```
