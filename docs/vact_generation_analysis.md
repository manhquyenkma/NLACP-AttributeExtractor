# Phân Tích Quá Trình Sinh Bộ Dữ Liệu VACT (100 Câu)

Bộ dữ liệu VACT được sinh ra tự động để phục vụ cho việc kiểm thử (testing) và đánh giá độ chính xác của Sub-module trích xuất thuộc tính Môi trường (Environment Attribute Extractor) và các thành phần cốt lõi của dự án NLACP. Quá trình này được thực hiện tự động thông qua script `src/generate_vact_dataset.py`.

Dưới đây là phân tích chi tiết về cơ chế hoạt động, cấu trúc dữ liệu và mục đích của đoạn script này.

## 1. Mục Đích Của Việc Sinh Dataset VACT
- **Tạo Ground-Truth (Dữ liệu chuẩn)**: Thay vì gán nhãn thủ công hàng trăm câu, phần mềm tự sinh ra 100 câu tiếng Anh mô tả chính sách bảo mật kèm theo nhãn gán sẵn (`annotated: True`). Bộ nhãn này đóng vai trò là "đáp án chuẩn" để so sánh với kết quả mà AI/Parser của chúng ta tự động trích xuất ra.
- **Mô phỏng Ngữ Cảnh Thực Tế (VACT context)**: Đặc tả danh từ, chức danh và tài nguyên bám sát mô hình của một học viện/tổ chức giáo dục (Ví dụ: `VACT students`, `exam grades`, `cryptography lab`).
- **Ép Tải Độ Khó (Stress Test)**: Đưa ra các trường hợp phức tạp (như câu có 2 môi trường cùng lúc, câu nối ghép 2 hành động bằng chữ `and`) để kiểm tra giới hạn của cây cú pháp (Dependency Tree).

## 2. Cấu Trúc Thành Phần Câu (Building Blocks)
Script định nghĩa 4 kho từ vựng cốt lõi làm mỏ neo để random:
1. **Subjects (Chủ thể)**: 13 chức danh như `Students`, `Lecturers`, `System administrators`, `Guest users`, v.v.
2. **Actions (Hành động)**: 12 động từ thao tác dữ liệu như `access`, `view`, `modify`, `download`, `approve`, v.v.
3. **Objects (Khách thể)**: 12 loại tài nguyên điển hình như `exam grades`, `course materials`, `system logs`, `network configurations`.
4. **Environment Phrases (Cụm từ Môi trường)**: Phân thành 3 nhóm rõ rệt để đối sánh với các Subcategory của module phân tích:
   - **Temporal (Thời gian)**: `during business hours`, `on weekends`, `at nighttime`, v.v.
   - **Spatial - Physical/Network (Không gian vật lý/mạng)**: `within the VACT intranet`, `from the campus network`, `at the cryptography lab`.
   - **Spatial - Device (Không gian thiết bị)**: `using authorized workstations`, `via a secure VPN`.

## 3. Thuật Toán Sinh Câu (Sentence Generation Logic)
Mã nguồn sử dụng vòng lặp 100 lần (được cố định Seed ngẫu nhiên `random.seed(42)` để đảm bảo mỗi lần sinh lại đều ra đúng 100 câu đó, giúp bài test nhất quán). Logic sinh câu qua các bước:

### Bước 1: Chọn Thành Phần Chính
Hệ thống lấy ngẫu nhiên 1 Subject, 1 Action, và 1 Object từ các danh sách trên.

### Bước 2: Trộn Môi Trường (Mấu chốt của Tester)
- **Tỉ lệ phân bổ Môi Trường**: Theo cấu hình, **100% các câu sinh ra đều ép buộc phải có Environment Attribute** để kiểm thử gắt gao. Trong đó:
  - **70%** số câu sẽ được gắn 1 cụm môi trường (VD: chỉ có thời gian hoặc không gian).
  - **30%** số câu sẽ được bốc 2 cụm môi trường cùng lúc (VD: vừa có thời gian vừa có không gian kiểu nhồi nhét: `"during business hours from the campus network"`).

### Bước 3: Đánh Lừa Parser (Compound Action)
- Có **30% tỷ lệ** thuật toán sẽ nhặt thêm 1 Hành động (Action thứ 2) và nối chúng bằng chữ `and` (Ví dụ: `"can view and download"`). Rất nhiều bộ Parser của các bài báo NLP cũ thường đứt đuôi phân tích khi gặp chữ `and` ở giữa câu. Do đó đây là một bẫy cố ý.

### Bước 4: Lắp Ráp & Gán Nhãn Tự Động
Hệ thống kết hợp các từ theo form mẫu: 
`[Subject] can [Action] [Object] [Environment_1] [Environment_2].`

Trình sinh mã cũng tự động xây sẵn Label (Nhãn) chuẩn dạng JSON lưu vào biến `env_attributes`. 
```json
{
  "category": "temporal",
  "value": "during business hours",
  "trigger": "Manual",
  "source": "manual_annotation"
}
```

## 4. Đầu Ra Của Script
Mỗi câu sau khi hoàn thiện sẽ được bọc lại bằng định dạng chuẩn và đẩy vào List. Khi kết thúc vòng lặp 100, toàn bộ kết quả được Dump (xuất) ra file JSON tại đường dẫn:
👉 `data/annotated/vact_env_annotated.json`

File JSON này chính là nguyên liệu lõi đầu tư để chạy Script `evaluator.py`, giúp đối chiếu kết quả bóc tách tự động với nhãn chuẩn (Ground Truth) để tính ra tỷ lệ chính xác (Precision/Recall/F1-Score).
