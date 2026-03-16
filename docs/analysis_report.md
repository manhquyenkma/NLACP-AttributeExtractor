# Phân Tích Chuyên Sâu: Trích Xuất Thuộc Tính Môi Trường (Environment Attributes)

Trong hệ thống ABAC (Attribute-Based Access Control), bên cạnh yếu tố **Ai** (Subject - Chủ thể), **Cái Gì** (Object - Khách thể) và **Hành Động** (Action), thì **Môi Trường** (Environment) đóng vai trò sống còn. Nó quy định các điều kiện ngoại cảnh như: thời gian nào, ở đâu, dùng thiết bị hay mạng nào thì mới được phép truy cập.

Bài báo gốc 2019 (Alohaly et al.) gặp khó khăn lớn khi tách Environment. Họ chỉ dùng các quy tắc cấu trúc câu (dependency parser) cơ bản. Nhưng trong tiếng Anh, từ chỉ thời gian/không gian thường nằm ở các "Cụm giới từ" hoặc đuôi của câu, rất dễ bị phần mềm nhận diện nhầm.

Để giải quyết triệt để, dự án này đã phát triển một file riêng biệt là `src/env_extractor.py`, áp dụng **Phương pháp Lai (Hybrid Approach) với 3 Lớp Lọc (Layers) và 1 Bước Hậu Xử Lý**. Dưới đây là cách hệ thống này hoạt động, được giải thích đơn giản nhất.

---

## Các Nhóm Thuộc Tính Môi Trường Xác Định Sẵn
Đầu tiên, hệ thống chia Environment Attribute thành 2 nhóm lớn để dễ xử lý:
1. **TEMPORAL (Thời gian)**: Các điều kiện về giờ giấc, ngày tháng, ca làm việc.
2. **SPATIAL (Không gian & Thiết bị)**: Vị trí địa lý, mạng lưới (mạng nội bộ/VPN), thiết bị (PC/Laptop).

---

## Cách Hoạt Động Chi Tiết (Từng Bước)

Khi một câu chính sách (VD: *"A doctor can view patient records during business hours."*) truyền vào hệ thống, thư viện trí tuệ nhân tạo **spaCy** sẽ đọc câu đó và biến nó thành một cây cấu trúc từ vựng (Dependency Tree). Sau đó, câu đi qua 3 "Lớp lưới lọc" sau:

### Lớp Lọc 1 (Layer 1): Bắt Cụm Giới Từ Bằng Bộ Quy Tắc (Rule-based)
Hệ thống dựa trên thói quen người dùng thường dùng **giới từ** để chỉ thời gian hoặc nơi chốn.

- **Bước 1 (Tìm Giới từ mồi - Triggers)**: 
  - Mồi thời gian: `during`, `between`, `after`, `within`, `at`, `on`.  
  - Mồi không gian: `from`, `at`, `inside`, `outside`, `via`, `on`.
- **Bước 2 (Gom Cụm Danh Từ - Noun Phrase)**: Nếu tìm thấy từ mồi (VD: từ `during` trong câu ví dụ), hệ thống sẽ tìm *tất cả chữ kết nối sau nó tạo thành một cụm danh từ* (hệ thống dùng hàm đệ quy chặt cành con - subtree). Ở đây nó gom được chữ `business hours`.
- **Bước 3 (Đối chiếu Từ Khóa Gợi Ý - Hints)**: Cụm `business hours` sẽ được mang đi quét xem có chứa các chữ đặc trưng của mốc thời gian không (như `hours`, `shift`, `day`, `pm`...). 
  - Do có kết quả trùng khớp, hệ thống chốt ngay: **"during business hours"** là thuộc tính **Thời gian (Temporal)**!

### Lớp Lọc 2 (Layer 2): Tìm Thiết Bị Truy Cập Đặc Thù (Device/Channel) - Fix Lỗi spaCy
Từ `using`, `via`, `through` là các chữ rất quan trọng để chỉ định phương thức truy cập (VD: *"using trusted workstations"*). Tuy nhiên, thư viện AI (spaCy) lại cấu hình sẵn rằng đây là các dạng "mệnh đề phụ" thay vì "giới từ", do đó Lớp Lọc 1 lọt lưới đoạn này.

- **Cách khắc phục**: Tạo một bộ quét song song. Cứ hễ gặp chữ `using`, chữ `via`, Lớp 2 sẽ tự động vét trọn vẹn cụm động/tính/danh từ phía sau nó (VD: `trusted workstations`), dò tìm các chữ gợi ý như `laptop`, `vpn`, `network`, `terminal`. Nếu khớp, nó chốt luôn đây là hệ Không Gian dạng **Thiết Bị (Device)**.

### Lớp Lọc 3 (Layer 3): Nhận Diện Thực Thể (NER - Named Entity Recognition)
Sau 2 lớp quy tắc gò bó trên, hệ thống triển khai mạng học sâu nhận diện thực thể (NER) của spaCy như một chiếc lưới vét cuối cùng, quét toàn bộ câu.

- Nó sẽ tìm các vật thể tự động mà spaCy đã được học từ hàng triệu văn bản:
  - Nếu gặp thực thể nhãn `TIME`, `DATE` $\rightarrow$ Nó tự ghi nhận là **Temporal**.
  - Nếu gặp nhãn GPE (Thành phố/Quốc gia), LOC (Địa điểm), FAC (Cơ sở vật chất) $\rightarrow$ Nó lưu là **Spatial**.
- **Thông minh ở chỗ**: Lớp 3 luôn kiểm tra lại xem cụm chữ đó có bị Lớp 1 hoặc 2 lấy đi trước đó chưa. Tránh việc một thuộc tính bị trích xuất lặp lại 2 lần (Deduplication).

---

## Chống Nhiễu - Bộ Lọc "Hậu Xử Lý" (Post-Processing)

Một parser phân loại tự động luôn có những "nhận diện ngớ ngẩn" (False Positives). Vì vậy, hệ thống thiết kế một bước làm sạch (`_filter_false_positives`):

1. **Từ chối ghép Môi trường với Chủ thể (Subject-att)**:
   - Các câu dạng *"from nurses / from doctors"*. Chữ `from` làm Layer 1 tưởng là không gian, nhưng bộ phận làm sạch quét dính chữ `nurses` (nhân viên y tế). Nghĩa là đây là quy định nói người gửi chứ không phải môi trường truy cập. Ngừng nhận diện!
2. **Từ chối Chuỗi hành động**:
   - Câu *"after reviewing documents"*. Layer 1 có chữ `after` sẽ tưởng là thời gian. Nhưng ngay sau nó chứa tiếp các dạng động từ đuôi `-ing` như `reviewing`, `submitting`, `approving`. Bộ hệ thống hiểu rằng đây là một chuỗi hành động bắt buộc, không tính làm môi trường. Block luôn!

---

## Giá Trị Triển Khai (Dễ gán Namespace sau này)
Không chỉ đơn thuần lấy cụm chữ ném vào là xong. Mã nguồn còn có các hàm cực kì quan trọng: `_classify_temporal` và `_classify_spatial`. 
- Nó phân loại nhỏ cụm từ đó ra. VD nếu cụm tgian có số (8am), gán loại là `absolute`. Còn nếu chứa chữ "weekend", gán nhãn `recurring` (lặp định kỳ). 
- Tính năng này vô giá vì về sau khi lập trình hệ tầng phân quyền (Namespace), hệ thống sẽ dựa vào cái Subcategory này để thiết kế luật máy tính tự động cho tường lửa.
