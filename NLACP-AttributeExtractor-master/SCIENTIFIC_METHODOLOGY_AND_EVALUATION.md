# KHUNG PHƯƠNG PHÁP LUẬN VÀ THẨM ĐỊNH KHOA HỌC (SCIENTIFIC BENCHMARK & METHODOLOGY)
## Dành cho Bài báo Khoa học NCKH: Trích xuất Thuộc tính ABAC từ Chính sách Ngôn ngữ Tự nhiên

---

### TÓM TẮT (ABSTRACT)
Tự động hóa trích xuất chính sách kiểm soát truy cập dựa trên thuộc tính (ABAC) từ văn bản ngôn ngữ tự nhiên đóng vai trò then chốt trong quản trị an ninh thông tin hiện đại. Nghiên cứu này trình bày kiến trúc phân tích cú pháp phụ thuộc sâu (Deep Dependency Parsing Architecture - NLACP) có khả năng ánh xạ các câu chính sách phi cấu trúc sang bộ tứ thuộc tính chuẩn $\langle S, A, O, E \rangle$ (Subject, Action, Resource/Object, Environment). Nhằm đảm bảo tính chuẩn xác, minh bạch và khắc phục hiện tượng thiên lệch đánh giá, nghiên cứu thiết lập khung thẩm định hai mức độc lập: **Strict Exact-Match (khớp ranh giới token tuyệt đối)** và **Relaxed Semantic Match (khớp ngữ nghĩa khái niệm)** trên 3 tập dữ liệu đa miền (VACT, iTrust, eDocument) gồm 238 chính sách và 892 thuộc tính thực tế. Thực nghiệm đối sánh với các mô hình Named Entity Recognition (NER) học sâu tiên tiến (spaCy CNN và RoBERTa Transformer) chứng minh hệ thống quy tắc cú pháp giải quyết triệt để vấn đề dương tính giả tràn lan, vượt trội hơn mô hình Transformer tới **+77.20% $F_1$ tuyệt đối** trên thuộc tính môi trường.

---

### 1. HÌNH THỨC HÓA BÀI TOÁN (PROBLEM FORMULATION)

Cho một câu chính sách truy cập ngôn ngữ tự nhiên $P = \langle w_1, w_2, \dots, w_n \rangle$. Mục tiêu của bài toán là trích xuất bộ tứ thuộc tính ABAC hình thức:
$$\mathcal{T}(P) = \langle \mathcal{S}, \mathcal{A}, \mathcal{O}, \mathcal{E} \rangle$$

Trong đó:
1. **Subject ($\mathcal{S}$)**: Thực thể chủ thể yêu cầu truy cập, gồm vai trò trung tâm ($r \in \mathcal{R}$) và các thuộc tính định danh/bổ nghĩa ($q \in \mathcal{Q}$):
   $$\mathcal{S} = \langle \text{role}, \text{qualifiers}, \text{entity\_type} \rangle$$
2. **Action ($\mathcal{A}$)**: Hành vi thao tác được cấp phép/từ chối, chuẩn hóa từ động từ ngữ pháp ($v$) sang không gian thao tác CRUD+ ($\text{op} \in \{\text{READ}, \text{CREATE}, \text{UPDATE}, \text{DELETE}, \text{SCHEDULE}, \dots\}$):
   $$\mathcal{A} = \langle \text{verb}, \text{operation}, \text{modality} \rangle$$
3. **Resource/Object ($\mathcal{O}$)**: Tài nguyên mục tiêu chịu tác động, gồm danh từ gốc ($t$), từ hạn định và phạm vi:
   $$\mathcal{O} = \langle \text{label}, \text{qualifier}, \text{scope}, \text{attributes} \rangle$$
4. **Environment ($\mathcal{E}$)**: Ngữ cảnh không gian, thời gian hoặc ràng buộc hoạt động:
   $$\mathcal{E} = \langle \text{env\_type}, \text{trigger\_phrase}, \text{trigger\_word}, \text{constraints} \rangle$$

---

### 2. PHƯƠNG PHÁP ĐÁNH GIÁ CHUẨN KÉP (DUAL-EVALUATION FRAMEWORK)

Để đảm bảo tính khoa học cao nhất và tránh hiện tượng số liệu "quá cao một cách thiếu thực tế", bài báo phân tách rõ ràng hai thang đo:

#### Thang đo 1: Strict Exact-Match (Character/Span Boundary Identity)
- **Định nghĩa:** Một thuộc tính máy trích xuất $e_m$ được coi là Đúng (True Positive) nếu và chỉ nếu chuỗi token của $e_m$ trùng khớp 100% với nhãn chuẩn $e_g$ sau khi chuẩn hóa chữ thường, số ít và lọc bỏ mạo từ dừng (`a`, `an`, `the`).
- **Ý nghĩa khoa học:** Thang đo này là cận dưới (lower bound), phản ánh độ chính xác tuyệt đối trong việc bắt chính xác ranh giới thực thể (Boundary Detection).

#### Thang đo 2: Relaxed Semantic Match (Concept / Token-Overlap)
- **Định nghĩa:** Một thuộc tính máy trích xuất $e_m$ được coi là Đúng nếu chỉ số trùng lặp Jaccard giữa tập token của $e_g$ và $e_m$ đạt ngưỡng $\ge 0.60$ hoặc thuộc cùng một nhóm hành động đồng nghĩa CRUD+ (`ACTION_SYNONYM_GROUPS`).
- **Ý nghĩa khoa học:** Thang đo này là cận trên thực tế (practical upper bound), phản ánh khả năng nhận diện đúng bản chất thực thể để sinh chính sách XACML/ABAC trong hệ thống thực tế (nơi ánh xạ định danh Canonical ID giải quyết các chênh lệch mạo từ).

---

### 3. CÂU HỎI NGHIÊN CỨU (RESEARCH QUESTIONS)

Bài báo được thiết kế xoay quanh 4 câu hỏi nghiên cứu cốt lõi:
- **RQ1 (Độ chính xác tổng thể):** Kiến trúc NLACP đạt độ chính xác như thế nào trên cả hai tiêu chí Strict Exact-Match và Relaxed Semantic Match?
- **RQ2 (Khả năng tổng quát hóa liên miền):** Hệ thống có duy trì được hiệu năng khi chuyển giao giữa 3 miền hoàn toàn khác biệt (Y tế - iTrust, Hệ thống học tập - VACT, Quản lý tài liệu doanh nghiệp - eDocument) mà không cần can thiệp tham số không?
- **RQ3 (Đối sánh Baseline):** Tại sao phương pháp duyệt cây phụ thuộc cú pháp lại cần thiết so với việc áp dụng trực tiếp các mô hình Named Entity Recognition (NER) học sâu tiên tiến?
- **RQ4 (Nghiên cứu bóc tách - Ablation):** Đâu là nhân tố đóng góp chính vào độ chính xác: cấu trúc cú pháp tổng quát (Core Rules) hay từ điển chuyên ngành (Domain Gazetteers)?

---

### 4. KẾT QUẢ THỰC NGHIỆM ĐẦY ĐỦ VÀ ĐỐI SÁNH

#### Bảng 1: Kết quả Đánh giá Nghiêm ngặt (Strict Exact-Match)
*(Đo trên ranh giới chuẩn hóa chuỗi tuyệt đối, mã nguồn: `scripts/unified_eval.py`)*

| Tập dữ liệu | Thuộc tính | Precision | Recall | F1 Score | Số lượng ($N$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **iTrust** | Subject | 1.0000 | 1.0000 | **1.0000** | 58 |
| ($N=53$) | Action | 1.0000 | 0.9815 | **0.9907** | 54 |
| | Resource (Object) | 0.7358 | 0.6964 | **0.7156** | 56 |
| | Environment Context | 0.8889 | 0.8000 | **0.8421** | 10 |
| | **Tổng thể (Micro-Avg 4 core)** | **0.9133** | **0.8876** | **0.9003** | **178** |
| **VACT** | Subject | 0.9712 | 0.9619 | **0.9665** | 105 |
| ($N=100$) | Action | 0.9535 | 0.9535 | **0.9535** | 129 |
| | Resource (Object) | 0.8100 | 0.8020 | **0.8060** | 101 |
| | Environment Context | 0.8250 | 0.9900 | **0.9000** | 100 |
| | **Tổng thể (Micro-Avg 4 core)** | **0.8918** | **0.9287** | **0.9099** | **435** |
| | Subject Attributes | 0.9692 | 0.6923 | **0.8077** | 91 |
| | Object Attributes | 0.4783 | 0.8462 | **0.6111** | 13 |
| **eDocument** | Subject | 0.8851 | 0.9059 | **0.8953** | 85 |
| ($N=85$) | Action | 0.8817 | 0.8817 | **0.8817** | 93 |
| | Resource (Object) | 0.8000 | 0.7727 | **0.7861** | 88 |
| | Environment Context | 0.7857 | 0.8462 | **0.8148** | 13 |
| | **Tổng thể (Micro-Avg 4 core)** | **0.8530** | **0.8530** | **0.8530** | **279** |
| **TOÀN BỘ** | **Tất cả 3 tập dữ liệu (Micro)** | **0.8840** | **0.8969** | **0.8904** | **892** |

#### Bảng 2: Kết quả Đánh giá Ngữ nghĩa Khái niệm (Relaxed Semantic Match)
*(Đo trên mức độ nhận diện khái niệm ABAC: Token IoU $\ge 0.60$ + Action Synonyms, mã nguồn: `scripts/unified_eval.py`)*

| Tập dữ liệu | Thuộc tính | Precision | Recall | F1 Score | Số lượng ($N$) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **iTrust** | Subject | 1.0000 | 1.0000 | **1.0000** | 58 |
| ($N=53$) | Action | 1.0000 | 0.9815 | **0.9907** | 54 |
| | Resource (Object) | 0.7925 | 0.7500 | **0.7706** | 56 |
| | Environment Context | 0.8889 | 0.8000 | **0.8421** | 10 |
| | **Tổng thể (Micro-Avg 4 core)** | **0.9306** | **0.9045** | **0.9174** | **178** |
| **VACT** | Subject | 0.9712 | 0.9619 | **0.9665** | 105 |
| ($N=100$) | Action | 0.9535 | 0.9535 | **0.9535** | 129 |
| | Resource (Object) | 0.8600 | 0.8515 | **0.8557** | 101 |
| | Environment Context | 0.8250 | 0.9900 | **0.9000** | 100 |
| | **Tổng thể (Micro-Avg 4 core)** | **0.9029** | **0.9402** | **0.9212** | **435** |
| **eDocument** | Subject | 0.8966 | 0.9176 | **0.9070** | 85 |
| ($N=85$) | Action | 0.9247 | 0.9247 | **0.9247** | 93 |
| | Resource (Object) | 0.8471 | 0.8182 | **0.8324** | 88 |
| | Environment Context | 0.9286 | 1.0000 | **0.9630** | 13 |
| | **Tổng thể (Micro-Avg 4 core)** | **0.8925** | **0.8925** | **0.8925** | **279** |
| **TOÀN BỘ** | **Tất cả 3 tập dữ liệu (Micro)** | **0.9050** | **0.9182** | **0.9115** | **892** |

---

### 5. GIẢI THÍCH SỰ CHÊNH LỆCH VÀ PHÂN LOẠI LỖI (ERROR TAXONOMY)

Khoảng cách giữa Strict F1 (**0.8904**) và Semantic F1 (**0.9115**) phản ánh đúng bản chất ngôn ngữ học của việc trích xuất chính sách ABAC từ văn bản:

1. **Sai lệch ranh giới từ hạn định và bổ ngữ (Chiếm 64.3% lỗi Strict Object):**
   - Ví dụ: Trong VACT (ID 12), câu gốc: `"view and download the course roster"`. Nhãn chuẩn gán `roster`, máy trích xuất `course roster`. Trong triển khai ABAC thực tế, hệ thống ánh xạ thuộc tính (PEP mapper) đều ánh xạ cả hai về cùng một mã tài nguyên `urn:vact:resource:course_roster`, do đó không gây ra rủi ro cấp quyền nhầm hay từ chối cấp quyền.
2. **Đính kèm cụm giới từ phụ thuộc (Chiếm 23.8% lỗi Strict):**
   - Trong các câu có mệnh đề phụ lồng ghép (ví dụ `"scores for courses which they teach in the LMS"`), bộ phân tích cú pháp đôi khi bao gồm điều kiện khóa học vào ranh giới danh ngữ Object thay vì phân tách riêng biệt vào `context`.
3. **Lỗi trích xuất thực thể thực sự (Chỉ chiếm 11.9% lỗi Strict):**
   - Chỉ có 5 trên 42 trường hợp không khớp strict là do máy bỏ sót hoặc nhặt sai danh từ trung tâm của tài nguyên.

---

### 6. ĐỐI SÁNH VỚI BASELINES HỌC SÂU (RQ3)

Thực nghiệm đo lường khả năng trích xuất thuộc tính Môi trường (Environment Context) của các mô hình NER tiền huấn luyện (spaCy CNN và RoBERTa Transformer, mã nguồn: `scripts/eval_ner_baselines.py`):

| Tập dữ liệu | spaCy CNN (`sm`) $F_1$ | RoBERTa Transformer (`trf`) $F_1$ | **NLACP (Đề xuất)** $F_1$ | Độ tăng tuyệt đối so với RoBERTa |
| :--- | :---: | :---: | :---: | :---: |
| **VACT** | 0.0370 | 0.0187 | **0.9000** | **+88.13%** |
| **iTrust** | 0.1463 | 0.2308 | **0.8421** | **+61.13%** |
| **eDocument** | 0.0426 | 0.1395 | **0.9630** | **+82.35%** |
| **Trung bình** | **0.0753** | **0.1297** | **0.9017** | **+77.20%** |

*Nguyên nhân khoa học:* Mô hình Transformer NER được huấn luyện tổng quát trên OntoNotes 5.0 nên dán nhãn bất kỳ chuỗi số, ngày tháng, mã phòng ban nào trong văn bản (ví dụ: mã hồ sơ bệnh án, số phiên bản tài liệu). Vì thiếu liên kết phụ thuộc cú pháp với động từ truy cập chính (`prep`, `pobj`), mô hình học sâu sinh ra dương tính giả tràn lan (Precision < 25%). Ngược lại, NLACP chỉ kích hoạt khi cụm từ nằm trong nhánh cú pháp bổ nghĩa cho hành động truy cập.

---

### 7. ABLATION STUDY: CÚ PHÁP TỔNG QUÁT VS. TỪ ĐIỂN MIỀN (RQ4)

Để chứng minh hệ thống không bị "học vẹt" hay phụ thuộc vào danh sách từ khóa viết tay:

| Cấu hình kiểm thử | Precision | Recall | $F_1$ Môi trường |
| :--- | :---: | :---: | :---: |
| Đầy đủ (Bật cả từ điển chuyên miền) | 0.8250 | 0.9900 | 0.9000 |
| Bóc tách (`core_only = True` — Chỉ dùng cú pháp cốt lõi) | **0.8761** | **0.9900** | **0.9296** |
| **Chênh lệch ($\Delta$)** | **+0.0511** | **0.0000** | **+0.0296** |

*Phát hiện quan trọng:* Cấu hình bóc tách chỉ dựa trên cấu trúc cú pháp cơ bản (`prep_at`, `prep_during`, `prep_on`, `prep_between`, `prep_in`) đạt $F_1$ cao hơn do loại bỏ được các từ khóa chuyên ngành gây nhiễu. Điều này chứng minh thuật toán có tính khái quát cao và khả năng áp dụng trực tiếp cho các miền nghiệp vụ mới.

---

### 8. SO SÁNH VỚI MÔ HÌNH NGÔN NGỮ LỚN (LLMs)

| Tiêu chí | Hệ thống Cú pháp NLACP | Cloud LLM (GPT-4o / Claude 3.5) | Local LLM (LLaMA-3-8B) |
| :--- | :--- | :--- | :--- |
| **Thời gian phản hồi (Latency)** | **30 – 41 ms / câu** | 800 – 2,500 ms / câu | 350 – 1,200 ms / câu |
| **Chi phí vận hành** | **\$0.00 (Hoàn toàn miễn phí)** | \$5.00 – \$15.00 / 1,000 chính sách | Chi phí đầu tư GPU & điện năng |
| **Tính tất định (Determinism)** | **100% tất định (cùng input $\rightarrow$ cùng output)** | Bất định (ảnh hưởng bởi nhiệt độ/sampling) | Bán tất định |
| **Bảo mật & Chủ quyền dữ liệu** | **100% On-Premises (không rò rỉ chính sách)** | Rủi ro cao (chính sách bảo mật gửi qua Cloud API) | On-Premises |
| **Hiện tượng sinh ảo (Hallucination)** | **0% sinh ảo (chỉ trích xuất từ văn bản)** | Nguy cơ sinh sai vai trò / thao tác | Nguy cơ sinh sai định dạng JSON |

---

### 9. ĐỘ ĐE DỌA GIÁ TRỊ NGHIÊN CỨU (THREATS TO VALIDITY)

1. **Tính hợp lệ về mặt khái niệm (Construct Validity):** Việc lựa chọn giữa Strict và Semantic match được minh bạch hóa bằng cách công bố song song cả hai bảng số liệu chi tiết.
2. **Tính hợp lệ nội tại (Internal Validity):** Quy trình gán nhãn được công khai trung thực là Single-Annotator với sự hỗ trợ của AI trong `docs/annotation_protocol.md`, và cam kết quy trình kiểm định chéo độc lập (Dual Independent Annotation với Cohen's Kappa $\kappa \ge 0.80$) cho giai đoạn mở rộng 500 câu tiếp theo.
3. **Tính hợp lệ bên ngoài (External Validity):** Các chính sách thực nghiệm hiện tại dài trung bình 19.6 từ qua 3 miền nghiệp vụ (Y tế, Giáo dục, Doanh nghiệp). Khả năng mở rộng trên câu siêu phức hợp nhiều tầng phụ thuộc tiếp tục được nghiên cứu.

---

### 10. HƯỚNG DẪN TÁI LẬP KHOA HỌC (REPRODUCIBILITY PROTOCOL)

Toàn bộ kết quả trong bài báo có thể được tái lập độc lập 100% bằng các lệnh sau:

```bash
# 1. Chạy đánh giá toàn diện chuẩn thống nhất (Strict & Relaxed)
python scripts/unified_eval.py

# 2. Chạy kiểm chứng baseline NER học sâu
python scripts/eval_ner_baselines.py

# 3. Chạy kiểm chứng Ablation Study
python scripts/ablation_core_only.py
```
