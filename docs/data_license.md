# Tình trạng pháp lý và nguồn dữ liệu

Cập nhật: 2026-10-05 · Bước PLAN 1.7 và 2.4

## 1. Dataset chính

| Mục | Nội dung |
|---|---|
| Tên | `cnamuangtoun/resume-job-description-fit` |
| Nguồn | HuggingFace Hub |
| Revision | **chưa ghim** (`data.dataset_revision: null`) — phải ghim trước khi chạy thí nghiệm chính thức |
| Số cặp | 8.000 (train 6.241 + test 1.759) |
| Số CV / JD | 643 CV, 351 JD, tổng 994 văn bản riêng biệt |
| Nhãn | `No Fit` 4.000 / `Potential Fit` 2.000 / `Good Fit` 2.000 (đúng 50/25/25) |
| Dataset card | **không có** |
| License | **không khai báo** |

## 2. Rủi ro đã biết

1. **Không có license.** Chưa xác minh được điều khoản sử dụng lại. Văn bản là CV và JD
   thật của người dùng, có thể chứa dữ liệu cá nhân.
2. **Không có dataset card.** Không rõ quy trình thu thập, nguồn gốc CV/JD, ai đã
   gán nhãn và theo tiêu chí nào. Vì vậy `label` phải được coi là **nhãn ủy quyền
   yếu**, không phải ground truth tuyệt đối.
3. **Chưa ghim revision.** Tải lại sau này có thể khác dữ liệu.

## 3. Quyết định đã áp dụng

- Chỉ dùng cho **mục đích nghiên cứu học thuật**, không phân phối lại dataset.
- **Không** commit văn bản thô vào repo. `.gitignore` đã loại `data/raw/`,
  `data/processed/`, `data/splits/`, `cache/`, `results/`.
- **Không** in nguyên văn CV/JD vào bảng, log hay `docs/research_log.md`.
  Mọi ví dụ trích dẫn phải cắt bớt hoặc ẩn danh.
- Bảng kết quả chỉ dùng số đếm và chỉ số tổng hợp.
- Ghi `revision` thật vào `docs/research_log.md` ngay khi chạy lần đầu.

## 4. Việc còn phải làm

- [ ] Liên hệ tác giả dataset để hỏi license và quy trình gán nhãn.
- [ ] Ghim `data.dataset_revision` sau khi xác nhận.
- [ ] Ghi nguồn gốc của 994 văn bản (từ đâu, khi nào, có sự đồng ý không).
- [ ] Nếu không được phép phân phối lại: chỉ công bố mã và script, không kèm dữ liệu.

## 5. Dataset phụ

`med2425/resume-job-fit-merged-v1` (93.733 dòng) được dùng ở Bước 11.3.

Lưu ý: dataset này **chứa chính dataset trên** và đã được dán nhãn lại bằng
Qwen2.5-32B. Vì vậy nó đo **độ nhạy với hàm nhãn khác**, KHÔNG phải dịch chuyển miền,
và không được dùng làm nguồn dữ liệu chính. Cần kiểm tra cột `source` trước khi so sánh.
License của dataset này cũng chưa xác minh.