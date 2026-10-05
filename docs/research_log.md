# Nhật ký nghiên cứu

Ghi lại theo từng mốc, không sửa lại mốc cũ. Mỗi mục ghi: việc đã làm, kết quả đo
được, quyết định đã chốt, và cái gì còn treo.

---

## 2026-10-05 — Mốc 1: hạ tầng, dữ liệu và audit rò rỉ

### Đã làm

- Dựng cấu trúc `src/` (10 module), `experiments/run_frevia.py`, `configs/base.yaml`.
- Cài thư viện trên Python 3.13.3 global. `torch 2.14.0+cpu`, `torch.cuda.is_available() == False`.
- Viết lại `PLAN.md` v3 theo `docs/plan_review.md`.
- Xây pipeline dữ liệu: `src/data/{load,clean,schemas,ids,folds,pipeline}.py`.
- Chạy `python experiments/run_frevia.py dataset` và `folds`.

### Kết quả đo được

**Dữ liệu (sau khi làm sạch):**

| Chỉ số | Giá trị |
|---|---|
| Dòng gốc | 8.000 (train 6.241 + test 1.759) |
| Dòng sau làm sạch | 7.993 |
| Cặp trùng bị loại | 7 |
| Văn bản rỗng / quá ngắn | 0 |
| Cặp có nhãn mâu thuẫn | 0 |
| CV / JD | 643 / 351 |

`job_id` và `resume_id` **không có sẵn** trong dataset, phải sinh từ hash nội dung
văn bản (`sha1`, 16 hex). Id vì vậy ổn định và không phụ thuộc thứ tự dòng.

**Audit rò rỉ (GroupKFold 5 fold trên 7.993 cặp):**

| Hệ split | Nhóm dùng để chia | Nhóm rò giữa train/val | Ghi chú |
|---|---|---|---|
| `job_id` | 351 JD | **JD 0**, **CV 452–504 (98,0–99,0% CV val)** | sạch JD, rò CV |
| `resume_id` | 643 CV | **CV 0**, **JD 299–317** | sạch CV, rò JD |

Không hệ split nào sạch cả hai chiều, vì mỗi CV xuất hiện ở nhiều JD. Số dòng mỗi
fold cân bằng tốt (1.598–1.599).

### Quyết định đã chốt

1. **Gộp train + test rồi chia fold mới** (7.993 cặp), vì split chính thức có
   476/477 CV test nằm trong train → 99,8% rò rỉ.
2. **Báo cáo hai hệ split** ở mọi bảng kết quả, không chọn một và giấu cái kia.
3. `data_license.md`: dataset **không có license và không có card**, chỉ dùng cho
   nghiên cứu, không phân phối lại, không in nguyên văn CV/JD.

### Còn treo

- Chưa ghim `data.dataset_revision`.
- Chưa liên hệ tác giả xin license.
- Ollama chưa cài server, `qwen3:4b` chưa pull → Bước 2.6 chưa chạy.
- Chưa đo MDE (Bước 3.7); `SESOI` và `δ` non-inferiority vẫn để `null`.