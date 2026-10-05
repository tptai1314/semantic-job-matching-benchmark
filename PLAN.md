# Kế hoạch thực hiện (14 tuần) — bản v3, đã đối chiếu `docs/plan_review.md`

> Bản này sửa các lỗi mức 🔴/🟠 và 16 mâu thuẫn nêu trong `docs/plan_review.md` (ngày 2026-10-05, số liệu đo thật trên `cnamuangtoun/resume-job-description-fit`).
> `research_log.md` được **cập nhật theo mốc**, mỗi lần ghi kèm ngày.

---

## Phần 0 — Nội dung khoá trước khi chạy thí nghiệm

### 0.1 Ba câu hỏi nghiên cứu (RQ)

| RQ | Câu hỏi | Phương pháp trả lời |
|---|---|---|
| **RQ1** | Trích xuất view có cấu trúc bằng LLM có cải thiện xếp hạng CV–JD so với SBERT trên văn bản toàn văn không? | B2 vs B3 (LLM-rút gọn) vs B3h (rút gọn heuristic, **không LLM**) vs multi-view static |
| **RQ2** | Fusion học được có vượt được fusion tĩnh tối ưu không? | static-đều vs static-tối-ưu (val) vs B7 concat-MLP vs adaptive gating (val) |
| **RQ3** *(tuỳ chọn)* | Phương pháp có **non-inferior** với cross-encoder fine-tune (B6) ở mức δ đã khoá, trong khi chi phí xếp hạng thấp hơn nhiều? | TOST một phía trên NDCG@K, kèm bảng chi phí |

### 0.2 Bốn giả thuyết và tiêu chí bác bỏ

| Giả thuyết | Phát biểu | Tiêu chí **bác bỏ** (kết luận trung thực) |
|---|---|---|
| **H1** | Trích xuất LLM tạo ra lợi ích vượt mức lọc nhiễu thuần tuý | `B3 − B3h < SESOI` → **H1 bị bác bỏ**; bài chỉ được nói: *"rút gọn văn bản giúp ích, LLM không thêm gì"* |
| **H2** | Multi-view theo cặp view đánh bại SBERT toàn văn | `multi-view_static − B2 < SESOI` **hoặc** CI 95% chồng 0 → **H2 bị bác bỏ** |
| **H3** | Fusion học được đánh bại fusion tĩnh tối ưu | `max(adaptive, B7) − static_tối_ưu < SESOI` → **H3 bị bác bỏ**, kết luận về gating là phủ định |
| **H4** | Non-inferiority so với B6 | Cận dưới CI 95% của `(đề xuất − B6)` **≤ −δ** → **H4 bị bác bỏ**; claim dời sang ablation + diễn giải |

Quy tắc chung: **không kết luận "vượt trội" nếu CI 95% chồng 0.** Mọi phát biểu định tính phải kèm số và CI.

### 0.3 Tuyên bố đóng góp (dự thảo — kiểm tra trùng lặp ở bước 1.3 trước khi chốt)

1. **Benchmark + bộ baseline chuẩn hoá** cho ghép CV–JD: 8.000 cặp / 351 JD / 643 CV, 8 phương pháp B0–B8, chạy bằng **một hàm đánh giá chung** và một giao thức split công khai.
2. **Leakage audit định lượng**: báo cáo tường minh mức rò rỉ CV của chính benchmark này (99,8% CV test có trong train) và **báo cáo song song** split theo `job_id` và split theo `resume_id`.
3. **Tách biệt hiệu quả của LLM khỏi hiệu quả của lọc nhiễu** bằng baseline rút gọn heuristic (B3h) — điểm mà các công trình multi-view hiện có thiếu.
4. **Diễn giải fusion**: trọng số gating theo view và theo nhãn, đối chiếu với mức giảm NDCG khi loại bỏ từng view (tương quan hạng Spearman).
5. **Đánh giá công khai chất lượng trích xuất LLM** trên *toàn bộ* 994 văn bản của benchmark, kèm gold schema test 100 văn bản.

### 0.4 Quyết định khoá sớm (pre-registration)

| # | Quyết định | Lý do |
|---|---|---|
| 1 | **Gộp toàn bộ 8.000 dòng** (train 6.241 + test 1.759), rồi **GroupKFold 5-fold theo `job_id`** | Split chính thức chỉ có 71 JD ở test → power quá thấp. Gộp cho 351 JD |
| 2 | **Metric chính: NDCG@5**, relevance 0/1/2. Báo thêm NDCG@3, NDCG@10 | Review: NDCG@10 có σ nhỏ nhất nhưng chỉ còn 175 JD; NDCG@5 giữ 256 JD |
| 3 | **P@K/R@K chính dùng relevant B (Good+Potential)**; A (chỉ Good) chỉ báo phụ kèm `n_jd` | A chỉ đo được trên 143/351 JD (41%) |
| 4 | **JD không có CV relevant → loại khỏi mọi metric**, ghi `n_jd` trong từng bảng | Tránh NDCG=0 gây nhiễu |
| 5 | **Hòa điểm: expected NDCG** cho *tất cả* metric (không `ignore_ties`) | Review 3.6: tie rất nhiều khi tín hiệu bị zero hoá và với TF-IDF thưa |
| 6 | **Split chính: theo `job_id`.** Split phụ: theo `resume_id`** (CV-disjoint). Báo cáo cả hai | Review 2.B: rò rỉ CV là 99,8%, không phải giả thuyết |
| 7 | **Chỉ chọn mô hình theo validation.** Test đánh giá đúng một lần, theo `docs/test_protocol.md` khoá trước tuần 10 | Tránh rò rỉ test |
| 8 | **Kiểm định:** paired bootstrap (cluster theo `job_id`) + Wilcoxon signed-rank, **hiệu chỉnh Holm**, CI 95%, effect size (Cohen's d_z và Cliff's δ) | Quy mô nhiều so sánh |
| 9 | **SESOI và δ lấy từ MDE đo được**, không dùng con số 0.01 | Review 2.A: 0.01 bất khả thi (MDE ≈ 0.023–0.046) |
| 10 | **Bảng kính định lượng chất lượng trích xuất** là cổng chặn trước mọi kết luận về phương pháp | Review 3.3 |
| 11 | **Số seed: 5** cho mọi mô hình học được (B4, B6, B7, adaptive). Tối thiểu 3 nếu trễ | Bỏ mâu thuẫn 3/4/5 seed giữa 4.5, 5.1, config |
| 12 | **Chuẩn hoá 9 tín hiệu trước fusion** (z-score trên train) | Cosine ∈ [−1,1] còn logits cross-encoder không cùng thang |

### 0.5 Ma trận rủi ro

| Rủi ro | Xác suất | Ảnh hưởng | Phương án |
|---|---|---|---|
| Chất lượng trích xuất LLM thấp (view rỗng, bịa nội dung) | Trung bình | Rất cao — hỏng toàn bộ phương pháp | Cổng chặn định lượng (0.4 #10); sửa prompt, chạy lại từ 6.1 |
| Không có GPU → B4/B6 không chạy nổi | **Cao (đã xảy ra)** | Cao — mất 2 baseline mạnh nhất | Thang phương án ở §0.6; luôn giữ B5 (cross-encoder pretrained) |
| Phương pháp không vượt B2 | Trung bình | Cao | Nhánh "Multi-view ≤ B2" ở bước 9.1; claim dời sang lọc nhiễu + diễn giải |
| Không tìm được người chấm thứ hai | Trung bình | Trung bình | Ghi rõ Limitations, hạ mức khẳng định |
| Dataset bị gỡ khỏi HF | Thấp | Rất cao | `docs/data_license.md` + ghi hash/mirror nội bộ ngay tuần 2 |
| Power không đủ để kết luận "=" | **Cao (thiết kế)** | Trung bình | Báo cáo CI + MDE; kết luận "không phân biệt được" thay vì "bằng nhau" |
| CV leak làm B4/B6 bị thổi phồng | **Đã xảy ra (99,8%)** | Cao | Split phụ theo `resume_id`; bảng leakage audit |

### 0.6 Ngân sách tính toán và thang phương án

Hiện trạng: `torch 2.14.0+cpu`, `cuda_available: false` (máy có RTX 3050 4GB nhưng chưa cài wheel CUDA).

| Việc | CPU hiện tại | Ghi chú |
|---|---|---|
| TF-IDF (B1), BM25 | vài phút | OK |
| SBERT toàn văn (B2), B3, B3h | ~10–20 phút | OK |
| B5 cross-encoder pretrained | ~1–3 giờ (chấm mọi cặp) | OK |
| Embed 994 văn bản + view | ~10 phút | OK |
| **Trích xuất LLM 994 văn bản** | **~10–20 giờ** (Ollama, 1 tiến trình) | Xem bước 2.6; có thể song song 2–3 tiến trình |
| **B4 / B6 fine-tune × 5 seed × 5 fold** | **không khả thi trên CPU** | Cần một trong các phương án dưới |

Thang phương án (chọn theo thứ tự, ghi quyết định vào log):

1. **Cài torch CUDA** cho RTX 3050 4GB (~3 GB, khả thi vì mô hình nhỏ: MiniLM 22M tham số).
2. **Colab Pro / Kaggle GPU** — miễn phí, ưu tiên cao nếu không có GPU local.
3. **Giảm quy mô**: B4/B6 chỉ trên **fold 1 × 5 seed** + **fold 2–5 × 1 seed**; báo rõ đây là thiết kế thiếu power.
4. **Bỏ fine-tune, dùng pretrained**: B2 + B5 + B7/adaptive (fusion nhỏ, train nhanh trên CPU). Nói rõ baseline mạnh nhất là B5, không phải B6.

### 0.7 Về ngôn ngữ và sản phẩm

Dataset và toàn bộ mô hình hiện **tiếng Anh**, trong khi sản phẩm Frevia nhắm người dùng Việt Nam. Quyết định: **đặt tên bài là nghiên cứu phương pháp trên benchmark tiếng Anh**; chuyển giao sang tiếng Việt nêu ở *Limitations + hướng phát triển*. Nếu còn thời gian (sau tuần 11), chạy một thí nghiệm nhỏ tiếng Việt (20–50 cặp tự thu thập, chỉ zero-shot) làm bằng chứng khả thi.

---

## TUẦN 1 – Khảo sát và khóa thiết kế

**Bước 1.1 — Giao thức tìm kiếm tài liệu (PRISMA mini)**
- Cơ sở: Google Scholar, arXiv, ACL Anthology, Semantic Scholar.
- Từ khoá: *person-job fit*, *resume job matching*, *multi-view resume matching*, *LLM resume parsing*, *cross-encoder re-ranking recruitment*, *LLM-as-a-judge hiring*.
- Tiêu chí include: 2019–2026, có mô hình hoặc benchmark đo được, báo cáo trên dữ liệu CV–JD công khai hoặc tương tự.
- Tiêu chí exclude: chỉ survey, không đo trên CV thật, trùng bản cập nhật.
- **Ghi ngày tìm, số kết quả, số bài giữ lại. Nhắm 25–35 tài liệu**, bắt buộc có bài 2024–2026 về LLM resume matching/judge.
- Tải PDF về `docs/papers/`.

**Bước 1.2 — Đọc và lập bảng related work**
- Đọc kỹ: PJFNN, ConFit, ConFit v2, CareerBERT, Sentence-BERT, Passage Re-ranking with BERT.
- Bắt buộc thêm: các bài 2024–2026 về LLM matching/judge, và **BERT-like encoder mạnh hơn** (BGE-M3, E5, gte) vì lợi ích multi-view có thể biến mất với encoder mạnh.
- Lập bảng: phương pháp | dữ liệu | metric | điểm yếu | liên hệ đề tài → `docs/related_work.md`.

**Bước 1.3 — Kiểm tra trùng lặp và chốt claim**
- Tìm công trình có đủ 3 thành phần: LLM trích xuất cấu trúc + so khớp theo cặp view + fusion học được.
- **Chuẩn bị sẵn 3 phương án tái định vị** nếu trùng: (a) nhấn vào **đóng góp benchmark + đo chất lượng trích xuất**; (b) nhấn vào **tách LLM khỏi lọc nhiễu** (H1); (c) nhấn vào **diễn giải fusion + đánh giá non-inferiority**.
- Chốt tuyên bố đóng góp ở §0.3 dựa trên kết quả này.

**Bước 1.4 — Viết `docs/research_log.md` (lần 1, có ngày)**
- 3 RQ và 4 giả thuyết + tiêu chí bác bỏ (chép từ §0.1–0.2, không viết lại từ đầu).
- Metric chính/phụ, quy tắc chọn mô hình (chỉ theo val), kế hoạch kiểm định (paired bootstrap + Wilcoxon, Holm, CI 95%, effect size).
- Quy tắc JD không có CV relevant (loại khỏi mọi metric) và hòa điểm (expected NDCG cho cả P/R/MRR).
- Danh sách 12 quyết định khoá sớm ở §0.4.
- **Chưa ghi SESOI/δ** (chờ MDE thật ở 3.7) — ghi rõ điều đó.

**Bước 1.5 — Chốt danh sách baseline**
- B0 Random · B1 TF-IDF · **B3h Heuristic reduced (không LLM)** · B2 SBERT toàn văn · B3 SBERT LLM-reduced · B4 bi-encoder fine-tune · B5 cross-encoder pretrained · B6 cross-encoder fine-tune · B7 concat-MLP · **B8 LLM zero-shot / LLM-as-judge**.
- Ghi lý do chọn vào log. Ghi rõ **chi tiết backbone/loss của B4/B6 sẽ chốt ở tuần 4 nhưng phải viết ra tiêu chí chọn loss ngay tuần này** (xem bước 4.3).

**Bước 1.6 — Xác định nơi nộp và định dạng**
- Xác định hội nghị/tạp chí mục tiêu, giới hạn trang, template, ngôn ngữ, citation style, tỉ lệ trùng lặp, mốc đăng ký.
- Đặt **ngày bắt đầu thật** và **hạn nộp**; dựng lịch ngược với 1 tuần đệm → `docs/schedule.md`.

**Bước 1.7 — Liên hệ tác giả dataset về license**
- Dataset chính **không có dataset card và không khai báo license** → không thể "kiểm tra license" như một bước đóng. Ghi `docs/data_license.md`: chỉ dùng cho nghiên cứu, không phân phối lại văn bản, không trích nguyên văn CV/JD trong bài, kèm ngày và nội dung trao đổi với tác giả.

✅ **Xong tuần 1 khi:** bảng related work (≥25 tài liệu) xong, log lần 1 có ngày, claim đã chốt, nơi nộp đã xác định, đã gửi email xin xác nhận license.

---

## TUẦN 2 – Dựng dự án + dataset + EDA + chia fold

**Bước 2.1 — Dựng cấu trúc thư mục**
```
configs/
data/{raw,processed,splits}/
cache/{views,embeddings}/
src/{data,extraction,representation,matching,fusion,training,baselines,evaluation}/
experiments/
tests/
notebooks/
results/
scripts/
docs/{papers,research_log.md,data_license.md,related_work.md,limitations.md}
```

**Bước 2.2 — Cố định môi trường**
- `requirements.txt` ghim phiên bản cụ thể (đã có) + **ghim `revision` của dataset và model HF** trong config.
- `set_seed()` trong `src/utils.py` (đã có).
- Khởi tạo git, commit đầu tiên; remote `origin` trỏ GitHub.
- Ghi `results/logs/env_info.json` (torch, CUDA, GPU, phiên bản).

**Bước 2.3 — Tải dataset chính**
- `cnamuangtoun/resume-job-description-fit`: 6.241 + 1.759 = **8.000 dòng**, 351 JD, 643 CV.
- **Không có dataset card, không có license** → ghi đây là hạn chế về nguồn gốc nhãn vào log (không thể khẳng định nhãn phản ánh đánh giá của chuyên gia tuyển dụng).

**Bước 2.4 — Ghi `docs/data_license.md`**
- Nội dung như bước 1.7, cập nhật ngày. Nếu có phản hồi từ tác giả → ghi lại.

**Bước 2.5 — Làm sạch dữ liệu**
- Gán `job_id`, `resume_id` bằng hash văn bản (sha1, 16 ký tự).
- Loại cặp trùng (đo được: 7 cặp trùng trong train).
- Báo cáo cặp có nhãn mâu thuẫn (đo được: **6 cặp** trong train) → loại và ghi số lượng.
- Ghi ra `data/processed/pairs.parquet` + báo cáo số liệu.

**Bước 2.6 — Cài Ollama, khoá cấu hình gọi model, và đo thời gian**
- `ollama pull qwen3:4b`. **Ghi ngay `model digest`** vào config (dùng cho cache key) — không đợi đến tuần 14.
- **Cấu hình bắt buộc** (review 3.4): `think: false`, `format: json`, `seed`, `keep_alive`, và **assert `num_ctx` thực sự được truyền** (client hay bỏ qua option này).
- **Nâng `num_ctx` 8192 → 16384**: CV dài nhất ~3.134 từ ≈ 4.200 token, cộng prompt hệ thống + schema JSON là sát trần 8192.
- Chạy thử trên **5–10 văn bản lấy từ dataset vừa tải**, đo thời gian từng văn bản.
- **Ước tính đúng phạm vi**: chỉ có **994 văn bản riêng biệt** (643 CV + 351 JD), không phải 8.000 vì cache theo văn bản. Ngân sách: **~10–20 giờ CPU**, có thể chạy 2–3 tiến trình song song.
- **Cache key = hash(văn bản) + model digest + hash(prompt) + params** để không trộn kết quả giữa các lần đổi cấu hình.

**Bước 2.7 — EDA**
Chạy `notebooks/01_eda.ipynb` (đã có số liệu sơ bộ trong `scripts/quick_eda.py`, cần chuyển thành notebook và bổ sung):
- Số cặp, số JD (351), số CV (643).
- Số CV mỗi JD: min 1 / p25 7 / **median 14,5** / p75 29,3 / max 111.
- Phân bố nhãn ~50% No Fit / 25% Potential / 25% Good.
- **JD không có Good Fit: 165/280 (58,9%)**; JD không có relevant nào: 46/351 (13,1%).
- **JD có ≥ K ứng viên**: K=3: 274 · K=5: 256 · K=10: 175.
- Độ dài token CV/JD và **tỉ lệ vượt giới hạn MiniLM ở CẢ CV và JD** (CV 72,7%, JD 27,0%) — không chỉ CV.

**Bước 2.8 — Ánh xạ nhãn và định nghĩa relevant (chốt, không để lần 3.7)**
- No = 0, Potential = 1, Good = 2. **NDCG dùng relevance 0/1/2.**
- **P@K/R@K chính dùng B (Good+Potential)**; A (chỉ Good) báo phụ kèm `n_jd` (chỉ 143/351 JD = 41%).
- Cập nhật `configs/base.yaml` cho khớp (đã đặt `primary: lenient`).

**Bước 2.9 — Chốt K và chiến lược chia fold**
- **Quy tắc chốt K bằng số liệu, không bằng quy tắc "median < 15"** (median thật = 14,5, rơi đúng vào lằn ranh):
  - **K chính = 5** (giữ 256 JD).
  - Báo thêm K=3 và K=10 **kèm `n_jd`**; với K=10 ghi rõ chỉ 175 JD và σ nhỏ nhất (power cao nhất) — nêu đây là điểm cân nhắc nếu ưu tiên power.
- **Gộp 8.000 dòng + GroupKFold 5-fold theo `job_id`**. Split chính thức (6.241/1.759) chỉ dùng làm **kiểm tra phụ** — ghi lý do (power: 71 JD vs 351 JD).
- **Tạo cả hai hệ split**: `data/splits/fold_{i}.json` (theo `job_id`) và `data/splits/fold_cvdisjoint_{i}.json` (theo `resume_id`).
- Kiểm tra: mỗi `job_id` chỉ thuộc một fold; các cặp không rỗng.

**Bước 2.10 — Leakage audit định lượng (bắt buộc, in ra bảng)**
- In 3 con số cho **từng cặp fold**: số `job_id` chung, số `resume_id` chung, số cặp (JD, CV) chung.
- Số liệu đo được trước: **JD chung giữa train/test = 0**, **CV chung = 476/477 (99,8%)**, cặp chung = 0.
- Với split theo `resume_id`: báo cáo số CV dùng chung (kỳ vọng = 0) và số JD bị mất.
- Lưu `results/tables/leakage_audit.csv`.
- **Ghi kết luận vào log**: đây là bài toán **re-ranking trong pool ứng viên dùng chung**, không phải retrieval toàn corpus. B4/B6 (fine-tune) hưởng lợi từ rò rỉ này nhiều hơn phương pháp không fine-tune → **báo cáo cả hai split như phân tích độ bền**, và coi kết quả "B4/B6 tụt mạnh ở split CV-disjoint còn multi-view không tụt" là phát hiện tích cực.

**Bước 2.11 — Cập nhật `research_log.md` (lần 2, có ngày)**
- Ghi: K và lý do chốt bằng số liệu; chiến lược hai split; kết quả leakage audit; quyết định chấp nhận rò rỉ CV (đã chốt ở §0.4 #6); con số 994 văn bản và ngân sách trích xuất.

✅ **Xong tuần 2 khi:** dataset sạch, **hai hệ fold** cố định, leakage audit có bảng, EDA xong, log lần 2, digest Ollama đã ghi.

---

## TUẦN 3 – Evaluation + baseline lexical + đo MDE

**Bước 3.1 — Viết `src/evaluation/metrics.py`**
- P@K, R@K, MRR, NDCG@K theo từng JD rồi lấy trung bình macro.
- **JD không có CV relevant → loại khỏi mọi metric**; trả về kèm `n_jd`.
- **Hòa điểm: expected NDCG cho CẢ P@K, R@K, MRR** (không chỉ NDCG) — `sklearn.metrics.ndcg_score(ignore_ties=False)` chính là expected NDCG, dùng được.
- **Quy tắc JD có < K ứng viên**: vẫn tính nhưng **luôn báo `n_jd` theo từng K**; phân tích chính trên tập JD có ≥ K ứng viên.

**Bước 3.2 — Viết `tests/test_metrics.py`**
- Ví dụ tính tay: hoàn hảo, đảo ngược, **hòa điểm** (bắt buộc có test riêng cho tie vì tie xuất hiện rất nhiều khi tín hiệu zero hoá).
- Đối chiếu NDCG với `sklearn.metrics.ndcg_score`.

**Bước 3.3 — Viết `src/evaluation/significance.py`**
- **Cluster paired bootstrap theo `job_id`** (bắt buộc, không bootstrap phẳng), Wilcoxon signed-rank, CI 95%, effect size (Cohen's d_z, Cliff's δ), hiệu chỉnh Holm.

**Bước 3.4 — Viết `evaluate()` chung**
- Trả về metric tổng hợp + **mảng điểm từng JD** + `n_jd`. Mọi phương pháp sau gọi đúng hàm này.

**Bước 3.5 — Chạy B0 (random ranking)**
- 5 fold, `results/baselines/b0.csv`. **B0 phải thấp hơn B1 rõ rệt** — nếu không, metric có bug, dừng lại sửa.

**Bước 3.6 — Chạy B1 (TF-IDF + Cosine)**
- Fit trên train, tune `ngram_range` và `max_features` trên val, chạy 5 fold, `results/baselines/b1.csv`.
- **BM25 (tuỳ chọn, rẻ):** chạy cùng lượt để không bỏ trùng `baselines.bm25` trong config. Nếu bỏ thì **xoá khối config đó**.

**Bước 3.7 — Đo MDE và chốt SESOI (deliverable bắt buộc, thay cho con số 0.01 cảm tính)**
- Tính **σ_d thật**: std của chênh lệch NDCG *theo từng JD*, paired, **gộp các fold sao cho mỗi JD xuất hiện đúng một lần**.
- In **MDE** (80% power, hai phía) cho NDCG@3/@5/@10; in cả hai kịch bản σ_d = σ_jd và σ_d = 0,5·σ_jd.
- **SESOI = 3 × MDE** (dự kiến 0,05–0,07), hoặc lấy từ literature (khoảng cách giữa các phương pháp trong ConFit/PJFNN thường 0,03–0,08 NDCG). Ghi nguồn.
- Ghi rõ: **0,01 là ngưỡng *bỏ qua* (negligible), KHÔNG phải ngưỡng *ý nghĩa*** — không dùng nó làm SESOI.
- **Định nghĩa đủ 3 ô của bảng quyết định**: `Δ ≥ SESOI và CI không chồng 0` → ">"; `Δ < SESOI và CI chồng lấn` → "≈"; `Δ < SESOI nhưng CI không chồng 0` → "thấp hơn nhưng không thực tiễn" (ô thứ ba, plan cũ thiếu).
- Cập nhật `research_log.md` (**lần 3**) và `configs/base.yaml` (`evaluation.sesoi`).

✅ **Xong tuần 3 khi:** metrics test pass, B0 và B1 có kết quả, **MDE và SESOI đã đo và ghi vào log**.

---

## TUẦN 4 – Baseline SBERT, cross-encoder, và chốt δ

**Bước 4.1 — Chạy B2 (SBERT toàn văn zero-shot)**
- Cache embedding vào `cache/embeddings/b2/`.
- **Báo tỉ lệ mẫu bị cắt ở CẢ CV và JD** (JD test có 39% vượt 512 từ).
- Chunking: `chunk_mean` 4×384 = 1.536 token, p90 CV ≈ 1.500 token → **sát ngưỡng**. Chạy thêm cấu hình `max_chunks: 8` như một ablation nhỏ.

**Bước 4.2 — Chạy B5 (cross-encoder pretrained MS MARCO MiniLM)**
- Không fine-tune, chấm mọi cặp. **Đây là baseline mạnh nhất nếu không có GPU** — ghi rõ trong log.

**Bước 4.3 — Viết pipeline fine-tune chung `src/training/finetune.py` (B4 và B6)**
- Early stopping theo NDCG@K trên val.
- **Chốt loss ngay, không để tới cuối tuần** (review 3.6 — yếu tố quyết định độ mạnh của baseline):
  - **B4 bi-encoder**: chọn 1 trong MarginMSE / MSE trên nhãn 0–1–2 / InfoNCE với hard negative trong cùng JD. Ghi tiêu chí chọn.
  - **B6 cross-encoder**: regression, ordinal regression hay 3-class CE — nêu rõ và lý do.
- Cách lấy điểm xếp hạng cho từng phương pháp (cosine chuẩn hoá hay raw logit) — ghi rõ.
- Có `--max-folds`, `--seeds` để chạy theo thang phương án §0.6 khi không có GPU.

**Bước 4.4 — Chạy B4 (bi-encoder fine-tune)**
- 3 seed × 5 fold (lên 5 seed nếu đủ tài nguyên), ghi thời gian train/inference.

**Bước 4.5 — Chạy B6 (cross-encoder fine-tune)**
- **5 seed trên fold 1** để đo std giữa seed.

**Bước 4.6 — Chốt biên δ (ghi log lần 4 TRƯỚC khi chạy test)**
- δ lấy từ **SESOI đã đo ở 3.7**, không dùng `max(1 std, 0.01)`.
- **Điều kiện khả thi bắt buộc**: nếu δ < 2 × (nửa chiều rộng CI 95%) thì ghi rõ **non-inferiority sẽ inconclusive với cỡ mẫu này** và thay bằng báo cáo hiệu số + CI. Đây là điều kiện đã biết trước, ghi vào log để tránh tự dệ ra kết luận âm sau này.
- Ghi δ kèm ngày vào `research_log.md` và `configs/base.yaml` (`evaluation.noninferiority.delta`).

**Bước 4.7 — Ghi `results/baselines_meta.csv`**
- Số tham số, thời gian train, thời gian inference của **mọi** baseline B0–B8, trên cùng phần cứng, ghi rõ phần cứng.

✅ **Xong tuần 4 khi:** B2, B4, B5, B6 có kết quả, **δ đã khoá**, log lần 4.

---

## TUẦN 5 – Hoàn tất baseline + trích xuất view

**Bước 5.1 — Chạy B6 đầy đủ**
- 5 seed × 5 fold nếu đủ tài nguyên; nếu không, 3 seed × 5 fold + fold 1 đã có 5 seed. Ghi rõ cấu hình thực tế đã chạy. `results/baselines/b6.csv`.

**Bước 5.2 — Viết `src/extraction/`**
- `schema.py`: 7 view CV, 5 view JD, **9 cặp so khớp = 7 cặp lõi + 2 cặp chéo**.
- `prompt.py`: prompt JSON cố định, có `prompt_version`.
- `validator.py`: kiểm tra JSON hợp lệ, đủ view, không rỗng; thử lại N lần; ghi cờ lỗi.
- `extractor.py`: gọi Ollama với cấu hình đã khoá ở 2.6 (`think: false`, `format=json`, `num_ctx=16384`), **cache key = hash(văn bản) + model digest + hash(prompt) + params**.

**Bước 5.3 — Ghi lý do thiết kế từng cặp view vào log (lần 5)**
- Đặc biệt **2 cặp chéo** — reviewer chắc chắn hỏi. Nêu rõ vì sao chéo giữa hai miền ngữ nghĩa.

**Bước 5.4 — Chỉnh prompt**
- Chỉ dùng **train của fold 1**, không nhìn val/test.
- **Ghi rõ hệ quả**: prompt được tune trên dữ liệu fold 1 → lạc quan cho fold 1. Hoặc tune trên từng fold (đắt), hoặc **cố định prompt và báo riêng kết quả fold 1** như một kiểm tra. Ghi lựa chọn vào log.

**Bước 5.5 — Chạy trích xuất thử 100 mẫu**
- Kiểm tra chất lượng, tỉ lệ view rỗng, tỉ lệ lỗi sau thử lại.
- Chạy trên **tập con hard** nếu đã tạo được (bước 12.1 nói về tập hard; nếu chưa, chỉ báo cáo tổng).
- Nếu chất lượng kém → sửa prompt, chạy lại.

**Bước 5.6 — Viết `src/baselines/heuristic.py` (B3h)**
- Rút gọn **không dùng LLM**: tách mục bằng heading regex + cắt giới hạn token + chỉ lấy phần Skills/Requirements.
- **Đây là baseline kiểm soát then chốt cho H1** — phải xong trước tuần 6 vì B3 (LLM-rút gọn) chỉ có ý nghĩa khi so với B3h.

✅ **Xong tuần 5 khi:** prompt chốt, 100 mẫu ổn, B6 xong, **B3h chạy được**.

---

## TUẦN 6 – Trích xuất toàn bộ + cổng chất lượng

**Bước 6.1 — Chạy trích xuất toàn bộ 994 văn bản**
- Cache `cache/views/<prompt_version>/<model_digest>/`.
- Ghi thời gian từng văn bản (dùng cho H4 ở tuần 11).
- Chạy song song 2–3 tiến trình; ngân sách ~10–20 giờ.

**Bước 6.2 — Cổng chất lượng: tiêu chí định lượng (BẮT BUỘC, chặn)**
Điều kiện đi qua cổng, ghi vào log:
- **≥ 95% JSON hợp lệ** sau tối đa 3 lần thử.
- **≤ 10% view rỗng** trên mỗi view.
- Không có hiện tượng bịa nội dung rõ rệt (kiểm bằng mẫu).
**Không đạt → dừng, sửa prompt, xóa cache, chạy lại từ 6.1.** Không được đi tiếp với view hỏng.

**Bước 6.3 — Đánh giá chất lượng trích xuất (bắt buộc, nâng từ "nếu kịp")**
- Tỉ lệ view rỗng theo từng view; tỉ lệ lỗi sau thử lại; độ dài view trung bình.
- **Gold schema test**: 100 văn bản có view chuẩn do người viết → đo **precision/recall/F1** của trích xuất (không chỉ chấm 0/1/2).
- **Chấm tay toàn bộ 994 văn bản** nếu khả thi (chỉ 994, ≈1–2 ngày công) — đây là điểm mạnh publish được. Nếu không đủ thời gian thì **tối thiểu 200 mẫu** và ghi rõ CI.
- **≥ 2 người chấm độc lập** trên một mẫu con → báo **Cohen's κ / Krippendorff α**. Nếu không có người thứ hai: ghi rõ trong `docs/limitations.md` **và hạ mức khẳng định** của mọi kết luận.
- Chạy lại 30 mẫu lần hai (cache riêng) → đo độ ổn định, **không dùng lại cache cũ**.

**Bước 6.4 — Chạy B3 (SBERT trên văn bản rút gọn) và B3h (heuristic)**
- **B3 phụ thuộc 6.1** — chỉ chạy sau khi có view.
- Nối các view thành một đoạn (không tách view), cache embedding riêng, chạy 5 fold.
- **Chạy B3h cùng lượt** để có cặp so sánh (B3 − B3h) ngay từ đầu.

**Bước 6.5 — Ghi `results/extraction_quality.md` và cập nhật log (lần 6)**

✅ **Xong tuần 6 khi:** cổng 6.2 **đạt**, báo cáo chất lượng xong, B3 + B3h có kết quả.

---

## TUẦN 7 – Embedding + tín hiệu + fusion static

**Bước 7.1 — Embed từng view bằng `all-MiniLM-L6-v2`**
- Cache `cache/embeddings/views/`. View rỗng → vector 0 + mask.
- Báo tỉ lệ view vượt giới hạn token.

**Bước 7.2 — Viết `src/matching/signals.py`**
- $s^{(k)}_{ij} = \cos(e^{JD}_{i,a_k}, e^{CV}_{j,b_k})$, k = 1..9. Vectorized, shape (n_jobs, n_resumes, 9).
- Kiểm tra: không NaN, giá trị trong [−1, 1], shape khớp fold.

**Bước 7.3 — Kiểm tra chất lượng tín hiệu (mở rộng so với plan cũ)**
- Tương quan mỗi tín hiệu với nhãn trên **train** → tín hiệu gần như không tương quan thì xem lại cặp view.
- **Ma trận tương quan giữa 9 tín hiệu**: nếu nhiều cặp > 0,9 thì "fusion học được" gần như không thể thắng trung bình cộng và **H3 sẽ bị bác bỏ một cách máy móc**. Phải đo **trước** khi đầu tư vào adaptive.
- **Chuẩn hoá tín hiệu trước fusion**: z-score trên train (hoặc rank-based), lưu lại tham số chuẩn hoá để áp dụng nhất quán cho val/test.
- **Kiểm tra mask có thành đặc trưng rò rỉ không**: "view rỗng" có thể tương quan với độ dài/chất lượng CV, mà độ dài lại tương quan với nhãn. Ghi kết quả, chuẩn bị ablation "bỏ mask".

**Bước 7.4 — Static fusion trọng số đều**
- $S_{ij} = \frac{1}{9}\sum_k s^{(k)}_{ij}$. Chạy 5 fold, cả hai split (job_id và CV-disjoint).

**Bước 7.5 — Static fusion tối ưu**
- Grid search trọng số **trên val**, ghi mọi cấu hình vào `results/hparam_log.csv`.
- Có số mốc tĩnh tối ưu để không so sánh với một static yếu.

✅ **Xong tuần 7 khi:** tensor tín hiệu có (kiểm tra đa cộng tuyến + chuẩn hoá xong), static đều và static tối ưu có kết quả trên cả hai split.

---

## TUẦN 8 – Adaptive fusion + concat-MLP + BPR

**Bước 8.1 — Viết `src/fusion/concat_mlp.py` (B7)**
- Input: 9 tín hiệu đã chuẩn hoá + mask. MLP ra điểm trực tiếp, không weighted sum.

**Bước 8.2 — Viết `src/fusion/adaptive.py`**
- MLP nhận vector tín hiệu + mask, softmax ra trọng số $w_{ij,k}$.
- $S_{ij} = \sum_k w_{ij,k}\, s^{(k)}_{ij}$.

**Bước 8.3 — Viết `src/training/pairs.py`**
- Dựng cặp BPR trong cùng JD của train: $j^+$ có relevance cao hơn $j^-$.
- Assert: không JD nào của val/test lọt vào tập cặp. Ghi số cặp theo fold.

**Bước 8.4 — Viết vòng huấn luyện BPR**
- $\mathcal{L} = -\sum_i \sum_{(j^+,j^-)} \log \sigma(S_{ij^+} - S_{ij^-})$.
- AdamW, early stopping theo NDCG@K trên val. Lưu checkpoint tốt nhất **và** trọng số gating để diễn giải.

**Bước 8.5 — Tune lưới nhỏ trên val**
- Ngân sách tương đương baseline. Ghi mọi cấu hình vào `results/hparam_log.csv`.

**Bước 8.6 — Chạy B7, adaptive, static tối ưu trên fold 1**
- So sánh trên val, chuẩn bị dữ liệu cho cổng tuần 9.

**Bước 8.7 — Cross-attention view-level (tuỳ chọn, là mục cắt số 1)**
- Cùng quy trình, cùng cách chọn checkpoint; số tham số phải cùng cấp so sánh với gating, ghi số tham số vào bảng.
- **Chạy sau cùng một tuần**: nếu không đủ thời gian thì bỏ, và **xoá khối `fusion.attention` mồ côi khỏi config**.

**Bước 8.8 — Viết nháp Method**
- Bắt đầu viết Method và Experimental Setup khi thiết kế đã chốt.

✅ **Xong tuần 8 khi:** B7 và adaptive chạy được trên fold 1, nháp Method bắt đầu, quyết định về cross-attention đã ghi.

---

## TUẦN 9 – Cổng quyết định + chạy full fold

**Bước 9.1 — Họp cổng quyết định**

Dùng **tiêu chí định lượng đã chốt ở 3.7** (SESOI từ MDE, ba ô: ">" / "≈" / "thấp hơn nhưng không thực tiễn"):

| Kết quả trên val | Hành động |
|---|---|
| Multi-view > B2 **và** > B3 | Đi tiếp RQ1, RQ2 |
| Multi-view > B2 nhưng ≈ B3 | Đổi hướng: lợi ích từ lọc nhiễu, kiểm tra B3h |
| **B3 ≈ B3h** | **H1 bị bác bỏ** → claim dời sang "rút gọn văn bản", LLM là phần phụ |
| Multi-view ≤ B2 | Kiểm tra lại chất lượng trích xuất, cặp view, cặp BPR |
| Adaptive ≈ static tối ưu **hoặc** ≈ B7 | **H3 bị bác bỏ** |
| Kém B6 rõ rệt | Đổi claim sang chi phí + diễn giải |

⚠️ **Cảnh báo đã biết trước**: cổng dùng chính val đã dùng để tune → **lạc quan có hệ thống**. Ghi rõ đây là ước lượng lạc quan; nếu có thể, dùng val của fold chưa dùng để tune (nested). Việc này ghi vào log trước khi cổng chạy, không ghi sau.

**Bước 9.2 — Ghi quyết định vào `research_log.md` (lần 7) kèm ngày**

**Bước 9.3 — Chạy full 5 fold cho các biến thể được chọn**
- 5 seed (3 seed nếu trễ). Ghi `results/fusion_runs.csv`. **Chạy trên cả hai split.**

**Bước 9.4 — Kiểm tra ổn định**
- Trọng số gating có nhất quán giữa seed không.

✅ **Xong tuần 9 khi:** cổng đã chốt, full fold chạy xong trên hai split.

---

## TUẦN 10 – Thực nghiệm chính (test lần đầu)

**Bước 10.0 — Khoá `docs/test_protocol.md` (TRƯỚC khi chạy test)**
- Liệt kê **toàn bộ** cấu hình sẽ chạy trên test: phương pháp × split × seed × metric.
- **Mọi thứ phát sinh sau bước này chỉ được chạy trên val.** Kể cả ablation và phân tích của tuần 11–12.
- Danh sách này được ký bằng commit hash.

**Bước 10.1 — Khóa siêu tham số và checkpoint theo val**
- Không đổi gì sau bước này.

**Bước 10.2 — Đánh giá test một lần**
- Chạy đúng danh sách ở 10.0. Lưu `results/main_runs.csv`.

**Bước 10.3 — Kiểm định thống kê**
- So đề xuất với từng baseline bằng NDCG từng JD, **gộp các fold sao cho mỗi JD xuất hiện đúng một lần**.
- Cluster paired bootstrap theo `job_id` + Wilcoxon signed-rank, **hiệu chỉnh Holm**, CI 95%, effect size.

**Bước 10.4 — Đóng băng bảng chính**
- mean ± std qua seed và fold, **kèm `n_jd` cho từng ô**. Đánh dấu mức ý nghĩa.
- Bảng phải gồm **cả hai split** (job_id và CV-disjoint).

✅ **Xong tuần 10 khi:** `docs/test_protocol.md` đã khoá, bảng chính đóng băng, kiểm định xong.

---

## TUẦN 11 – RQ3 + chi phí + độ nhạy hàm nhãn

**Bước 11.1 — Non-inferiority với B6 (đã sửa đúng hướng và dấu)**
- Dùng **δ đã khoá ở 4.6**.
- Công thức đúng: **cận dưới CI 95% của `(đề xuất − B6)` so với `−δ`**.
- Dùng **kiểm định một phía kiểu TOST**, không chỉ so CI.
- Nếu điều kiện khả thi ở 4.6 không thoả (δ < 2 × nửa CI) → báo **inconclusive**, không kết luận.

**Bước 11.2 — Đo chi phí suy luận**
- Cùng phần cứng, ghi rõ phần cứng.
- Tách: **chi phí trích xuất một lần** (đã đo ở 6.1) + **chi phí xếp hạng mỗi JD** (chỉ phần fusion/encoder, không tính trích xuất lại).
- So với B5 (và B6 nếu có). `results/cost.csv`.

**Bước 11.3 — Độ nhạy với hàm nhãn khác (đã đổi tên, KHÔNG gọi là "chuyển miền")**
- `med2425/resume-job-fit-merged-v1` **gộp chính dataset A** và **dán nhãn lại bằng Qwen2.5-32B** → thí nghiệm này đo **sự đổi hàm nhãn**, không phải dịch chuyển miền.
- Lọc bằng **cột `source`** để loại dòng có nguồn gốc từ A (không chỉ dedupe hash), rồi mới đo.
- **Phương án thay thế tốt hơn (chọn nếu kịp): dùng dataset B làm nguồn mở rộng số JD để tăng power** — chỉ chạy được với baseline không-LLM (B0–B2, B3h), vì trích xuất LLM 93.733 dòng bất khả thi trên CPU.
- Ghi rõ kết luận nào đúng với thiết kế đã chọy.

**Bước 11.4 — Độ nhạy LLM (tuỳ chọn)**
- Chạy lại trích xuất với LLM thứ hai, **cache riêng theo digest + prompt hash**, báo độ lệch NDCG.

✅ **Xong tuần 11 khi:** chi phí đo xong, non-inferiority có kết luận **hoặc** kết luận "inconclusive" nêu rõ lý do, các mục tuỳ chọn có hoặc bỏ rõ lý do.

---

## TUẦN 12 – Ablation và phân tích

**Bước 12.1 — Tập con "hard" và ablation RQ1**
- Dùng `resume_domain` / `jd_domain` của `med2425`, hoặc tự gán domain bằng LLM, để **giới hạn pool ứng viên trong cùng ngành** → tín hiệu từ vựng biến mất, chỉ còn khác biệt tinh tế.
- Báo NDCG trên **cả "easy pool" và "hard pool"**.
- Thêm: NDCG của B1 theo từng mức nhãn, và NDCG trên tập con **Potential-vs-Good** (bỏ No Fit) → đo khả năng phân biệt tinh tế.
- Đây là **đóng góp benchmark có thể publish**.

**Bước 12.2 — Ablation RQ1**
- B2 vs B3h vs B3 vs multi-view static. **Đây là cặp quyết định H1.**

**Bước 12.3 — Ablation RQ2**
- Static đều, static tối ưu, B7 concat-MLP, adaptive (+ cross-attention nếu đã chạy).

**Bước 12.4 — Ablation thành phần tín hiệu**
- 7 cặp lõi · 9 cặp đầy đủ · bỏ từng view (leave-one-out) · một tín hiệu đơn.
- **Thêm: bỏ mask** (kiểm tra mask có phải đặc trưng rò rỉ không — chuẩn bị ở 7.3).
- Mỗi cấu hình chạy **cùng số seed**.

**Bước 12.5 — Ablation backbone (BẮT BUỘC, nâng từ optional)**
- Thử **ít nhất 1 encoder mạnh hơn MiniLM-L6** (BGE-M3 / E5 / gte).
- Lý do: lợi ích multi-view có thể **biến mất** khi encoder mạnh hơn; bỏ qua thì reviewer chắc chắn hỏi.
- Thêm: cross-encoder trên văn bản view đã nối, để tách "cần kiến trúc multi-view" khỏi "chỉ cần cross-encoder mạnh hơn + văn bản sạch".

**Bước 12.6 — Diễn giải trọng số (log lần 8)**
- Trọng số gating trung bình theo view và theo nhãn; độ nhất quán giữa seed.
- **Tương quan hạng Spearman** giữa trọng số view và mức giảm NDCG khi bỏ view.
- 3–5 ví dụ định tính: một cặp được xếp cao, view nào đóng góp nhiều nhất.

**Bước 12.7 — Phân tích lỗi**
- 10–15 JD có NDCG thấp nhất. Phân loại nguyên nhân: trích xuất sai, view thiếu, nhãn mơ hồ, CV quá ngắn hoặc nhiễu. Bảng đếm kèm ví dụ.

**Bước 12.8 — Phân tích khác**
- NDCG theo độ dài CV (ngắn/vừa/dài).
- Che tên và thông tin định danh trong CV rồi chạy lại. Nếu không làm được, ghi hạn chế.

✅ **Xong tuần 12 khi:** ablation (RQ1, RQ2, thành phần, backbone) xong, diễn giải, phân tích lỗi xong.

---

## TUẦN 13 – Viết bài

**Bước 13.1 — Hoàn thiện Method và Experimental Setup**
- Nháp đã viết từ tuần 8; cập nhật theo kết quả cuối. Lấy từ `research_log.md`.

**Bước 13.2 — Viết Results và Analysis**
- **Mọi số lấy từ file kết quả bằng script, không gõ tay.** Mỗi bảng kèm `n_jd`.

**Bước 13.3 — Viết Related Work** (từ `docs/related_work.md`, ≥25 tài liệu, có bài 2024–2026)

**Bước 13.4 — Viết Introduction và Abstract**
- Động cơ, khoảng trống, RQ, **tuyên bố đóng góp** (từ §0.3 đã chốt ở 1.3).

**Bước 13.5 — Viết Limitations, Ethics, Conclusion**
- Limitations lấy từ `docs/limitations.md` (đã viết dần từ tuần 3).
- **Ethics/PII:** CV là dữ liệu cá nhân → chính sách lưu trữ, ẩn danh, **không đưa nguyên văn CV/JD vào bài hay phụ lục**. Ghi nguồn + license của mọi model dùng (MiniLM, MS MARCO MiniLM, Qwen3, BGE/E5 nếu dùng).

**Bước 13.6 — Viết `scripts/make_figures.py`**
- Sinh **mọi** hình và bảng từ file kết quả. Bao gồm: đường cong khớp lợi ích–chi phí, heatmap trọng số gating, leakage audit, đường cong MDE, so sánh easy/hard pool.

**Bước 13.7 — Chuẩn bị bảo vệ**
- Slide 12–15 trang từ `scripts/make_figures.py`.
- Chuẩn bị trả lời **15 câu hỏi phản biện dự kiến**, gồm: vì sao không dùng cross-encoder fine-tune; trích xuất LLM sai thì sao; vì sao BPR; kết quả bền qua seed không; dataset có đại diện thực tế không; **tại sao claim 0.01 là bất khả thi**; **rò rỉ CV 99,8% xử lý thế nào**; nguồn gốc nhãn không rõ thì sao.

✅ **Xong tuần 13 khi:** bản thảo đầy đủ, slide xong, câu hỏi phản biện đã chuẩn bị.

---

## TUẦN 14 – Tái lập, phản biện, nộp

**Bước 14.1 — Viết README chạy từ đầu**
- Config, seed, danh sách fold (cả hai split), hash commit, **digest Ollama**, **revision dataset/model HF**.

**Bước 14.2 — Chạy lại một thí nghiệm từ môi trường sạch**
- Kiểm tra khớp kết quả; nếu lệch → ghi nguyên nhân.
- Bổ sung **script một lệnh chạy lại toàn bộ** và Dockerfile/environment.yml (review 5.12).

**Bước 14.3 — Phản biện nội bộ**
- Nhờ giảng viên hướng dẫn đọc như reviewer; nhờ một người ngoài dự án đọc. Ghi lại và xử lý từng nhận xét.

**Bước 14.4 — Chỉnh sửa theo phản biện**

**Bước 14.5 — Kiểm tra định dạng và deadline nơi nộp**
- Xem trực tiếp trên trang hội nghị/tạp chí. **Không dùng danh sách trích dẫn chưa xác minh.**

**Bước 14.6 — Tổng duyệt bảo vệ**

**Bước 14.7 — Nộp**

✅ **Xong tuần 14 khi:** bản thảo nộp, gói tái lập xong, phản biện xử lý xong.

---

## Cập nhật `research_log.md` theo mốc

| Lần | Bước | Tuần | Nội dung |
|---|---|---|---|
| 1 | 1.4 | 1 | RQ, 4 giả thuyết + tiêu chí bác bỏ, metric, quy tắc chọn mô hình, 12 quyết định khoá sớm |
| 2 | 2.11 | 2 | K và lý do chốt bằng số liệu, hai chiến lược split, kết quả leakage audit, 994 văn bản, ngân sách trích xuất |
| 3 | 3.7 | 3 | **σ_d, MDE, SESOI**, định nghĩa 3 ô bảng quyết định |
| 4 | 4.6 | 4 | **Biên δ** + điều kiện khả thi của non-inferiority (trước khi chạy test) |
| 5 | 5.3 | 5 | Lý do thiết kế từng cặp view, đặc biệt 2 cặp chéo |
| 6 | 6.5 | 6 | Kết quả cổng chất lượng trích xuất, kết quả F1 trên gold schema, κ/α |
| 7 | 9.2 | 9 | Quyết định cổng, kết luận H1/H2/H3 |
| 8 | 12.6 | 12 | Diễn giải trọng số, tương quan Spearman, kết luận H4 |

---

## Thứ tự cắt nếu trễ

1. Cross-attention view-level (bước 8.7, tuỳ chọn — xoá luôn khối `fusion.attention` khỏi config nếu bỏ).
2. LLM trích xuất thứ hai (bước 11.4).
3. Dataset thứ hai / độ nhạy hàm nhãn (bước 11.3).
4. Ablation backbone (bước 12.5) — **cắt được, nhưng phải ghi rõ trong Limitations rằng không kiểm chứng trên encoder mạng hơn**.
5. Bớt seed 5 → 3.
6. BM25, LLM zero-shot.

**Không được cắt dù đến hạn:** B0, B1, B2, B3h, B3, B4, B6, B7, static tối ưu, adaptive, **cổng chất lượng trích xuất (6.2)**, leakage audit, MDE/SESOI, bảng chính, kiểm định thống kê, bảng chi phí.

> Lưu ý: **B8 (LLM zero-shot / LLM-as-judge)** chỉ cần chạy trên một mẫu test vài nghìn cặp để có upper bound, nên **giữ lại trong danh sách chính** — đây là câu hỏi số 1 của hội đồng 2025–2026.

---

## Việc làm ngay hôm nay

1. ~~Tạo repo git, dựng cấu trúc thư mục~~ — xong, commit `c37eb43`.
2. ~~Viết `requirements.txt` và `set_seed()`~~ — xong.
3. Tải dataset chính, đọc dataset card và license (Bước 2.3, 2.4). **Lưu ý: không có card, không có license — dùng kết quả đo sẵn trong `docs/plan_review.md`.**
4. Cài Ollama, khoá `think=false` + `num_ctx=16384`, ghi digest, test trên mẫu từ dataset (Bước 2.6).
5. Chạy `notebooks/01_eda.ipynb` (chuyển `scripts/quick_eda.py` sang notebook) — Bước 2.7.
6. Viết `src/data/` load + clean, chạy **GroupKFold 5-fold theo `job_id`** và **split CV-disjoint theo `resume_id`**, in bảng leakage audit (Bước 2.9–2.10).

Sau khi có kết quả leakage audit, chốt K và cập nhật `research_log.md` lần 2.
