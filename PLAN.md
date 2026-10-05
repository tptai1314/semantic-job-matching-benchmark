# Kế hoạch thực hiện từng bước (14 tuần) — bản đã sửa

Dưới đây là **các bước làm cụ thể theo thứ tự thời gian**. Mỗi bước ghi rõ: làm gì → sản phẩm → xong khi nào.

**Lưu ý chung:** `research_log.md` được **cập nhật theo mốc**, không viết một lần. Log ghi rõ ngày mỗi lần cập nhật.

---

## TUẦN 1 – Khảo sát và khóa thiết kế

**Bước 1.1 — Tìm tài liệu**
- Lên Google Scholar, arXiv, ACL Anthology.
- Từ khoá: *person-job fit*, *resume job matching*, *multi-view resume matching*, *LLM resume parsing*, *cross-encoder re-ranking recruitment*.
- Tải PDF về `docs/papers/`.

**Bước 1.2 — Đọc và lập bảng related work**
- Đọc kỹ: PJFNN, ConFit, ConFit v2, CareerBERT, Sentence-BERT, Passage Re-ranking with BERT.
- Lập bảng: phương pháp | dữ liệu | metric | điểm yếu | liên hệ đề tài.

**Bước 1.3 — Kiểm tra trùng lặp**
- Tìm công trình có đủ 3 thành phần: LLM trích xuất cấu trúc + so khớp theo cặp view + fusion học được.
- Nếu có công trình rất gần → điều chỉnh claim trước khi làm tiếp.

**Bước 1.4 — Viết `docs/research_log.md` (lần 1)**
- Ghi 3 RQ, 4 giả thuyết, tiêu chí bác bỏ từng giả thuyết.
- Ghi metric chính (NDCG@K relevance 0/1/2), metric phụ (P@K, R@K, MRR).
- Ghi quy tắc chọn mô hình: chỉ theo validation.
- Ghi kế hoạch kiểm định: paired bootstrap/Wilcoxon, Holm, CI 95%, effect size.
- Ghi quy tắc xử lý JD không có CV relevant (loại khỏi NDCG) và hòa điểm (expected NDCG).
- **Chưa ghi K, chưa ghi δ** — sẽ cập nhật ở tuần 2 và tuần 4.
- Ghi ngày.

**Bước 1.5 — Danh sách baseline dự kiến (chưa chốt chi tiết)**
- B0 Random, B1 TF-IDF, B2 SBERT toàn văn, B3 SBERT trên văn bản rút gọn, B4 bi-encoder fine-tune, B5 cross-encoder pretrained, B6 cross-encoder fine-tune, B7 concat-MLP.
- Chi tiết backbone, loss, cách lấy điểm sẽ chốt ở tuần 4.
- Ghi lý do chọn vào log.

✅ **Xong tuần 1 khi:** bảng related work xong, log lần 1 có ngày, claim đã chốt.

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
docs/
```

**Bước 2.2 — Cố định môi trường**
- Viết `requirements.txt` với phiên bản cụ thể.
- Viết `set_seed()` trong `src/utils.py`.
- Khởi tạo git, commit đầu tiên.

**Bước 2.3 — Tải dataset chính**
- `cnamuangtoun/resume-job-description-fit`.
- Đọc dataset card, ghi cách tạo nhãn và hạn chế vào log.

**Bước 2.4 — Kiểm tra license dataset**
- Kiểm tra license, điều khoản sử dụng, có được phép công bố ví dụ không.
- Ghi vào `docs/data_license.md`.

**Bước 2.5 — Làm sạch dữ liệu**
- Gán `job_id`, `resume_id` bằng hash văn bản.
- Loại cặp trùng.
- Báo cáo cặp có nhãn mâu thuẫn.

**Bước 2.6 — Cài Ollama và test Qwen3:4b**
- Chạy thử trên 5–10 văn bản mẫu **lấy từ dataset vừa tải**.
- Đo thời gian xử lý mỗi văn bản.
- Ước tính tổng thời gian trích xuất toàn bộ dataset → ghi vào log.

**Bước 2.7 — EDA**
Chạy `notebooks/01_eda.ipynb`, ghi ra:
- Số cặp, số JD riêng biệt, số CV riêng biệt.
- Số CV mỗi JD (median, min, max).
- Phân bố nhãn, số JD không có Good Fit.
- Độ dài token CV/JD.

**Bước 2.8 — Ánh xạ nhãn**
- No = 0, Potential = 1, Good = 2.
- Định nghĩa relevant A (chỉ Good), B (Good + Potential).

**Bước 2.9 — Quyết định K và chiến lược chia fold**
- K theo median số CV mỗi JD. Nếu median < 15 → K = 3, 5.
- Dùng GroupKFold 5-fold theo `job_id`.
- Lưu vào `data/splits/fold_*.json`.

**Bước 2.10 — Kiểm tra rò rỉ**
- Kiểm tra JD không trùng giữa train/val/test.
- Kiểm tra CV không trùng giữa train/val/test.
- Kiểm tra cặp (JD, CV) không trùng.
- Nếu CV xuất hiện ở cả train và test với JD khác nhau → quyết định chấp nhận hay chia theo cả `resume_id`, ghi vào log.

**Bước 2.11 — Cập nhật `research_log.md` (lần 2)**
- Ghi K, chiến lược fold, quyết định xử lý rò rỉ CV.
- Ghi ngày.

✅ **Xong tuần 2 khi:** dataset sạch, fold cố định, không rò rỉ, EDA xong, log cập nhật lần 2.

---

## TUẦN 3 – Evaluation + baseline lexical

**Bước 3.1 — Viết `evaluation/metrics.py`**
- P@K, R@K, MRR, NDCG@K theo từng JD rồi lấy trung bình.
- Quy tắc: JD không có CV relevant → loại khỏi NDCG.
- Hòa điểm → dùng expected NDCG (không `ignore_ties`).

**Bước 3.2 — Viết `tests/test_metrics.py`**
- Ví dụ tính tay: hoàn hảo, đảo ngược, hòa điểm.
- Đối chiếu NDCG với `sklearn.metrics.ndcg_score`.

**Bước 3.3 — Viết `evaluation/significance.py`**
- Paired bootstrap, Wilcoxon, CI 95%, effect size, hiệu chỉnh Holm.

**Bước 3.4 — Viết `evaluate()` chung**
- Trả về metric tổng hợp và mảng điểm từng JD.

**Bước 3.5 — Chạy B0 (random ranking)**
- Chạy 5 fold, ghi `results/baselines/b0.csv`.
- Kiểm tra metric có hoạt động không: B0 phải thấp hơn B1 rõ rệt.

**Bước 3.6 — Chạy B1 (TF-IDF + Cosine)**
- Fit trên train, tune ngram và max_features trên val.
- Chạy 5 fold, ghi `results/baselines/b1.csv`.

**Bước 3.7 — Chốt tiêu chí định lượng cho cổng quyết định**
- Ghi vào log: chênh lệch ≥ 0.01 NDCG coi là ">" thực tiễn; < 0.01 và CI chồng lấn coi là "≈".
- Kiểm định bằng paired bootstrap trên điểm từng JD.

✅ **Xong tuần 3 khi:** metrics test pass, B0 và B1 có kết quả, tiêu chí cổng quyết định đã ghi.

---

## TUẦN 4 – Baseline SBERT và cross-encoder

**Bước 4.1 — Chạy B2 (SBERT toàn văn zero-shot)**
- Chiến lược cắt/chunk cố định.
- Cache embedding vào `cache/embeddings/b2/`.
- Báo cáo tỉ lệ mẫu bị cắt.

**Bước 4.2 — Chạy B5 (cross-encoder pretrained MS MARCO MiniLM)**
- Không fine-tune, chạy trực tiếp.

**Bước 4.3 — Viết pipeline fine-tune chung**
- `src/training/finetune.py` dùng cho B4 và B6.
- Early stopping theo NDCG@K trên val.
- Chốt chi tiết: backbone, loss, cách lấy điểm xếp hạng.

**Bước 4.4 — Chạy B4 (bi-encoder fine-tune)**
- 3 seed, 5 fold.
- Ghi thời gian train/inference.

**Bước 4.5 — Chạy B6 (cross-encoder fine-tune)**
- **5 seed trên fold 1** để đo std giữa seed.
- Tính std NDCG@K giữa seed.

**Bước 4.6 — Chốt biên δ**
- δ = max(1 std, 0.01 NDCG).
- Ghi δ vào `research_log.md` kèm ngày **trước khi chạy test**.

**Bước 4.7 — Ghi `results/baselines_meta.csv`**
- Số tham số, thời gian train, thời gian inference của mọi baseline.

✅ **Xong tuần 4 khi:** B2, B4, B5, B6 có kết quả, δ đã ghi, log cập nhật lần 3.

---

## TUẦN 5 – Hoàn tất baseline + bắt đầu trích xuất view

**Bước 5.1 — Chạy B6 đầy đủ (3 seed × 5 fold)**
- 3 seed để tiết kiệm thời gian; fold 1 đã có 5 seed từ tuần 4.
- Ghi `results/baselines/b6.csv`.

**Bước 5.2 — Viết `src/extraction/`**
- `schema.py`: định nghĩa 7 view CV, 5 view JD, **9 cặp so khớp = 7 cặp lõi + 2 cặp chéo**.
- `prompt.py`: prompt JSON cố định.
- `validator.py`: kiểm tra JSON hợp lệ, thử lại N lần.
- `extractor.py`: gọi Ollama, cache theo `resume_id`/`job_id`.

**Bước 5.3 — Ghi lý do thiết kế từng cặp view vào log**
- Đặc biệt 2 cặp chéo — reviewer sẽ hỏi.

**Bước 5.4 — Chỉnh prompt trên train**
- Chỉ dùng train của fold 1.
- Không nhìn val/test.
- Cố định `prompt_version` trong config.

**Bước 5.5 — Chạy trích xuất thử 100 mẫu**
- Kiểm tra chất lượng, tỉ lệ view rỗng, tỉ lệ lỗi sau thử lại.
- Nếu chất lượng kém → sửa prompt, chạy lại.

✅ **Xong tuần 5 khi:** prompt chốt, trích xuất thử 100 mẫu ổn, B6 full xong.

---

## TUẦN 6 – Trích xuất toàn bộ + đánh giá chất lượng

**Bước 6.1 — Chạy trích xuất toàn bộ dataset chính**
- Cache vào `cache/views/<prompt_version>/`.
- Ghi thời gian xử lý mỗi văn bản (dùng cho H4).
- Chạy qua đêm nếu cần.

**Bước 6.2 — Đánh giá chất lượng trích xuất (bắt buộc)**
- Tỉ lệ view rỗng theo từng view.
- Tỉ lệ lỗi sau thử lại.
- Chạy lại 30 mẫu → đo độ ổn định.

**Bước 6.3 — Đánh giá chất lượng trích xuất (nếu kịp)**
- Chấm tay 50 mẫu (thang 0/1/2).
- Nhờ người thứ hai chấm một phần → ước lượng độ đồng thuận.
- Nếu không có người thứ hai → ghi vào hạn chế.

**Bước 6.4 — Chạy B3 (SBERT trên văn bản rút gọn)**
- **Phụ thuộc Bước 6.1** — chỉ chạy sau khi có view.
- Nối các view thành một đoạn, không tách view.
- Cache embedding riêng.
- Chạy 5 fold.

**Bước 6.5 — Ghi báo cáo chất lượng trích xuất**
- `results/extraction_quality.md`.

✅ **Xong tuần 6 khi:** mọi CV/JD có view, B3 có kết quả, báo cáo chất lượng xong.

---

## TUẦN 7 – Embedding + tín hiệu + fusion static

**Bước 7.1 — Embed từng view bằng `all-MiniLM-L6-v2`**
- Cache vào `cache/embeddings/views/`.
- View rỗng → vector 0 + mask.
- Kiểm tra view vượt giới hạn token, báo tỉ lệ.

**Bước 7.2 — Viết `src/matching/signals.py`**
- Tính $s^{(k)}_{ij} = \cos(e^{JD}_{i,a_k}, e^{CV}_{j,b_k})$, k = 1..9.
- Vectorized, kích thước (n_jobs, n_resumes, 9).
- Kiểm tra không NaN, giá trị trong [-1, 1].

**Bước 7.3 — Tính tương quan tín hiệu với nhãn trên train**
- Tín hiệu gần như không tương quan → xem lại cặp view.

**Bước 7.4 — Chạy static fusion trọng số đều**
- $S_{ij} = \sum_k w_k s^{(k)}_{ij}$.
- Ghi kết quả 5 fold.

**Bước 7.5 — Chạy static fusion tối ưu**
- Grid search trọng số trên val.
- Ghi cấu hình tốt nhất vào `results/hparam_log.csv`.

✅ **Xong tuần 7 khi:** có tensor tín hiệu, static đều và static tối ưu có kết quả.

---

## TUẦN 8 – Adaptive fusion + concat-MLP + BPR

**Bước 8.1 — Viết `src/fusion/concat_mlp.py` (B7)**
- Input: 9 tín hiệu + mask.
- MLP ra điểm trực tiếp, không weighted sum.

**Bước 8.2 — Viết `src/fusion/adaptive.py`**
- MLP nhận vector tín hiệu + mask.
- Softmax ra trọng số $w_{ij,k}$.
- $S_{ij} = \sum_k w_{ij,k} s^{(k)}_{ij}$.

**Bước 8.3 — Viết `training/pairs.py`**
- Dựng cặp BPR trong cùng JD của train.
- $j^+$ có relevance cao hơn $j^-$.
- Kiểm tra không dùng JD của val/test.

**Bước 8.4 — Viết vòng huấn luyện BPR**
- Loss: $\mathcal{L} = -\sum_i \sum_{(j^+, j^-)} \log \sigma(S_{ij^+} - S_{ij^-})$.
- AdamW, early stopping theo NDCG@K trên val.
- Lưu checkpoint và trọng số gating.

**Bước 8.5 — Tune lưới nhỏ trên val**
- Ngân sách tương đương baseline.
- Ghi mọi cấu hình vào `results/hparam_log.csv`.

**Bước 8.6 — Chạy B7, adaptive, static tối ưu trên fold 1**
- So sánh trên val.

**Bước 8.7 — Viết nháp Method**
- Bắt đầu viết Method và Experimental Setup khi thiết kế đã chốt.
- Sẽ cập nhật khi có kết quả.

✅ **Xong tuần 8 khi:** B7 và adaptive chạy được trên fold 1, nháp Method bắt đầu.

---

## TUẦN 9 – Cổng quyết định + chạy full fold

**Bước 9.1 — Họp cổng quyết định**

So sánh trên val, dùng tiêu chí định lượng đã chốt ở Bước 3.7 (chênh lệch ≥ 0.01 NDCG coi là ">"; < 0.01 và CI chồng lấn coi là "≈"):

| Kết quả | Hành động |
|---|---|
| Multi-view > B2 và B3 | Đi tiếp RQ1, RQ2 |
| Multi-view > B2 nhưng ≈ B3 | Đổi hướng: lợi ích từ lọc nhiễu LLM |
| Multi-view ≤ B2 | Kiểm tra chất lượng trích xuất, cặp view, cặp BPR |
| Adaptive ≈ static tối ưu hoặc ≈ concat-MLP | Báo H3 bị bác bỏ |
| Kém B6 rõ rệt | Đổi claim sang chi phí và diễn giải |

**Bước 9.2 — Ghi quyết định vào `research_log.md` kèm ngày**

**Bước 9.3 — Chạy full 5 fold cho các biến thể được chọn**
- 3 seed (5 seed nếu kịp).
- Ghi `results/fusion_runs.csv`.

**Bước 9.4 — Kiểm tra ổn định**
- Trọng số gating có nhất quán giữa seed không.

✅ **Xong tuần 9 khi:** cổng quyết định đã chốt, full fold chạy xong.

---

## TUẦN 10 – Thực nghiệm chính (test lần đầu)

**Bước 10.1 — Khóa siêu tham số và checkpoint theo val**
- Không đổi gì sau bước này.

**Bước 10.2 — Đánh giá test một lần**
- Chạy tất cả mô hình trên test.
- Lưu `results/main_runs.csv`.

**Bước 10.3 — Kiểm định thống kê**
- So đề xuất với từng baseline bằng NDCG từng JD, **gộp các fold sao cho mỗi JD xuất hiện đúng một lần**.
- Paired bootstrap + Wilcoxon.
- Hiệu chỉnh Holm.
- Báo CI 95% và effect size.

**Bước 10.4 — Đóng băng bảng chính**
- mean ± std qua seed và fold.
- Đánh dấu mức ý nghĩa.

✅ **Xong tuần 10 khi:** bảng chính đóng băng, kiểm định xong.

---

## TUẦN 11 – RQ3 (nếu làm) + chi phí + chuyển miền

**Bước 11.1 — Non-inferiority với B6**
- So sánh theo biên δ đã ghi.
- Cận dưới CI 95% của hiệu số NDCG (B6 − đề xuất) so với −δ.

**Bước 11.2 — Đo chi phí suy luận**
- Cùng phần cứng.
- Tách: chi phí trích xuất một lần + chi phí xếp hạng mỗi JD.
- Ghi `results/cost.csv`.

**Bước 11.3 — Chuyển miền (nếu kịp)**
- Tải `med2425/resume-job-fit-merged-v1`.
- Loại cặp trùng với dataset chính, báo số lượng.
- Train A → test B và ngược lại.
- So độ tụt NDCG với B4 và B6.
- Nếu còn quá ít mẫu → bỏ phần này, ghi hạn chế.

**Bước 11.4 — Độ nhạy LLM (nếu kịp)**
- Chạy lại trích xuất với LLM thứ hai.
- Cache riêng.
- Báo độ lệch NDCG.

✅ **Xong tuần 11 khi:** chi phí đo xong, chuyển miền và độ nhạy LLM có hoặc bỏ rõ lý do.

---

## TUẦN 12 – Ablation và phân tích

**Bước 12.1 — Ablation RQ1**
- B2 vs B3 vs multi-view static.

**Bước 12.2 — Ablation RQ2**
- Static đều, static tối ưu, concat-MLP, adaptive.

**Bước 12.3 — Ablation thành phần**
- 7 cặp lõi (bỏ 2 cặp chéo).
- 9 cặp đầy đủ (7 lõi + 2 chéo).
- Bỏ từng view.
- Một tín hiệu đơn.

**Bước 12.4 — Ablation backbone (optional)**
- Thử thêm một embedding model khác.

**Bước 12.5 — Diễn giải trọng số**
- Trọng số gating trung bình theo view và theo nhãn.
- Độ nhất quán giữa seed.
- Tương quan hạng Spearman giữa trọng số view và mức giảm NDCG khi bỏ view.
- 3–5 ví dụ định tính.

**Bước 12.6 — Phân tích lỗi**
- Chọn 10–15 JD có NDCG thấp nhất.
- Phân loại nguyên nhân: trích xuất sai, view thiếu, nhãn mơ hồ, CV nhiễu.
- Bảng đếm kèm ví dụ.

**Bước 12.7 — Phân tích khác**
- NDCG theo độ dài CV.
- Che tên và thông tin định danh trong CV rồi chạy lại (nếu không làm được, ghi hạn chế).

✅ **Xong tuần 12 khi:** ablation, diễn giải, phân tích lỗi xong.

---

## TUẦN 13 – Viết bài

**Bước 13.1 — Hoàn thiện Method và Experimental Setup**
- Đã viết nháp từ tuần 8.
- Cập nhật theo kết quả cuối.

**Bước 13.2 — Viết Results và Analysis**
- Mọi số lấy từ file kết quả bằng script, không gõ tay.

**Bước 13.3 — Viết Related Work**

**Bước 13.4 — Viết Introduction và Abstract**
- Động cơ, khoảng trống, RQ, đóng góp.

**Bước 13.5 — Viết Limitations, Ethics, Conclusion**

**Bước 13.6 — Viết `scripts/make_figures.py`**
- Sinh mọi hình và bảng từ file kết quả.

✅ **Xong tuần 13 khi:** bản thảo đầy đủ.

---

## TUẦN 14 – Tái lập, phản biện, nộp

**Bước 14.1 — Viết README chạy từ đầu**
- Config, seed, danh sách fold, hash commit, model hash Ollama.

**Bước 14.2 — Chạy lại một thí nghiệm từ môi trường sạch**
- Kiểm tra khớp kết quả.
- Nếu lệch → ghi nguyên nhân.

**Bước 14.3 — Phản biện nội bộ**
- Nhờ giảng viên hướng dẫn đọc như reviewer.
- Nhờ một người ngoài dự án đọc.
- Ghi lại và xử lý từng nhận xét.

**Bước 14.4 — Chỉnh sửa theo phản biện**

**Bước 14.5 — Kiểm tra định dạng và deadline nơi nộp**
- Xem trực tiếp trên trang hội nghị/tạp chí.
- Không dùng danh sách chưa xác minh.

**Bước 14.6 — Nộp**

✅ **Xong tuần 14 khi:** bản thảo nộp, gói tái lập xong, phản biện xử lý xong.

---

## Cập nhật `research_log.md` theo mốc

| Lần | Tuần | Nội dung |
|---|---|---|
| 1 | 1 | RQ, giả thuyết, tiêu chí bác bỏ, metric, quy tắc chọn mô hình |
| 2 | 2 | K, chiến lược fold, quyết định xử lý rò rỉ CV |
| 3 | 3 | Tiêu chí định lượng cổng quyết định |
| 4 | 4 | Biên δ (ghi trước khi chạy test) |
| 5 | 9 | Quyết định cổng |

---

## Thứ tự cắt nếu trễ

1. Cross-attention view-level.
2. LLM trích xuất thứ hai.
3. Dataset thứ hai (chuyển miền).
4. Ablation backbone.
5. Bớt seed (5 → 3).
6. BM25, LLM zero-shot.

**Giữ bằng mọi giá:** B0, B1, B2, B3, B4, B6, B7, static tối ưu, adaptive, bảng chính, kiểm định.

---

## Việc làm ngay hôm nay

1. Tạo repo git, dựng cấu trúc thư mục (Bước 2.1).
2. Viết `requirements.txt` và `set_seed()` (Bước 2.2).
3. Tải dataset chính, đọc dataset card và license (Bước 2.3, 2.4).
4. Cài Ollama, test Qwen3:4b trên mẫu từ dataset (Bước 2.6).
5. Mở `notebooks/01_eda.ipynb`, chạy EDA sơ bộ (Bước 2.7).

Sau khi có kết quả EDA (số JD, median CV mỗi JD), mình có thể chốt K và viết script chia fold cho bạn.
