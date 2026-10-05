# Kế hoạch thực hiện đồ án Frevia (AI Resume–Job Matching) — bản v1, 8 tuần

> **Lưu ý:** đây là bản kế hoạch cũ, đã được thay thế bởi `PLAN.md` (bản 14 tuần).
> Giữ lại để đối chiếu. Cấu trúc `frevia/` trong bản này đã đổi thành `src/`.

Bám theo cấu trúc code `frevia/extraction`, `representation`, `matching`, `fusion`, `attention`, `evaluation`, `experiments`.
Các file mới đề xuất thêm được đánh dấu **(mới)**. Mỗi bước có đầu ra cụ thể và tiêu chí hoàn thành.

---

## Giai đoạn 0: Chuẩn bị môi trường (ngày 1)

1. Tạo cấu trúc thư mục: `data/`, `cache/`, `results/`, `configs/`, `notebooks/`.
2. Cố định phiên bản thư viện trong `requirements.txt` (torch, sentence-transformers, scikit-learn, pandas, ollama client).
3. Tạo `configs/base.yaml` chứa seed, đường dẫn, tên model, K, siêu tham số. Mọi script đọc từ file này, không hard-code.
4. Viết hàm `set_seed()` dùng chung cho random, numpy, torch.

**Hoàn thành khi:** chạy được `python experiments/run_frevia.py --help` trên môi trường sạch.

---

## Giai đoạn 1: Dữ liệu và EDA (tuần 1)

### Bước 1.1: Tải và kiểm tra dữ liệu

- Tải `cnamuangtoun/resume-job-description-fit` bằng `datasets`, lưu bản gốc vào `data/raw/`.
- Kiểm tra cột, giá trị thiếu, nhãn lạ.

### Bước 1.2: Làm sạch và gán ID

- Gán `job_id` bằng hash của văn bản JD, `resume_id` bằng hash của văn bản CV.
- Loại cặp trùng lặp hoàn toàn. Ghi lại số lượng đã loại.
- Cảnh báo nếu cùng một cặp (CV, JD) có nhãn mâu thuẫn.

### Bước 1.3: EDA (`notebooks/01_eda.ipynb`)

- Số cặp, số JD riêng biệt, số CV riêng biệt.
- Phân bố số CV mỗi JD (min, median, max) và phân bố nhãn tổng thể và theo JD.
- Số JD không có CV nào Good Fit (JD này không đo được recall).
- Độ dài văn bản tính theo token, tỉ lệ vượt 256 và 512 token.

### Bước 1.4: Ánh xạ nhãn và định nghĩa relevant

- No Fit = 0, Potential Fit = 1, Good Fit = 2.
- Hai định nghĩa relevant cho P@K và R@K: (A) chỉ Good Fit, (B) Good + Potential. NDCG dùng relevance 0/1/2.

### Bước 1.5: Split theo JD

- Chia theo `job_id` (ví dụ 70/15/15), cùng một JD không xuất hiện ở hai split.
- Lưu `data/splits/split_seed42.json` (danh sách `job_id` mỗi split).
- Kiểm tra: không giao nhau giữa các split, phân bố nhãn mỗi split gần nhau.

### Bước 1.6: Quyết định K

- Nếu median số CV mỗi JD dưới 15, dùng K = 3, 5 làm chính và vẫn báo K = 10. Ghi lý do vào nhật ký.

**Hoàn thành khi:** có file split cố định, báo cáo EDA, và định nghĩa relevant đã chốt.

---

## Giai đoạn 2: Evaluation (tuần 2, làm trước baseline)

### Bước 2.1: `evaluation/metrics.py`

- Cài P@K, R@K, MRR, NDCG@K. Tính theo từng JD, rồi lấy trung bình.
- Quy ước xử lý JD không có CV relevant: bỏ khỏi P@K, R@K, MRR, nhưng NDCG đặt bằng 0 hoặc bỏ, thống nhất một cách và ghi rõ.
- Xử lý hòa điểm (tie): sắp xếp ổn định theo `resume_id`.

### Bước 2.2: Unit test (`tests/test_metrics.py`) **(mới)**

- Dựng 3 đến 4 ví dụ tính tay (ranking hoàn hảo, ranking đảo ngược, có hòa điểm) và so với kết quả code.
- Có thể đối chiếu NDCG với `sklearn.metrics.ndcg_score`.

### Bước 2.3: Hàm đánh giá chung `evaluate(score_matrix, labels, split)`

- Đầu vào là ma trận điểm (n_jobs, n_resumes) hoặc danh sách điểm theo JD, đầu ra là dict metric và mảng điểm từng JD (để làm kiểm định thống kê sau).

**Hoàn thành khi:** test metric qua hết, và mọi phương pháp sau đều gọi cùng một hàm này.

---

## Giai đoạn 3: Baseline (tuần 2)

### Bước 3.1: TF-IDF + Cosine (`baselines/tfidf.py`) **(mới)**

- Fit vectorizer chỉ trên train, biến đổi val/test, tính cosine mỗi cặp (JD, CV).
- Thử nhanh 2 đến 3 cấu hình (ngram, max_features) trên val, chọn tốt nhất.

### Bước 3.2: SBERT toàn văn + Cosine (`baselines/sbert_full.py`) **(mới)**

- Dùng `all-MiniLM-L6-v2` cho toàn văn CV và JD. Dùng một chiến lược cắt cố định hoặc chia chunk rồi lấy trung bình, và báo tỉ lệ mẫu bị cắt.
- Cache embedding vào `cache/sbert_full/`.

### Bước 3.3 (tuỳ chọn): BM25 và Pretrained Cross-Encoder

- BM25 bằng `rank_bm25`. Cross-encoder `cross-encoder/ms-marco-MiniLM-L-6-v2` không fine-tune, chấm mọi cặp trong test.

### Bước 3.4: Chạy và lưu kết quả

- Xuất `results/baselines.csv` (metric trên val và test) và `results/per_job_<method>.csv`.

**Hoàn thành khi:** có bảng baseline đầu tiên.

---

## Giai đoạn 4: Trích xuất view (tuần 3)

### Bước 4.1: Kiểm tra prompt trên mẫu nhỏ

- Chạy `frevia/extraction/qwen.py` trên 20 CV và 20 JD, đọc kết quả bằng mắt.
- Chỉnh prompt cho đến khi JSON đầu ra ổn định.

### Bước 4.2: Validator đầu ra

- Kiểm tra JSON hợp lệ, đủ 7 view (CV) và 5 view (JD), không rỗng. Nếu sai thì thử lại tối đa N lần, và ghi cờ lỗi.

### Bước 4.3: Chạy toàn bộ có cache

- Lưu mỗi văn bản thành một file JSON trong `cache/views/` theo `resume_id` hoặc `job_id`. Chạy lại phải bỏ qua mục đã có.
- Ghi thời gian xử lý mỗi văn bản (dùng cho bảng chi phí).

### Bước 4.4: Thống kê chất lượng (`notebooks/02_extraction_quality.ipynb`) **(mới)**

- Tỉ lệ view rỗng theo từng view, tỉ lệ lỗi sau thử lại, độ dài view trung bình.
- Chấm tay 30 đến 50 mẫu: view có đúng và đủ không (thang 0/1/2), ghi lại ví dụ lỗi.
- Kiểm tra độ ổn định: chạy lại 20 mẫu lần hai, đo mức giống nhau.

### Bước 4.5: Embed view (`frevia/representation/views.py`)

- Embed từng view bằng MiniLM, lưu `cache/view_emb/*.npy`. View rỗng dùng vector 0 và kèm mask để fusion biết.

**Hoàn thành khi:** mọi CV và JD có view và embedding, thống kê chất lượng đã có. Nếu tỉ lệ view hỏng cao, sửa prompt rồi chạy lại trước khi đi tiếp.

---

## Giai đoạn 5: Phương pháp chính (tuần 4)

### Bước 5.1: Tensor tín hiệu (`frevia/matching/signals.py`)

- Tạo (n_jobs, n_resumes, 9) cho từng split. Với mask view rỗng, đặt tín hiệu bằng 0 hoặc giá trị trung lập, và thống nhất cách này.
- Kiểm tra kích thước, không có NaN, giá trị cosine trong [-1, 1].

### Bước 5.2: Static fusion (`fusion/scoring.py`)

- Đặt trọng số cố định (đều nhau hoặc theo kinh nghiệm). Chạy đánh giá trên val và test, đây là dòng 3.
- Tuỳ chọn: tìm trọng số bằng grid search trên val để có "static tốt nhất", tránh so sánh với một static yếu.

### Bước 5.3: Dựng cặp BPR (`training/pairs.py`) **(mới)**

- Với mỗi JD trong train: positive là CV có relevance cao hơn, negative là CV có relevance thấp hơn. Lấy mẫu mỗi epoch.
- Kiểm tra: chỉ dùng JD của train, không rò rỉ sang val và test.

### Bước 5.4: Adaptive fusion (`fusion/gating.py`)

- Huấn luyện bằng BPR, tối ưu AdamW, early stopping theo NDCG@K trên val.
- Lưu checkpoint tốt nhất theo val, và lưu cả trọng số gating để diễn giải sau.

### Bước 5.5: View-level cross-attention (`attention/cross_attention.py`)

- Huấn luyện cùng quy trình và cùng cách chọn checkpoint.
- Giữ số tham số của hai mô hình ở mức so sánh được, và ghi số tham số vào bảng.

### Bước 5.6: Chọn siêu tham số

- Chỉ tinh chỉnh learning rate, weight decay, hidden size trên val, bằng lưới nhỏ. Ghi lại mọi cấu hình đã thử.

**Điểm quyết định cuối tuần 4:** so dòng 3, 4, 5 với SBERT toàn văn (dòng 2) trên val.

- Hơn: đi tiếp.
- Ngang hoặc kém: kiểm tra lại chất lượng trích xuất và cách dựng cặp BPR trước. Nếu vẫn không hơn, chuyển trọng tâm sang phân tích diễn giải và ablation theo view.

---

## Giai đoạn 6: Thực nghiệm đầy đủ (tuần 5)

### Bước 6.1: Chạy nhiều seed

- Các mô hình học được chạy 5 seed (ví dụ 42, 43, 44, 45, 46). Split dữ liệu giữ nguyên, chỉ đổi seed khởi tạo và lấy mẫu.
- Lưu `results/main_runs.csv` (method, seed, split, các metric).

### Bước 6.2: Đánh giá test một lần duy nhất

- Tất cả siêu tham số đã khóa từ val. Không chỉnh gì theo test.

### Bước 6.3: Kiểm định thống kê (`evaluation/significance.py`) **(mới)**

- Dùng điểm NDCG từng JD, so cặp (đề xuất vs baseline) bằng paired bootstrap hoặc Wilcoxon signed-rank. Báo p-value và khoảng tin cậy.

### Bước 6.4: Bảng kết quả chính

- Mean ± std qua seed cho mỗi phương pháp, in đậm giá trị tốt nhất. Đóng băng bảng này.

**Hoàn thành khi:** có `results/main_table.csv` và kết quả kiểm định.

---

## Giai đoạn 7: Ablation và diễn giải (tuần 6)

### Bước 7.1: Ablation trích xuất view

- So SBERT toàn văn (dòng 2) với multi-view static (dòng 3).

### Bước 7.2: Ablation tín hiệu

- Chạy lại với: chỉ 7 tín hiệu lõi; 7 lõi + 2 chéo; bỏ lần lượt từng view (leave-one-out); dùng một tín hiệu duy nhất. Mỗi cấu hình chạy cùng số seed.

### Bước 7.3: Ablation fusion

- Static, Adaptive, Cross-attention (đã có từ giai đoạn 6, chỉ cần trình bày lại).

### Bước 7.4: Diễn giải trọng số

- Lấy trọng số gating trung bình theo từng view, theo nhãn (Good, Potential, No Fit). Vẽ heatmap hoặc biểu đồ cột.
- Lấy 3 đến 5 ví dụ cụ thể: một cặp được xếp cao và giải thích view nào đóng góp nhiều nhất.

**Hoàn thành khi:** có bảng ablation và 2 đến 3 hình diễn giải.

---

## Giai đoạn 8: Phân tích lỗi và chi phí (tuần 7)

### Bước 8.1: Phân tích lỗi

- Chọn 10 đến 15 JD có NDCG thấp nhất. Phân loại nguyên nhân: trích xuất sai, view thiếu, nhãn mơ hồ (Potential so với Good), CV quá ngắn hoặc nhiễu.
- Làm bảng đếm theo nhóm nguyên nhân kèm 2 đến 3 ví dụ minh hoạ.

### Bước 8.2: Hiệu năng theo độ dài và theo nhóm

- Chia theo độ dài CV (ngắn, vừa, dài) và theo nhóm nghề nếu suy ra được. Báo NDCG mỗi nhóm.

### Bước 8.3: Chi phí

- Đo: trích xuất Qwen mỗi văn bản, embed mỗi view, chấm một JD với N CV (chỉ phần fusion), và so với Pretrained Cross-Encoder nếu có.
- Làm bảng thời gian trên cùng phần cứng, ghi rõ phần cứng.

### Bước 8.4: Mục hạn chế

- Dataset nhỏ và có thể mang tính tổng hợp (đọc dataset card để xác nhận).
- LLM 4B có nhiễu trích xuất.
- Embedding đóng băng, attention chỉ ở mức view.
- Chưa so với cross-encoder fine-tune đầy đủ.
- Thông tin cá nhân trong CV có thể gây thiên lệch.

**Hoàn thành khi:** có bảng lỗi, bảng chi phí, và đoạn hạn chế nháp xong.

---

## Giai đoạn 9: Viết báo cáo và slide (tuần 8)

1. **Ngày 1 đến 2:** viết Phương pháp và Thiết lập thực nghiệm (có thể bám nhật ký quyết định).
2. **Ngày 3 đến 4:** viết Kết quả và Phân tích, mọi con số lấy từ file kết quả, không gõ tay.
3. **Ngày 5:** viết Giới thiệu, Công trình liên quan (PJFNN, ConFit và ConFit v2, CareerBERT, SBERT, cross-encoder re-ranking), Kết luận.
4. **Ngày 6:** kiểm tra khớp số liệu giữa bảng, hình và văn bản; kiểm tra trích dẫn tồn tại và đúng.
5. **Ngày 7:** làm slide (khoảng 12 đến 15 trang) và chuẩn bị câu trả lời cho các câu hỏi dự kiến.

**Các câu hỏi hội đồng có thể hỏi:** vì sao không dùng cross-encoder fine-tune, trích xuất LLM sai thì sao, vì sao dùng BPR thay vì phân loại, kết quả có bền qua các seed không, dataset có đại diện thực tế không.

---

## Checklist theo dõi nhanh

| Mốc | Sản phẩm | Xong |
|---|---|---|
| Cuối tuần 1 | Split theo JD, EDA, định nghĩa relevant | ☐ |
| Cuối tuần 2 | Metrics đã test, bảng baseline | ☐ |
| Cuối tuần 3 | View và embedding toàn bộ, thống kê chất lượng | ☐ |
| Cuối tuần 4 | Static, Adaptive, Cross-attention chạy được, quyết định hướng | ☐ |
| Cuối tuần 5 | Bảng chính nhiều seed, kiểm định | ☐ |
| Cuối tuần 6 | Ablation, hình diễn giải | ☐ |
| Cuối tuần 7 | Phân tích lỗi, chi phí, hạn chế | ☐ |
| Cuối tuần 8 | Báo cáo và slide | ☐ |

---

## Lưu ý để tránh sai sót

- Không bao giờ chọn mô hình, ngưỡng hay siêu tham số theo điểm test.
- Mỗi lần đổi prompt trích xuất phải xóa cache và chạy lại từ bước 4.3, nếu không kết quả sẽ lẫn hai phiên bản.
- Ghi `git commit` sau mỗi mốc và lưu hash commit cùng file kết quả.
- Mọi bảng và hình trong báo cáo được sinh từ script, không chỉnh tay.
