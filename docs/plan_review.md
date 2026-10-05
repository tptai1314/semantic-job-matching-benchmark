# Phân tích PLAN.md cho bài NCKH — thiếu sót & đề xuất sửa

> Người thực hiện: kiểm tra trực tiếp repo + chạy thử EDA và baseline B1 (TF-IDF) trên dataset thật.
> Ngày: 2026-10-05. Trạng thái repo: commit `20b4078`, mới xong tuần 1–2 (cấu trúc thư mục, config, entry point).
> Mọi số liệu trong tài liệu này là **đo được**, không phải suy đoán; script tái lập nằm ở `scripts/quick_eda.py` và `scripts/probe_b1_tfidf.py` (xem Phụ lục).

---

## 0. Kết luận ngắn (đọc trước)

`PLAN.md` là một **kế hoạch thi công rất tốt** (trình tự hợp lý, có pre-registration, có cổng quyết định, có danh sách cắt khi trễ, có quy tắc tie/lỗi JD). Nhưng nó **chưa phải là đề cương nghiên cứu**: thiếu tuyên bố đóng góp, thiếu RQ/giả thuyết trong bản thân plan, thiếu mốc/định dạng nơi nộp, thiếu kế hoạch tính toán, và có **3 lỗi thiết kế ở mức có thể làm phản biện bác cả bài**:

| # | Vấn đề | Mức độ | Bằng chứng |
|---|---|---|---|
| **A** | Ngưỡng hiệu quả thực tiễn 0.01 NDCG và biên δ = max(std, 0.01) **bất khả thi về mặt thống kê** | 🔴 Chặn | Với 305 JD đo được: MDE (80% power) ≈ **0.023–0.046 NDCG**; muốn phát hiện 0.01 cần **1.600–6.500 JD** |
| **B** | Rò rỉ CV giữa các fold: **476/477 (99,8%)** CV trong test cũng có trong train; 607/642 CV lặp >1 lần trong train | 🔴 Chặn | Đo trên dataset |
| **C** | Thí nghiệm "chuyển miền" (bước 11.3) **sai thiết kế**: dataset thứ hai chứa chính dataset chính và được **dán nhãn lại bằng LLM khác** → đo label shift, không đo domain shift | 🔴 Chặn | Dataset card của `med2425/resume-job-fit-merged-v1` |
| **D** | Thiếu **kiểm soát then chốt**: rút gọn văn bản **không dùng LLM** (heuristic). Không có nó thì không thể kết luận "LLM giúp gì" | 🟠 Nặng | Xem §3.1 |
| **E** | Baseline mạnh nhất bị xếp vào danh sách **cắt được**: LLM zero-shot / LLM-as-judge; và chỉ dùng 1 encoder yếu (MiniLM-L6) | 🟠 Nặng | Bước 1.5, "Thứ tự cắt nếu trễ" #6 |
| **F** | Đánh giá chất lượng trích xuất (bước 6.3) là **"nếu kịp"**, nhưng đây là mắt xích sống còn của cả phương pháp | 🟠 Nặng | Bước 6.3 |
| **G** | **Không có GPU**: `torch 2.14.0+cpu`, `cuda_available: false`, device = cpu — nhưng kế hoạch yêu cầu fine-tune B4/B6 × 3–5 seed × 5 fold | 🟠 Nặng | `results/logs/env_info.json` |
| **H** | Nguồn gốc + license dataset **không tồn tại**: dataset chính không có dataset card, không có license | 🟠 Nặng | HF API: chỉ có `train.csv`, `test.csv`, `.gitattributes` |
| **I** | Benchmark có thể **quá dễ / sai construct validity**: TF-IDF đã đạt NDCG@5 = 0,657; 50% nhãn là No Fit; **59% JD không có Good Fit nào** | 🟡 Trung bình | Probe B1 |

**Tin tốt (ngược với lo lắng trong plan):** khối lượng trích xuất LLM **rất nhỏ**. Toàn bộ dataset chỉ có **994 văn bản riêng biệt** (643 CV + 351 JD) cho 8.000 cặp (nén ≈ 8 lần). Nghĩa là plan **có thể chi trả** cho: LLM thứ hai, LLM mạnh hơn, và một tập gold chuẩn hoá tay — những thứ đang bị xếp vào "nếu kịp".

---

## 1. Bằng chứng: số liệu thật của dataset (thay cho EDA tuần 2)

Chạy trên `cnamuangtoun/resume-job-description-fit`, gộp train+test = **8.000 dòng**.

| Chỉ số | Train | Test | Gộp |
|---|---|---|---|
| Số cặp | 6.241 | 1.759 | 8.000 |
| JD riêng biệt | 280 | 71 | **351** |
| CV riêng biệt | 642 | 477 | **643** |
| CV/JD: min / p25 / **median** / p75 / max | 1 / 7 / **14,5** / 29,3 / 111 | 2 / 8 / 20 / 33,5 / 89 | — |
| Nhãn No Fit / Potential / Good | 50,4% / 24,9% / 24,7% | 48,7% / 25,2% / 26,0% | ~50/25/25 |
| JD **không có Good Fit** | **165/280 (58,9%)** | 43/71 (60,6%) | — |
| JD không có relevant nào (Good+Potential) | 36/280 | 10/71 | **46/351 (13,1%)** |
| JD có ≥ K ứng viên | K=3: 274 · K=5: 256 · K=10: **175** | — | — |
| Từ/CV: median / p90 / max | 618 / 1.153 / 3.134 | — | — |
| Từ/JD: median / p90 / max | 328 / 695 / 1.079 | — | — |
| Tỉ lệ vượt 512 từ (giới hạn MiniLM) | CV **72,7%** · JD 27,0% | CV 72,8% · JD 39,2% | — |
| Cặp trùng / nhãn mâu thuẫn (train) | 7 cặp trùng, **6 cặp mâu thuẫn nhãn** | 0 | — |
| CV xuất hiện >1 lần | **607/642** | 344/477 | — |
| **Trùng lặp giữa train và test** | JD chung: **0** · CV chung: **476/477 (99,8%)** · cặp chung: 0 | | |
| **Khối lượng trích xuất LLM** | 922 văn bản | 548 | **994 văn bản** |

Nguồn gốc dataset (kiểm tra bằng Hugging Face API):

- `cnamuangtoun/resume-job-description-fit`: **không có README/dataset card, không có license**. Chỉ có `train.csv`, `test.csv`.
- `med2425/resume-job-fit-merged-v1` (dataset "chuyển miền" trong bước 11.3): 93.733 dòng, 13 domain, có cột `source`; card ghi rõ: *"Combined two public datasets: `cnamuangtoun/resume-job-description-fit` and `kens1ang/resume-job-fit-augmented`… **Labeled using Qwen2.5-32B** (via Ollama, temperature=0)"* và *"Test Set: Created from the original test split"*.

---

## 2. Ba lỗi thiết kế mức nghiêm trọng (phải sửa trước khi chạy)

### 2.A. Cổng quyết định 0.01 NDCG và δ = max(1 std, 0.01) là bất khả thi về thống kê

Bước 3.7 chốt: chênh lệch **≥ 0,01 NDCG** coi là "lớn hơn thực tiễn"; bước 4.6 chốt **δ = max(1 std, 0,01)** và bước 11.1 làm non-inferiority theo δ đó. Với dữ liệu thật:

| Đại lượng (probe B1 out-of-fold, 305 JD) | Giá trị |
|---|---|
| NDCG@3 / **NDCG@5** / NDCG@10 (macro) | 0,630 / **0,657** / 0,685 |
| Độ lệch chuẩn **giữa các JD** của NDCG@5 | **0,288** |
| SE của trung bình / nửa khoảng CI 95% | 0,0165 / **±0,032** |
| **MDE** (80% power, 2 phía) nếu σ_d = σ_jd | **0,046** |
| MDE nếu σ_d = 0,5·σ_jd (tương quan cao giữa 2 mô hình) | 0,023 |
| Số JD cần để phát hiện **0,01** | **1.622 – 6.487** (đang có 305) |
| Số JD cần để phát hiện **0,03** | 180 – 721 |

Hệ quả trực tiếp:

1. **Cổng quyết định ở bước 9.1 không thể vận hành như đã viết.** Kết cục gần như chắc chắn là "CI chồng lấn" → rơi vào nhánh "≈", tức plan sẽ tự đẩy mình vào bảng "Multi-view ≤ B2 → kiểm tra lại" dù phương pháp có thể thực sự tốt hơn.
2. **Non-inferiority ở bước 11.1 gần như không bao giờ kết luận được.** Muốn thiết lập non-inferiority với δ = 0,01 cần **cận trên CI < 0,01**, trong khi nửa khoảng CI đã là ~0,032. Test này sẽ luôn "inconclusive" — tệ hơn cả việc không làm.
3. Ngưỡng 0,01 **không có cơ sở**: nó vừa là ngưỡng cổng (3.7) vừa là thành phần của δ (4.6) → **vòng tròn logic**, không phải SESOI.

**Sửa thế nào:**

- Ở tuần 3 (khi đã có B0/B1), **đo σ_d thật** (paired, theo từng JD, gộp fold) và **in ra MDE**. Đây phải là một *deliverable* của bước 3.7, không chỉ là một con số cảm tính.
- Đặt **SESOI = 3 × MDE** (dự kiến ~0,05–0,07) hoặc lấy từ literature (khoảng cách giữa các phương pháp trong ConFit/PJFNN thường 0,03–0,08 NDCG). Ghi rõ: "0,01 là ngưỡng **bỏ qua** (negligible), không phải ngưỡng **ý nghĩa**".
- Nếu muốn δ nhỏ: **tăng số JD** (xem §2.C và §5 phương án B), hoặc **đổi metric chính** — NDCG@10 có σ = 0,241 (nhỏ nhất trong 3 K) nên **có power cao nhất**; bảng số liệu ở §1 là căn cứ để chốt K chính bằng dữ liệu chứ không bằng quy tắc "median < 15".
- Bổ sung: **cluster bootstrap theo `job_id`** và báo cáo cả **CI của effect size** (Cliff's δ hoặc Cohen's d_z) — reviewer sẽ hỏi "khác 0,03 thì có nghĩa gì về mặt tuyển dụng".
- Sửa câu chữ bước 11.1: đúng phải là *"cận dưới CI 95% của (đề xuất − B6) so với −δ"*. Như đang viết ("B6 − đề xuất" so với "−δ") là **ngược dấu**, và nên dùng **kiểm định một phía (TOST-style)** thay vì chỉ so CI.

### 2.B. Rò rỉ CV — không phải giả thuyết, mà là 99,8%

Bước 2.10 mới chỉ *"nếu CV xuất hiện ở cả train và test với JD khác nhau → quyết định chấp nhận hay chia theo cả `resume_id`, ghi vào log"*. Thực tế: **476/477 CV trong test có mặt trong train**, và **607/642 CV trong train xuất hiện nhiều hơn một lần**. Với bất kỳ split nào theo `job_id` (kể cả GroupKFold), tỉ lệ này gần như **100%** — vì chỉ có 643 CV cho 351 JD.

Điều này không sai về nguyên tắc (đây là bài toán **re-ranking trong một pool ứng viên dùng chung**, không phải retrieval toàn corpus), nhưng:

- Với **B4/B6 (fine-tune encoder)**, rò rỉ CV cho phép mô hình **học thuộc văn bản CV** → lợi thế giả tạo, và **lợi thế này lớn hơn** so với phương pháp không fine-tune. Kết luận "đề xuất ≈ B6" có thể chỉ là hệ quả của việc B6 bị thổi phồng.
- Plan hiện **hoãn quyết định** và không có quy tắc → vi phạm chính tinh thần pre-registration mà plan đang theo đuổi.

**Sửa:**

- Chốt **primary = split theo `job_id`** (chuẩn cho bài toán), **secondary = split theo `resume_id`** (CV-disjoint, sẽ khó hơn nhiều) và **báo cáo cả hai** như một phân tích độ bền. Việc chủ động công bố "khoảng cách giữa hai split" là một **đóng góp về độ chặt chẽ** rất dễ bán với hội đồng, thay vì để reviewer tự phát hiện.
- Bắt buộc: bảng "leakage audit" (JD chung, CV chung, cặp chung) như một hình/bảng trong bài.
- Cảnh báo trước cho chính mình: nếu B4/B6 tụt mạnh ở split theo `resume_id` còn multi-view không tụt → đó là **kết quả đẹp**, hãy chuẩn bị cho nó.

### 2.C. Thí nghiệm chuyển miền (11.3) sai thiết kế

`med2425/resume-job-fit-merged-v1` **không độc lập** với dataset chính: nó gộp chính `cnamuangtoun/...` với một dataset khác, rồi **dán nhãn lại toàn bộ bằng Qwen2.5-32B**. Vì vậy "train A → test B" đo **sự đổi hàm nhãn (label shift)**, không đo **dịch chuyển miền**. Kết quả "độ tụt NDCG" sẽ không diễn giải được.

**Sửa (đều rẻ):**

- Dùng cột **`source`** của dataset B để **lọc bỏ** các dòng có nguồn gốc từ A (`ds1_original`) thay vì chỉ dedupe theo hash văn bản.
- Hoặc **đổi mục tiêu** thí nghiệm: không gọi là "chuyển miền" mà gọi đúng là **"độ nhạy của phương pháp với hàm nhãn do LLM khác tạo ra"** — đây là một câu hỏi hợp lệ và thú vị (label-function sensitivity), nhưng phải đặt tên đúng.
- Tốt hơn nữa: dùng dataset B như **nguồn mở rộng số JD để tăng power** (xem §5). Lưu ý chi phí: B có 93.733 dòng, trích xuất LLM toàn bộ là **bất khả thi trên CPU**; nhưng **baseline không-LLM (B0–B2) thì chạy được** → dùng B để kiểm tra lại kết luận RQ1/RQ2 ở quy mô lớn hơn với các phương pháp không cần view.

---

## 3. Lỗ hổng phương pháp (làm yếu khả năng "bảo vệ được kết luận")

### 3.1. Thiếu kiểm soát then chốt: rút gọn **không dùng LLM**

Plan có B2 (SBERT toàn văn) vs B3 (SBERT trên văn bản rút gọn = nối các view do LLM trích xuất) — đây là cặp so sánh tốt cho RQ1. Nhưng nó **không tách được hai giả thuyết cạnh tranh**:

- H1: *LLM trích xuất cấu trúc* mới là thứ tạo ra lợi ích.
- H2: *bất kỳ việc lọc nhiễu/rút gọn văn bản nào* cũng tạo ra lợi ích (kể cả regex tách mục, cắt 512 token đầu, hoặc chỉ lấy phần "Skills/Requirements").

Nếu không có baseline rút gọn **không-LLM** (ví dụ: tách mục bằng heading regex + TF-IDF/SBERT) thì kết luận trung thực nhất mà bài có thể nói chỉ là *"giảm nhiễu giúp ích"*, và toàn bộ phần LLM trở thành dư thừa về mặt khoa học. **Đây là baseline tôi khuyến nghị thêm trước cả B7.**

### 3.2. Baseline bị bỏ sót hoặc bị xếp nhầm vào danh sách cắt

| Baseline còn thiếu | Vì sao reviewer sẽ hỏi |
|---|---|
| **LLM zero-shot / LLM-as-judge** (đưa thẳng cặp CV–JD cho LLM chấm điểm) | Câu hỏi số 1 ở mọi hội đồng 2025–2026. Hiện nằm ở mục "cắt được" #6 → **phải nâng thành baseline chính**. Chi phí: chỉ cần chạy trên **một mẫu test** (~vài nghìn cặp) để có upper bound. |
| **Encoder mạnh hơn MiniLM-L6** (BGE-M3, E5-large, gte) | Toàn bộ lợi ích của multi-view có thể **biến mất** khi encoder mạnh hơn. Hiện nằm ở bước 12.4 "optional" → phải là **ablation bắt buộc** cho ít nhất 1 encoder mạnh. |
| **Cross-encoder trên văn bản view đã nối** | Để trả lời "liệu có cần đến kiến trúc multi-view, hay chỉ cần cross-encoder mạnh hơn + văn bản sạch". |
| **BM25** | Có trong `configs/base.yaml` nhưng **không có bước nào chạy** trong PLAN.md → config mồ côi. |

### 3.3. Chất lượng trích xuất bị đánh giá ở mức "nếu kịp" — trong khi nó là mắt xích sống còn

Nếu tỉ lệ view rỗng cao hoặc LLM bịa nội dung, **mọi kết luận phía sau vô nghĩa** — và đây là kịch bản xác suất cao nhất của toàn dự án. Bước 6.3 hiện là optional, chỉ 50 mẫu, một người chấm, người thứ hai "nếu có".

Sửa:

- **Bắt buộc**, và là **cổng chặn**: chất lượng trích xuất không đạt → dừng, sửa prompt, chạy lại (plan đã có tinh thần này ở cuối tuần 6, nhưng phải đưa lên thành tiêu chí định lượng trước: ví dụ "≥95% JSON hợp lệ sau 3 lần thử, ≤10% view rỗng/view").
- 50 mẫu với tỉ lệ ~0,9 cho CI 95% khoảng **±8–14%** → quá thô. Với chỉ 994 văn bản, **chấm tay toàn bộ 994** là khả thi (≈ 1–2 ngày công) và biến điểm yếu lớn nhất thành điểm mạnh ("chúng tôi định lượng chất lượng trích xuất trên **100%** văn bản").
- Có **≥2 người chấm độc lập** trên một mẫu con + báo **Cohen's κ / Krippendorff α**; nếu không có người thứ hai, ghi rõ trong Limitations **và** giảm mạnh mức độ khẳng định của kết luận.
- Thêm **gold schema test**: 100 văn bản với view chuẩn do người viết → đo precision/recall/F1 của trích xuất, không chỉ "chấm 0/1/2".

### 3.4. Rủi ro kỹ thuật cụ thể của Qwen3:4b qua Ollama (chưa được plan xử lý)

- **Chế độ suy luận (thinking)**: Qwen3 bật thinking mặc định; với Ollama phải đặt `think: false` (hoặc `/no_think`) thì mới ổn định khi yêu cầu JSON. Nếu không, model sẽ tiêu hàng nghìn token vào phần suy luận, dễ **cụt JSON** và tăng thời gian gấp nhiều lần. Plan chỉ nói "prompt JSON cố định + validator thử lại N lần" → cần thêm mục "cấu hình gọi model" (think=false, `format=json`, seed, keep_alive).
- **Ngữ cảnh mặc định của Ollama là 4096** token; config đặt `num_ctx: 8192` là đúng, nhưng phải **assert** nó thực sự được truyền (nhiều client bỏ qua option này) và **đo tỉ lệ văn bản vượt ngưỡng**. Kiểm tra nhanh: CV dài nhất ~3.134 từ ≈ 4.200 token → **vừa** 8.192, nhưng cộng prompt hệ thống + schema JSON thì **sát trần** → nên nâng lên 16.384 hoặc chunk theo mục.
- **Tính bất định**: `temperature: 0` **không** bảo đảm tái lập trên GPU/CPU khác nhau (thứ tự rút gọn số học). Bước 6.2 chạy lại 30 mẫu để đo độ ổn định là đúng, nhưng phải **lưu hash của model digest Ollama** (`ollama show --modelfile` / digest) ngay từ tuần 2, không phải tuần 14.
- **Tư duy đúng hướng nhưng thiếu**: cache theo `prompt_version` là tốt; cần thêm **cache key = hash(văn bản) + model digest + prompt hash + params** để tránh trộn kết quả giữa các lần đổi cấu hình (bản plan 8 tuần có lưu ý này, bản 14 tuần làm mất).

### 3.5. Vấn đề construct validity của benchmark (rủi ro với cả bài)

- TF-IDF thuần đã đạt **NDCG@5 = 0,657 / MRR = 0,763** → bài toán phần lớn là **trùng khớp từ vựng/ngành nghề**, không phải suy luận về mức độ phù hợp.
- **50% nhãn là No Fit** và **59% JD không có Good Fit nào** → cấu trúc dữ liệu giống "ghép CV với JD cùng ngành hay không" hơn là "đánh giá mức phù hợp tinh tế".
- Nguồn nhãn **không được tài liệu hoá** (không có dataset card) → **không thể khẳng định nhãn phản ánh đánh giá của chuyên gia tuyển dụng**. Đây là đe doạ validity lớn nhất của cả bài và **phải nằm ở Limitations**, không phải một dòng phụ.

Đề xuất biến rủi ro thành đóng góp:

1. Tạo **tập con "hard"**: dùng cột `resume_domain` / `jd_domain` của `med2425` (hoặc tự gán domain bằng LLM) để **giới hạn pool ứng viên trong cùng ngành** → khi đó tín hiệu từ vựng biến mất và chỉ còn khác biệt tinh tế (Potential vs Good). Báo cáo NDCG trên cả "easy pool" và "hard pool". Đây là **đóng góp benchmark** có thể publish được, và cũng là cách trung thực nhất để chứng minh multi-view có ích.
2. Báo cáo **NDCG của B1 theo từng mức nhãn** và **NDCG trên tập con Potential-vs-Good** (chỉ giữ CV có nhãn ≥1, bỏ No Fit) → đo khả năng phân biệt tinh tế, thứ mà lexical model kém nhất.

### 3.6. Chi tiết kỹ thuật khác còn thiếu

- **Hàm mất mát của B4 (bi-encoder fine-tune) không được định nghĩa ở đâu cả** (MSE trên nhãn 0/1/2? MNRL? MarginMSE?). Đây là yếu tố quyết định độ mạnh của baseline B4; "chốt ở tuần 4" là quá muộn vì nó ảnh hưởng đến toàn bộ kết luận.
- **B6 fine-tune trên nhãn 0/1/2** cũng chưa nói là regression, ordinal, hay cross-entropy 3 lớp.
- **Chưa có quy tắc cho JD có < K ứng viên**: min CV/JD = **1**. NDCG@10 với 1 ứng viên là tầm thường; cần chốt lọc JD có ≥ K ứng viên (K=5 → còn 256/280 JD, K=10 → chỉ 175/280).
- **R@K (strict, chỉ Good Fit) chỉ tính được trên 143/351 JD (41%)** → power cực thấp và dễ gây hiểu nhầm. Phải ghi rõ "n_jd" cho từng metric trong mọi bảng.
- **Chuẩn hoá thang điểm trước khi fusion**: cosine SBERT nằm trong [−1,1], logits cross-encoder không cùng thang. Với B7/adaptive (MLP nhận 9 tín hiệu) nên chuẩn hoá (z-score trên train hoặc rank-based) — hiện không có bước nào.
- **Kiểm tra đa cộng tuyến giữa 9 tín hiệu** (ma trận tương quan) — bước 7.3 chỉ tính tương quan với nhãn. Nếu 9 tín hiệu tương quan >0,9 thì "fusion học được" gần như không thể thắng "trung bình cộng", và H3 sẽ bị bác bỏ một cách máy móc. Phải đo **trước** khi đầu tư vào adaptive.
- **Mask + view rỗng = tín hiệu 0** có thể trở thành **đặc trưng rò rỉ**: "view rỗng" tương quan với chất lượng văn bản/độ dài CV, mà độ dài CV lại tương quan với nhãn. Cần ablation "bỏ mask" để kiểm tra.
- **`chunk_mean` với 4 chunk × 384 token = 1.536 token** trong khi p90 CV ≈ 1.153 từ (~1.500 token) → **sát ngưỡng**; JD test có 39% vượt 512 từ. Phải báo **tỉ lệ mẫu bị cắt ở cả CV và JD** (plan mới chỉ yêu cầu cho B2), và thử `max_chunks: 8` như một ablation.
- **Expected NDCG cho tie**: đúng hướng, nhưng tie sẽ **rất nhiều** khi tín hiệu bị zero hoá (view rỗng) và với TF-IDF thưa. Cần định nghĩa tie-handling cho **P@K/R@K/MRR** nữa (hiện chỉ nói cho NDCG). Ngoài ra `sklearn.metrics.ndcg_score(ignore_ties=False)` chính là expected NDCG → dùng được, nhưng phải viết test đối chiếu riêng cho trường hợp tie.

---

## 4. Lỗi/mâu thuẫn cụ thể trong PLAN.md (sửa nhanh, mất 1 giờ)

| # | Vị trí | Vấn đề | Sửa |
|---|---|---|---|
| 1 | Bảng "Cập nhật research_log" vs bước 4.7 | Bảng phân công lần 3 = tuần 3 (tiêu chí cổng), lần 4 = tuần 4 (δ). Nhưng **bước 4.7 ghi "log cập nhật lần 3"** → mâu thuẫn, và tuần 4 không được đánh số | Đánh số lại: lần 3 = 3.7, lần 4 = 4.6, lần 5 = 9.2; bổ sung mốc log cho tuần 5 (5.3), 6 (6.5), 12 (12.5) |
| 2 | 4.5 vs 5.1 vs config | Số seed không nhất quán: "5 seed trên fold 1" (4.5) → "3 seed × 5 fold" (5.1) → config `b6 seeds: [42..46]`; bảng chính "3 seed (5 nếu kịp)" | Chốt **một** quy tắc: đề xuất 5 seed cho mọi mô hình học được (chi phí chấp nhận được vì mô hình nhỏ + dữ liệu nhỏ) |
| 3 | "Thứ tự cắt nếu trễ" #1 | Liệt kê **cross-attention view-level** để cắt, nhưng **PLAN.md 14 tuần không có bước nào làm cross-attention** (chỉ có ở bản 8 tuần cũ). `configs/base.yaml` vẫn có khối `fusion.attention` | Hoặc thêm bước, hoặc xoá khỏi danh sách cắt + xoá config mồ côi |
| 4 | Danh sách cắt #6 | "BM25, LLM zero-shot" để cắt, nhưng **không có bước nào chạy BM25** và LLM zero-shot không có trong B0–B7 | Chuyển LLM zero-shot **lên baseline chính**; thêm bước BM25 hoặc xoá |
| 5 | 2.8 vs config | 2.8 nói "định nghĩa relevant A (chỉ Good), B (Good+Potential)" chưa chốt, nhưng config đã đặt `primary: lenient` | Chốt ngay trong plan: **NDCG dùng relevance 0/1/2**; **P/R dùng B (lenient) làm chính**, A chỉ báo phụ (vì A chỉ dùng được 41% JD) |
| 6 | 2.3 + 2.9 | Không nói rõ **dùng split chính thức (6.241/1.759) hay gộp 8.000 dòng rồi GroupKFold 5 fold**. Điều này quyết định số JD cho kiểm định (71 vs 351) | Chốt: **gộp 8.000 dòng + GroupKFold 5 fold theo `job_id`**, dùng split chính thức làm **kiểm tra phụ**; ghi lý do (power) |
| 7 | 2.9 | Quy tắc "median < 15 → K = 3, 5" rơi **đúng vào lằn ranh** (median thật = **14,5**), và K=10 chỉ đo được trên **175/280** JD | Chốt K **bằng số JD còn đo được + power**: K chính = 5 (256 JD), báo thêm K=10 với ghi chú n_jd; hoặc chọn K=10 nếu ưu tiên power (σ nhỏ nhất) |
| 8 | 3.7 | Trường hợp "≥0,01 **nhưng** CI chồng lấn" không được định nghĩa | Bổ sung ô thứ ba trong bảng quyết định |
| 9 | 9.1 | Cổng quyết định dùng **chính val đã dùng để tune** → lạc quan có hệ thống | Dùng val của fold khác (nested) hoặc nêu rõ đây là ước lượng lạc quan |
| 10 | 10.2 vs tuần 11–12 | "Đánh giá test **một lần**" nhưng tuần 11–12 vẫn còn chuyển miền, ablation, phân tích | Thêm file **`docs/test_protocol.md`** khoá trước tuần 10: liệt kê **toàn bộ** cấu hình sẽ chạy trên test; mọi thứ phát sinh sau đó chỉ chạy trên val |
| 11 | 11.1 | Sai hướng non-inferiority (xem §2.A) | Sửa công thức + dùng kiểm định một phía |
| 12 | 5.4 | "Chỉnh prompt trên train của **fold 1**" nhưng sau đó chạy 5 fold → prompt được tune trên dữ liệu của fold 1, gây lạc quan cho fold 1 | Hoặc tune trên train của **từng fold**, hoặc cố định prompt và **báo riêng kết quả fold 1** như một kiểm tra |
| 13 | 2.6 | Ước tính thời gian trích xuất "toàn bộ dataset" — nên nói rõ **chỉ có 994 văn bản riêng biệt** (không phải 8.000), và cache theo văn bản | Ghi con số 994 vào plan; đặt ngân sách thời gian thực tế (~10–20 giờ CPU) |
| 14 | Thiếu bước | Không có bước **đo MDE/power** trước khi chốt cổng và δ | Thêm vào 3.7 và 4.6 (xem §2.A) |
| 15 | Thiếu bước | Không có bước **leakage audit** định lượng (chỉ có checklist) | Thêm vào 2.10: in bảng 3 con số (JD chung / CV chung / cặp chung) |
| 16 | Tuần 13.5 | Limitations chỉ xuất hiện ở tuần 13 — nhưng các limitation lớn (nguồn nhãn, rò rỉ, đơn ngữ) đã biết từ tuần 2 | Viết **`docs/limitations.md` từ tuần 3**, cập nhật dần; đây cũng là tài liệu để trả lời phản biện |

---

## 5. Thiếu sót ở tầng "bài NCKH" (không phải tầng kỹ thuật)

Đây là nhóm thiếu sót khiến plan hiện tại **chưa thể nộp như một đề cương NCKH**:

1. **Không có tuyên bố đóng góp (contribution statement).** Trong plan không có một câu nào dạng "bài này đóng góp X, khác với [công trình cụ thể] ở Y". Ở bước 1.4 ghi "3 RQ, 4 giả thuyết, tiêu chí bác bỏ" — nhưng **chúng không có trong plan** để người đọc/giảng viên đánh giá. → Viết ngay §1.4 thành nội dung thật trong plan (3 RQ + 4 H + tiêu chí bác bỏ), không chỉ là "sẽ ghi vào log".
2. **Nguy cơ novelty thấp**: "LLM trích xuất view + so khớp theo cặp view + fusion học được" là tổ hợp của ba ý đã có trong literature (ConFit dùng multi-view; các bài re-ranking dùng cross-encoder + supervised fusion). Plan có bước 1.3 kiểm tra trùng lặp nhưng **không có phương án B nếu trùng**. → Chuẩn bị trước 3 hướng tái định vị (dưới đây).
3. **Không có mốc/ngày thực tế.** Plan chỉ có "tuần 1..14", không có ngày bắt đầu, không có buffer, không có mốc đăng ký/thuyết minh/báo cáo tiến độ/bảo vệ. NCKH có deadline cứng. → Thêm cột ngày thực + 1 tuần đệm trước hạn nộp.
4. **Không có deliverable "đề cương/thuyết minh"** và **không có slide/poster/bảo vệ**. Bản plan 8 tuần cũ có mục "làm slide 12–15 trang + chuẩn bị câu hỏi hội đồng"; **bản 14 tuần đã bỏ mất**. → Thêm tuần 13.7: slide + 15 câu hỏi phản biện dự kiến; tuần 14: tổng duyệt bảo vệ.
5. **Chưa xác định nơi nộp/định dạng** (giới hạn trang, template, ngôn ngữ, citation style, tỉ lệ trùng lặp). Bước 14.5 mới xem là quá muộn. → Xác định ở tuần 1.
6. **Không có kế hoạch tính toán/ngân sách.** Không có GPU (§2.G) và không có phương án (Colab/Kaggle/thuê GPU/API). Cần bảng: việc nào chạy CPU, việc nào cần GPU, tốn bao nhiêu giờ, dự phòng là gì.
7. **Không có đánh giá con người.** Chỉ có metric tự động. Một đánh giá nhỏ (2–3 người tuyển dụng/HR chấm 30–50 cặp, so với mô hình) sẽ làm bài mạnh hơn nhiều và là điểm cộng lớn ở NCKH ứng dụng.
8. **Không có ethics/PII.** CV là dữ liệu cá nhân (dù tổng hợp): cần chính sách lưu trữ, ẩn danh, không đưa nguyên văn CV vào bài/phụ lục (bước 12.7 ẩn tên chỉ là "nếu làm được"), và **model license** (MiniLM/MS MARCO MiniLM/Qwen3 đều cần ghi nguồn + license).
9. **License dataset chính không tồn tại** (không có tag license, không có card) → **không thể "kiểm tra license"** như bước 2.4 yêu cầu. Phải ghi rõ: *"dataset không công bố license; chúng tôi chỉ dùng cho mục đích nghiên cứu, không phân phối lại văn bản, không trích nguyên văn ví dụ"* + **liên hệ tác giả xin xác nhận** (việc này nên làm ở tuần 1, và ghi lại ngày/nội dung trao đổi).
10. **Related work quá mỏng và không có giao thức tìm kiếm.** Chỉ 6 tài liệu được nêu tên. Bước 1.3 "tìm công trình đủ 3 thành phần" không có từ khoá, cơ sở dữ liệu, tiêu chí include/exclude, ngày tìm. → Viết protocol ngắn (kiểu PRISMA mini), nhắm **25–35 tài liệu**, bắt buộc có bài 2024–2026 về LLM resume matching/judge.
11. **Không có ma trận rủi ro** (chỉ có "thứ tự cắt"). Cần xác suất × mức ảnh hưởng × phương án cho: chất lượng trích xuất thấp, không có GPU, phương pháp không thắng baseline, không tìm được người chấm thứ hai, dataset bị gỡ.
12. **Tái lập chưa đủ chặt**: chưa có lock file (chỉ pin version), chưa pin **revision** của dataset/model HF, chưa pin **digest** model Ollama, chưa có Dockerfile/environment.yml, chưa có script một lệnh chạy lại toàn bộ.
13. **Khoảng cách với bối cảnh sản phẩm (Frevia)**: dataset và mô hình **hoàn toàn tiếng Anh**, trong khi sản phẩm nhắm người dùng Việt Nam. Plan không có một dòng nào về tiếng Việt, không có kiểm tra chuyển ngữ, không có dữ liệu Việt. Hội đồng sẽ hỏi ngay. → Hoặc (a) nêu rõ đây là nghiên cứu phương pháp trên benchmark tiếng Anh, kết quả chuyển giao là hướng phát triển; hoặc (b) thêm một thí nghiệm nhỏ tiếng Việt (20–50 CV/JD tự thu thập, chỉ chạy zero-shot) như bằng chứng khả thi.
14. **Không có artifact/demo** (nếu NCKH gắn với sản phẩm): một CLI/notebook tái lập + ảnh demo xếp hạng sẽ được điểm.

---

## 6. Ba phương án tái định vị đề tài (chọn 1 trước khi viết tuần 3)

Plan hiện tại đặt cược vào việc **adaptive fusion > static > SBERT** với chênh lệch kỳ vọng nhỏ (0,01–0,05) trên **305 JD** → xác suất "kết quả không kết luận được" là **cao**. Ba hướng giảm rủi ro:

**Phương án A — Giữ nguyên phương pháp, nâng độ chặt chẽ (an toàn nhất, ít novelty nhất).**
Đóng góp = *"đánh giá có kiểm soát, tiền đăng ký (pre-registered), có kiểm toán rò rỉ, cho multi-view LLM extraction trong person–job fit"*. Bổ sung: baseline rút gọn không-LLM, LLM judge, encoder mạnh, split CV-disjoint, power analysis. Nếu kết quả là âm (multi-view ≈ SBERT), **bài vẫn nộp được** với thông điệp "khi nào multi-view KHÔNG giúp" — miễn là có power và có phân tích. **Đây là phương án tôi khuyên** nếu deadline gần.

**Phương án B — Chuyển trọng tâm sang benchmark & đo lường (novelty rõ nhất, hợp NCKH).**
Đóng góp = *"hai đánh giá lại benchmark person–job fit phổ biến: (i) đo và định lượng rò rỉ CV (99,8%) và ảnh hưởng của nó lên các mô hình fine-tune; (ii) xây tập con 'hard' cùng-ngành nơi tín hiệu từ vựng bị vô hiệu; (iii) chỉ ra trần hiệu năng thực của họ phương pháp multi-view"*. Chi phí thấp (không cần LLM nhiều), mở rộng được sang `med2425` (93k dòng) → **giải quyết luôn vấn đề power**. Vẫn giữ được phần multi-view như một hệ thống được đánh giá.

**Phương án C — Chuyển sang chi phí/hiệu quả (phù hợp sản phẩm).**
Đóng góp = *"so sánh Pareto giữa chi phí suy luận (LLM extraction một lần + xếp hạng) và chất lượng, so với cross-encoder fine-tune"*. Với chỉ **994 văn bản** cần trích xuất, câu chuyện "chi phí biên gần bằng 0 khi thêm ứng viên" là một luận điểm mạnh và **đo được** — và tránh được việc phải thắng B6 về NDCG.

---

## 7. Patch đề xuất cho PLAN.md (theo tuần, tối thiểu mà đủ)

| Tuần | Bổ sung/sửa |
|---|---|
| **1** | Xác định nơi nộp + template + deadline cứng. Viết **contribution statement + 3 RQ + 4 H + tiêu chí bác bỏ** vào plan (không chỉ vào log). Viết protocol tìm tài liệu (từ khoá/CSDL/ngày) nhắm 25–35 refs. Liên hệ tác giả dataset xin xác nhận license. Thêm 3–5 tài liệu về LLM-as-judge & dense retriever mạnh. |
| **2** | Thêm bước **EDA bắt buộc** (§1 tài liệu này — số đã có sẵn, chỉ cần xác nhận lại). **Chốt split: primary job-disjoint (GroupKFold 5), secondary resume-disjoint.** Ghi rõ gộp 8.000 dòng. Ghi rõ 994 văn bản cho trích xuất. Thêm **leakage audit** định lượng. Pin model digest Ollama + dataset revision. Đo thời gian 1 văn bản và suy ra tổng. |
| **3** | Thêm **B1b: rút gọn không-LLM** (regex heading). Thêm **đo MDE/power** và in ra trước khi chốt cổng. Sửa cổng 3.7: SESOI = 3×MDE (~0,05), 0,01 chỉ là ngưỡng "bỏ qua". Chốt K chính bằng số JD đo được (đề xuất K=5, báo thêm K=10 với n_jd). Viết `docs/limitations.md` v1. |
| **4** | Sửa δ: **δ = max(1 std_seed, SESOI)**, và **ghi kèm MDE**. Viết rõ loss cho B4/B6 (MSE/ordinal/CE — chốt ở đây, không để trôi). Thêm **B8: LLM zero-shot judge** trên ~2.000 cặp test (upper bound). Bổ sung cột "GPU/hour" cho mọi baseline trong `baselines_meta.csv`. |
| **5–6** | **Bắt buộc** đánh giá chất lượng trích xuất, **chấm 100% (994 văn bản)**, ≥2 người chấm + κ. Thêm gate định lượng (đạt/không đạt) trước khi sang tuần 7. Thêm gold schema test. |
| **7** | Thêm **ma trận tương quan 9 tín hiệu** và **chuẩn hoá tín hiệu** trước fusion. Thêm ablation **bỏ mask**. Báo tỉ lệ cắt ở cả CV và JD, thử `max_chunks` 4 vs 8. |
| **8–9** | Cổng quyết định dùng **SESOI** đã sửa; thêm ô "≥SESOI nhưng CI chồng lấn". Ghi rõ cổng dùng val lạc quan. |
| **10** | Viết **`docs/test_protocol.md`** khoá danh sách cấu hình chạy test **trước** khi chạy. Kiểm định: cluster bootstrap theo `job_id`, báo effect size + CI, hiệu chỉnh Holm **trên tất cả so sánh chính** (không chỉ giữa các baseline). |
| **11** | Sửa 11.1 (dấu + kiểm định một phía). Sửa 11.3: dùng cột `source`, hoặc đổi tên thành "độ nhạy hàm nhãn"; hoặc dùng `med2425` để **tăng số JD** cho các baseline không-LLM. Thêm **encoder mạnh (BGE-M3/E5)** — nâng từ "optional" lên bắt buộc. |
| **12** | Thêm **tập con hard (cùng domain)** và **NDCG trên tập Potential-vs-Good**. Thêm **so sánh với đánh giá của con người** (2–3 người chấm 30–50 cặp). |
| **13** | Thêm **slide + 15 câu hỏi phản biện dự kiến** + phụ lục tái lập. Sinh bảng/hình từ file kết quả (đã có 13.6). |
| **14** | Thêm **environment lock/Dockerfile**, pin dataset revision + Ollama digest, script một lệnh tái lập toàn bộ, và **tổng duyệt bảo vệ**. |

---

## 8. Cần bạn xác nhận để mình chốt patch

1. **Loại NCKH và hạn nộp**: cấp trường / Euréka / hội nghị? Hạn nộp và hạn báo cáo tiến độ cụ thể là ngày nào?
2. **Phần cứng**: có GPU dùng được không (máy cá nhân, phòng lab, Colab/Kaggle)? Nếu chỉ CPU, mình sẽ viết lại tuần 4–5 theo ngân sách CPU.
3. **Được dùng API trả phí không** (cho baseline LLM judge / encoder mạnh)? Ngân sách khoảng bao nhiêu?
4. **Số người tham gia** và **có người chấm thứ hai** cho phần gán nhãn/chất lượng trích xuất không?
5. **Bài nộp bằng tiếng Việt hay tiếng Anh**, giới hạn trang?

## 9. Việc nên làm ngay (thứ tự)

1. Chốt §8 (30 phút) → khóa phạm vi và định dạng.
2. Viết contribution statement + 3 RQ + 4 H vào `PLAN.md` (1–2 giờ).
3. Chốt split (job-disjoint + resume-disjoint) và viết `scripts/make_splits.py` + leakage audit (nửa ngày).
4. Thêm baseline rút gọn không-LLM và đo MDE ở tuần 3 — hai thứ này quyết định toàn bộ phần sau.
5. Sửa 16 mục ở §4 (1 giờ) và bổ sung các mục §7.

---

## Phụ lục — Cách đo (để tái lập)

Chạy: `python scripts/quick_eda.py` và `python scripts/probe_b1_tfidf.py` (đã commit trong repo).

- **EDA** (`scripts/quick_eda.py`): `load_dataset("cnamuangtoun/resume-job-description-fit")`, gộp train+test = 8.000 dòng; `job_id`/`resume_id` = SHA1(văn bản)[:16], đúng như bước 2.5 của plan. Đếm JD/CV riêng biệt, phân bố CV mỗi JD, phân bố nhãn, giao giữa hai split.
- **Probe B1**: gộp 8.000 dòng → `GroupKFold(n_splits=5)` theo `job_id` → `TfidfVectorizer(ngram_range=(1,2), max_features=200k, min_df=2, sublinear_tf=True)` **fit trên văn bản của train**, cosine từng cặp, xếp hạng CV trong từng JD, NDCG relevance 0/1/2, macro-average theo JD, **bỏ JD không có nhãn >0** (đúng quy tắc `ndcg_empty_jobs: drop` trong config).
- **Lưu ý về độ chính xác**: probe dùng `np.argsort(kind="stable")` cho tie (plan dùng expected NDCG) và chỉ chạy 1 cấu hình TF-IDF (plan tune trên val) → **NDCG@5 = 0,657 là ước lượng tốt nhưng có thể lệch nhẹ**. Con số quan trọng cho lập luận là **σ giữa các JD = 0,288** và **số JD = 305**, hai đại lượng này ổn định và không phụ thuộc cách xử lý tie.
- **Power**: MDE (2 phía, power 80%, α=0,05) = 2,80 × σ_d/√N; σ_d là độ lệch chuẩn của **hiệu số theo cặp** giữa hai phương pháp trên cùng một JD (chưa đo được ở giai đoạn này, nên báo cáo theo 2 kịch bản σ_d = σ_jd và σ_d = 0,5·σ_jd). Khi có B0/B1 ở tuần 3, **đo σ_d thật** và thay vào công thức này.

