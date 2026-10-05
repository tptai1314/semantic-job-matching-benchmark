"""EDA nhanh để kiểm chứng các giả định của PLAN.md tuần 2 (K, số JD, rò rỉ CV, khối lượng trích xuất).

Trả lời trực tiếp các câu hỏi mà plan đang để ngỏ:
  - Bước 2.7: số cặp / JD / CV, phân bố CV mỗi JD, phân bố nhãn, JD không có Good Fit, độ dài văn bản.
  - Bước 2.9: median CV mỗi JD -> quy tắc K; số JD đủ điều kiện cho từng K.
  - Bước 2.10: rò rỉ JD / CV / cặp giữa hai split chính thức.
  - Bước 2.6: khối lượng trích xuất LLM thật = số VĂN BẢN riêng biệt (đã cache theo id), không phải số cặp.

Chạy: python scripts/quick_eda.py
"""
from __future__ import annotations

import hashlib
import json
from collections import Counter

import pandas as pd
from datasets import load_dataset

HASH_LEN = 16  # khớp configs/base.yaml: data.hashing


def _doc_id(text: str) -> str:
    return hashlib.sha1(str(text).encode("utf-8")).hexdigest()[:HASH_LEN]


def _describe(df: pd.DataFrame, c_res: str, c_job: str, c_lab: str) -> dict:
    df = df.copy()
    df["resume_id"] = df[c_res].map(_doc_id)
    df["job_id"] = df[c_job].map(_doc_id)
    lab = df[c_lab].astype(str).str.strip()

    per_jd = df.groupby("job_id").size()
    good = set(df[lab.str.contains("Good", case=False)]["job_id"])
    pot = set(df[lab.str.contains("Potential", case=False)]["job_id"])
    n_jd = int(df["job_id"].nunique())
    w_res = df[c_res].astype(str).str.split().str.len()
    w_job = df[c_job].astype(str).str.split().str.len()

    return {
        "rows": int(len(df)),
        "unique_jd": n_jd,
        "unique_resume": int(df["resume_id"].nunique()),
        "label_counts": dict(Counter(lab).most_common()),
        "label_pct": {k: round(100 * v / len(df), 2) for k, v in Counter(lab).most_common()},
        "cv_per_jd": {
            "min": int(per_jd.min()),
            "p25": float(per_jd.quantile(0.25)),
            "median": float(per_jd.median()),
            "p75": float(per_jd.quantile(0.75)),
            "max": int(per_jd.max()),
            "mean": round(float(per_jd.mean()), 2),
        },
        "jds_with_at_least": {f"K={k}": int((per_jd >= k).sum()) for k in (3, 5, 10)},
        "jd_without_good": n_jd - len(good),
        "jd_no_relevant_lenient": n_jd - len(good | pot),
        "words_resume": {
            "median": float(w_res.median()),
            "p90": float(w_res.quantile(0.9)),
            "max": int(w_res.max()),
            "pct_over_512_words": round(100 * float((w_res > 512).mean()), 2),
        },
        "words_job": {
            "median": float(w_job.median()),
            "p90": float(w_job.quantile(0.9)),
            "max": int(w_job.max()),
            "pct_over_512_words": round(100 * float((w_job > 512).mean()), 2),
        },
        "n_unique_pairs": int(df.drop_duplicates(subset=["job_id", "resume_id"]).shape[0]),
        "conflicting_pair_labels": int((df.groupby(["job_id", "resume_id"])[c_lab].nunique() > 1).sum()),
        "resumes_appearing_more_than_once": int((df.groupby("resume_id").size() > 1).sum()),
    }


def main() -> None:
    ds = load_dataset("cnamuangtoun/resume-job-description-fit")
    cols = ds["train"].column_names
    c_res = next(c for c in cols if "resume" in c.lower())
    c_job = next(c for c in cols if "job" in c.lower())
    c_lab = next(c for c in cols if "label" in c.lower() or "fit" in c.lower())
    print(f"columns={cols} -> {c_res} | {c_job} | {c_lab}")

    tr = ds["train"].to_pandas()
    te = ds["test"].to_pandas()
    report = {name: _describe(df, c_res, c_job, c_lab) for name, df in (("train", tr), ("test", te))}

    tr_ids = (tr[c_job].map(_doc_id), tr[c_res].map(_doc_id))
    te_ids = (te[c_job].map(_doc_id), te[c_res].map(_doc_id))
    report["overlap"] = {
        "jd_shared_train_test": len(set(tr_ids[0]) & set(te_ids[0])),
        "resume_shared_train_test": len(set(tr_ids[1]) & set(te_ids[1])),
        "resume_shared_pct_of_test": round(
            100 * len(set(tr_ids[1]) & set(te_ids[1])) / te_ids[1].nunique(), 2
        ),
        "pair_shared_train_test": len(set(zip(*tr_ids)) & set(zip(*te_ids))),
    }
    report["llm_workload"] = {
        "unique_docs_train": int(tr_ids[1].nunique() + tr_ids[0].nunique()),
        "unique_docs_test": int(te_ids[1].nunique() + te_ids[0].nunique()),
        "unique_docs_total": int(
            len(set(tr_ids[1]) | set(te_ids[1])) + len(set(tr_ids[0]) | set(te_ids[0]))
        ),
        "total_rows": int(len(tr) + len(te)),
    }
    report["llm_workload"]["compression_factor"] = round(
        report["llm_workload"]["total_rows"] / report["llm_workload"]["unique_docs_total"], 2
    )

    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
