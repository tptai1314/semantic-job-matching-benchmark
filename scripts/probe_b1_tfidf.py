"""Probe B1 (TF-IDF + cosine) + đo phương sai theo JD — dùng để chốt ngưỡng cổng quyết định và biên δ.

Mục đích: PLAN.md bước 3.7/4.6 đặt ngưỡng "0,01 NDCG" và δ = max(1 std, 0,01) mà không có cơ sở.
Script này (a) chạy baseline lexical đúng như quy tắc trong configs/base.yaml
(GroupKFold 5 fold theo job_id, NDCG relevance 0/1/2, macro theo JD, bỏ JD không có relevant),
và (b) in ra σ giữa các JD + MDE để chốt SESOI bằng số liệu.

Chạy: python scripts/probe_b1_tfidf.py
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import GroupKFold

N_SPLITS = 5


def dcg(rels: np.ndarray) -> float:
    rels = np.asarray(rels, dtype=float)
    return float(np.sum((2**rels - 1) / np.log2(np.arange(2, len(rels) + 2))))


def ndcg_at_k(rels_ranked: np.ndarray, rels_ideal: np.ndarray, k: int) -> float:
    idcg = dcg(np.sort(rels_ideal)[::-1][:k])
    return dcg(np.asarray(rels_ranked)[:k]) / idcg if idcg > 0 else np.nan


def main() -> None:
    ds = load_dataset("cnamuangtoun/resume-job-description-fit")
    df = pd.concat([ds["train"].to_pandas(), ds["test"].to_pandas()], ignore_index=True)
    df["label_id"] = df["label"].map({"No Fit": 0, "Potential Fit": 1, "Good Fit": 2})
    y = df["label_id"].to_numpy()
    groups = df["job_description_text"].factorize()[0]
    print(f"rows={len(df)} unique_JD={len(set(groups))} unique_CV={df['resume_text'].nunique()}")

    metrics = {"ndcg@3": [], "ndcg@5": [], "ndcg@10": [], "p@5": [], "r@5": [], "mrr": []}
    gkf = GroupKFold(n_splits=N_SPLITS)
    for tr, va in gkf.split(df, y, groups):
        vec = TfidfVectorizer(ngram_range=(1, 2), max_features=200_000, min_df=2, sublinear_tf=True)
        vec.fit(pd.concat([df.iloc[tr]["job_description_text"], df.iloc[tr]["resume_text"]]))
        v_job = vec.transform(df.iloc[va]["job_description_text"])
        v_res = vec.transform(df.iloc[va]["resume_text"])
        # cả hai ma trận đã chuẩn hoá L2 theo hàng -> tích từng hàng = cosine của cặp tương ứng
        val = df.iloc[va].copy()
        val["score"] = np.asarray(v_job.multiply(v_res).sum(axis=1)).ravel()

        for _, g in val.groupby("job_description_text"):
            rels = g["label_id"].to_numpy()
            if (rels > 0).sum() == 0:  # quy tắc ndcg_empty_jobs: drop
                continue
            ranked = rels[np.argsort(-g["score"].to_numpy(), kind="stable")]
            for k in (3, 5, 10):
                metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked, rels, k))
            top5 = ranked[:5]
            n_good = int((rels >= 2).sum())
            metrics["p@5"].append(float((top5 >= 2).mean()))
            metrics["r@5"].append(float((top5 >= 2).sum() / n_good) if n_good else np.nan)
            hits = np.flatnonzero(ranked >= 1)
            metrics["mrr"].append(float(1 / (hits[0] + 1)) if len(hits) else 0.0)

    for name, values in metrics.items():
        v = np.asarray(values, dtype=float)
        v = v[~np.isnan(v)]
        sigma = float(v.std(ddof=1))
        se = sigma / np.sqrt(len(v))
        print(f"{name:8s} mean={v.mean():.4f} sigma_jd={sigma:.4f} n_jd={len(v)} "
              f"se={se:.4f} ci95=+/-{1.96 * se:.4f}")

    n = len(metrics["ndcg@5"])
    sigma = float(np.std(metrics["ndcg@5"], ddof=1))
    print("\n--- power (2 phía, alpha=0.05, power=80% -> 2.80 * sigma_d / sqrt(n)) ---")
    for sigma_d in (sigma, 0.75 * sigma, 0.5 * sigma):
        se = sigma_d / np.sqrt(n)
        print(f"sigma_d={sigma_d:.3f}: se={se:.4f} ci95=+/-{1.96 * se:.4f} mde={2.80 * se:.4f}")
    for target in (0.01, 0.02, 0.03, 0.05):
        for sigma_d in (sigma, 0.5 * sigma):
            print(f"phát hiện {target} với sigma_d={sigma_d:.3f} -> cần n_jd={(2.80 * sigma_d / target) ** 2:.0f}")


if __name__ == "__main__":
    main()
