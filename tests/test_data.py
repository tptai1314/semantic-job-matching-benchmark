"""Test cho pipeline dữ liệu: chuẩn hoá, id ổn định, fold không rò nhóm."""

from __future__ import annotations

import pandas as pd
import pytest

from src.data.clean import clean_pairs, normalize_text
from src.data.folds import audit_leakage, build_group_folds, split_summary
from src.data.schemas import build_pairs


def _fake_pairs(n_jobs: int = 12, n_resumes: int = 8) -> pd.DataFrame:
    jobs = [f"Job description number {j} with skills and requirements text." for j in range(n_jobs)]
    resumes = [
        f"Resume number {r} with experience and education section text." for r in range(n_resumes)
    ]
    rows = []
    for j, job in enumerate(jobs):
        for r, resume in enumerate(resumes):
            rows.append((resume, job, ["No Fit", "Potential Fit", "Good Fit"][(j + r) % 3]))
    return pd.DataFrame(rows, columns=["resume_text", "job_text", "label"])


def test_normalize_text_keeps_newlines():
    text = "Summary\n\n\n\nSkills  Python\t\tSQL\x00"
    out = normalize_text(text)
    assert "\x00" not in out
    assert "  " not in out
    assert "\t" not in out
    # gộp 4 dòng trống thành đúng một dòng trống, KHÔNG dính các heading vào nhau
    assert out.count("\n") == 2
    assert "Skills Python SQL" in out


def test_text_id_is_content_based_and_stable():
    a = _fake_pairs()
    b = a.iloc[::-1].reset_index(drop=True)
    label_map = {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2}
    left = build_pairs(a["resume_text"], a["job_text"], a["label"], label_map, {0: 0, 1: 1, 2: 1})
    right = build_pairs(b["resume_text"], b["job_text"], b["label"], label_map, {0: 0, 1: 1, 2: 1})
    # id phai phu thuoc NOI DUNG van ban, khong phu thuoc thu tu dong
    assert set(left["pair_id"]) == set(right["pair_id"])
    assert left["pair_id"].nunique() == len(left)
    assert left["job_id"].nunique() == 12
    assert left["resume_id"].nunique() == 8


def test_build_pairs_rejects_unknown_label():
    df = pd.DataFrame(
        {"resume_text": ["a" * 60], "job_text": ["b" * 60], "label": ["Excellent Fit"]}
    )
    with pytest.raises(ValueError, match="label_map"):
        build_pairs(df["resume_text"], df["job_text"], df["label"], {"No Fit": 0}, {0: 0})


def test_clean_drops_duplicate_pairs():
    df = _fake_pairs()
    doubled = pd.concat([df, df.head(3)], ignore_index=True)
    pairs = build_pairs(
        doubled["resume_text"],
        doubled["job_text"],
        doubled["label"],
        {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2},
        {0: 0, 1: 1, 2: 1},
    )
    cleaned, report = clean_pairs(pairs)
    assert report["rows_dropped_duplicate_pair"] == 3
    assert len(cleaned) == report["rows_in"] - 3
    assert cleaned["pair_id"].is_unique


def test_clean_drops_short_text():
    df = _fake_pairs()
    df.loc[0, "resume_text"] = "qua ngan"
    pairs = build_pairs(
        df["resume_text"],
        df["job_text"],
        df["label"],
        {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2},
        {0: 0, 1: 1, 2: 1},
    )
    _, report = clean_pairs(pairs, min_chars=40)
    assert report["rows_dropped_short_text"] == 1


def test_group_folds_never_share_group_column():
    pairs = build_pairs_fake()
    folds = build_group_folds(pairs, "job_id", n_splits=4)
    audit = audit_leakage(folds, "job_id")
    assert (audit["job_id_shared_train_val"] == 0).all()
    assert sorted(folds["fold"].unique()) == [0, 1, 2, 3]
    assert folds["fold"].value_counts().sum() == len(folds)


def test_secondary_split_is_clean_on_its_own_axis():
    pairs = build_pairs_fake()
    folds = build_group_folds(pairs, "resume_id", n_splits=4)
    audit = audit_leakage(folds, "resume_id")
    assert (audit["resume_id_shared_train_val"] == 0).all()
    assert (audit["cv_shared_pct_of_val"] == 0.0).all()
    # nhưng JD vẫn trùng -> đây chính là lý do cần báo cáo cả hai hệ split
    assert audit["job_id_shared_train_val"].gt(0).all()


def test_build_group_folds_rejects_too_few_groups():
    pairs = build_pairs_fake()
    with pytest.raises(ValueError, match="nhom"):
        build_group_folds(pairs, "job_id", n_splits=99)


def test_split_summary_covers_every_fold():
    pairs = build_pairs_fake()
    folds = build_group_folds(pairs, "job_id", n_splits=4)
    summary = split_summary(folds, "job_id")
    assert summary["n_splits"] == 4
    assert len(summary["per_fold"]) == 4
    assert summary["rows_min"] <= summary["rows_max"]


def build_pairs_fake() -> pd.DataFrame:
    df = _fake_pairs()
    return build_pairs(
        df["resume_text"],
        df["job_text"],
        df["label"],
        {"No Fit": 0, "Potential Fit": 1, "Good Fit": 2},
        {0: 0, 1: 1, 2: 1},
    )
