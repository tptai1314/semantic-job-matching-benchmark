"""Pipeline dữ liệu: tải -> chuẩn hoá -> gán id -> chia fold -> audit rò rỉ."""

from __future__ import annotations

from typing import Any

import pandas as pd

from src.config import get
from src.data.clean import clean_pairs
from src.data.folds import (
    attach_fold,
    audit_leakage,
    build_group_folds,
    split_summary,
    write_folds,
)
from src.data.load import dataset_fingerprint, load_raw
from src.data.schemas import build_pairs

PARQUET_NAME = "pairs.parquet"


def processed_path(cfg: dict[str, Any]) -> str:
    from pathlib import Path

    return str(Path(get(cfg, "paths.processed")) / PARQUET_NAME)


def prepare_pairs(cfg: dict[str, Any], force: bool = False) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Bước 2.3-2.5: nạp, chuẩn hoá và lưu bảng cặp đã làm sạch."""
    import json
    from pathlib import Path

    out = Path(processed_path(cfg))
    report_path = out.with_name("cleaning_report.json")
    if out.exists() and not force:
        return pd.read_parquet(out), json.loads(report_path.read_text(encoding="utf-8"))

    raw = load_raw(cfg)
    fingerprint = dataset_fingerprint(cfg, raw)

    relevance = {0: 0, 1: 1, 2: 1}
    pairs = build_pairs(
        resume_text=raw["resume_text"],
        job_text=raw["job_text"],
        label_raw=raw["label"],
        label_map=get(cfg, "data.label_map"),
        relevance=relevance,
        source_split=raw["source_split"],
        hashing_cfg=get(cfg, "data.hashing"),
    )
    pairs, clean_report = clean_pairs(pairs)
    clean_report["fingerprint"] = fingerprint

    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out, index=False)
    report_path.write_text(json.dumps(clean_report, indent=2), encoding="utf-8")
    return pairs, clean_report


def build_all_splits(cfg: dict[str, Any]) -> dict[str, Any]:
    """Bước 2.9-2.10: dựng hệ split chính theo job_id và hệ phụ theo resume_id."""
    import json
    from pathlib import Path

    pairs, clean_report = prepare_pairs(cfg)
    splits_dir = Path(get(cfg, "paths.splits"))
    tables_dir = Path(get(cfg, "paths.tables"))

    split_cfg = get(cfg, "data.split")
    n_splits = int(split_cfg["n_splits"])
    seed = int(split_cfg.get("seed", 42))

    systems = [split_cfg.get("by", "job_id")]
    secondary = split_cfg.get("secondary") or {}
    if secondary.get("enabled", False):
        systems.append(secondary.get("by", "resume_id"))

    audits: list[pd.DataFrame] = []
    report: dict[str, Any] = {
        "cleaning": clean_report,
        "split_systems": {},
    }
    combined = pairs.copy()
    primary_col = split_cfg.get("by", "job_id")

    for group_col in systems:
        folds = build_group_folds(pairs, group_col, n_splits, seed)
        audit = audit_leakage(folds, group_col)
        audits.append(audit)

        is_primary = group_col == primary_col
        prefix = "fold_" if is_primary else secondary.get("prefix", f"fold_{group_col}_")
        target_col = "fold" if is_primary else f"fold_{group_col}"

        write_folds(folds, splits_dir, prefix=prefix)
        audit.to_csv(tables_dir / f"leakage_audit_{group_col}.csv", index=False)

        combined = attach_fold(combined, folds, fold_col=target_col)
        summary = split_summary(folds, group_col)
        summary["fold_column"] = target_col
        summary["files"] = [f"{prefix}{i}.json" for i in range(n_splits)]
        report["split_systems"][group_col] = summary

    combined.to_parquet(processed_path(cfg).replace(".parquet", "_with_folds.parquet"), index=False)

    pd.concat(audits, ignore_index=True).to_csv(tables_dir / "leakage_audit.csv", index=False)
    (splits_dir / "split_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["combined"] = combined
    return report
