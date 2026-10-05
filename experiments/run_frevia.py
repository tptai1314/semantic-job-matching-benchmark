"""Entry point chính của Frevia.

Dùng cho mọi giai đoạn: `python experiments/run_frevia.py <stage> --config configs/base.yaml`
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def _force_utf8() -> None:
    """Console Windows mặc định dùng GBK, gây lỗi khi in tiếng Việt."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            try:
                reconfigure(encoding="utf-8", errors="replace")
            except Exception:
                pass


_force_utf8()

from src.config import ensure_dirs, get, load_config  # noqa: E402
from src.utils import resolve_device, set_seed  # noqa: E402

STAGES = {
    "prepare": "Kiem tra moi truong, tao thu muc, ghi lai phien ban thu vien",
    "info": "In cau hinh dang dung (khong chay thi nghiem)",
    "dataset": "Tai + lam sach dataset, ghi data/processed/pairs.parquet",
    "folds": "Chia fold job_id + resume_id, ghi bang ro ri leak vao results/tables",
}


def git_hash(root: Path) -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, timeout=10
        )
        return out.stdout.strip() or None
    except Exception:
        return None


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_frevia.py",
        description="Frevia: multi-view LLM + SBERT cho resume-job matching.",
    )
    parser.add_argument("--config", default="configs/base.yaml", help="duong dan file yaml cau hinh")
    parser.add_argument("--seed", type=int, default=None, help="ghi de seed")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default=None)
    sub = parser.add_subparsers(dest="stage", required=True)
    for name, help_text in STAGES.items():
        p = sub.add_parser(name, help=help_text)
        p.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                       help="ghi de gia tri trong config, vi du --set evaluation.primary_k=10")
        if name in ("dataset", "folds"):
            p.add_argument("--force", action="store_true",
                           help="tao lai du du lieu da ton tai")
    return parser


def apply_overrides(cfg: dict, items: list[str]) -> dict:
    for item in items:
        if "=" not in item:
            raise SystemExit(f"Gia tri --set khong hop le: {item!r} (can dang key=value)")
        key, raw = item.split("=", 1)
        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            value = raw
        node = cfg
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = value
    return cfg


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["experiments"]["split_seed"] = args.seed
    if args.device is not None:
        cfg["runtime"]["device"] = args.device
    if getattr(args, "set", None):
        apply_overrides(cfg, args.set)

    set_seed(int(get(cfg, "experiments.split_seed", 42)))
    ensure_dirs(cfg)
    device = resolve_device(cfg)

    root = Path(__file__).resolve().parents[1]
    print(f"[frevia] config      : {args.config}")
    print(f"[frevia] root        : {root}")
    print(f"[frevia] python      : {platform.python_version()} ({sys.platform})")
    print(f"[frevia] device      : {device}")
    print(f"[frevia] dataset     : {get(cfg, 'data.hf_dataset')}")
    print(f"[frevia] split       : {get(cfg, 'data.split.strategy')} n_splits="
          f"{get(cfg, 'data.split.n_splits')} by={get(cfg, 'data.split.by')}")
    print(f"[frevia] embed model : {get(cfg, 'embedding.model')}")
    print(f"[frevia] llm         : {get(cfg, 'extraction.provider')}:{get(cfg, 'extraction.model')}")

    if args.stage == "info":
        print(json.dumps(cfg, indent=2, ensure_ascii=False))
        return 0

    if args.stage == "prepare":
        import torch

        info = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "git_commit": git_hash(root),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_version": torch.version.cuda,
            "device": device,
        }
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
        out = Path(get(cfg, "paths.logs")) / "env_info.json"
        out.write_text(json.dumps(info, indent=2), encoding="utf-8")
        print(f"[frevia] da ghi {out}")
        return 0

    if args.stage == "folds":
        from src.data.pipeline import build_all_splits

        report = build_all_splits(cfg)
        for group_col, summary in report["split_systems"].items():
            print(
                f"[frevia] split theo {group_col}: "
                f"{summary['n_groups_total']} nhom, cot '{summary['fold_column']}', "
                f"nhom/fold it nhat {min(p['n_cv'] for p in summary['per_fold'])} CV, "
                f"{summary['rows_min']}-{summary['rows_max']} dong"
            )
        print(f"[frevia] da ghi bang ro ri: {Path(get(cfg, 'paths.tables')) / 'leakage_audit.csv'}")
        return 0

    if args.stage == "dataset":
        from src.data.pipeline import prepare_pairs

        pairs, report = prepare_pairs(cfg, force=bool(getattr(args, "force", False)))
        fp = report["fingerprint"]
        print(f"[frevia] dataset : {fp['hf_dataset']} ({fp['n_rows']} dong goc)")
        print(f"[frevia] unique  : {fp['n_resumes']} CV, {fp['n_jobs']} JD")
        print(
            f"[frevia] labels  : {report['rows_in']} -> {report['rows_out']} dong "
            f"(loai ngan {report['rows_dropped_short_text']}, "
            f"trung cap {report['rows_dropped_duplicate_pair']})"
        )
        return 0

    raise SystemExit(f"Chua cai dat stage: {args.stage}")


if __name__ == "__main__":
    raise SystemExit(main())
