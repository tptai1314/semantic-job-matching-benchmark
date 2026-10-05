from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG_PATH = "configs/base.yaml"


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else project_root() / DEFAULT_CONFIG_PATH
    if not cfg_path.is_absolute():
        cfg_path = project_root() / cfg_path
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    if overrides:
        cfg = deep_update(cfg, overrides)
    resolve_paths(cfg)
    return cfg


def deep_update(base: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def resolve_paths(cfg: dict[str, Any]) -> dict[str, Any]:
    root = Path(cfg["project"]["root"])
    if not root.is_absolute():
        root = project_root() / root
    for key, value in cfg["paths"].items():
        p = Path(value)
        cfg["paths"][key] = str(root / p) if not p.is_absolute() else str(p)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    paths = cfg["paths"]
    for key in ("raw", "processed", "splits", "results", "baselines_results", "tables",
                "figures", "logs", "checkpoints"):
        Path(paths[key]).mkdir(parents=True, exist_ok=True)
    for sub in ("views", "embeddings", "llm_logs"):
        Path(paths["cache"], sub).mkdir(parents=True, exist_ok=True)
    Path(paths["cache_embeddings"], "b2").mkdir(parents=True, exist_ok=True)
    Path(paths["cache_embeddings"], "views").mkdir(parents=True, exist_ok=True)


def get(cfg: dict[str, Any], dotted: str, default: Any = None) -> Any:
    node: Any = cfg
    for part in dotted.split("."):
        if not isinstance(node, dict) or part not in node:
            return default
        node = node[part]
    return node
