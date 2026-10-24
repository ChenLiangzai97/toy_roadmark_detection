"""Small shared utilities: YAML loading, JSON write, deterministic helpers."""
from __future__ import annotations

import json
import pathlib

import yaml


def load_yaml(path: str | pathlib.Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_json(path: str | pathlib.Path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def write_json(path: str | pathlib.Path, obj) -> None:
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=1), encoding="utf-8")


def deep_merge(base: dict, override: dict) -> dict:
    """Recursive merge for --override configs; override wins."""
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out
