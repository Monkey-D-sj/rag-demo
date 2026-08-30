"""答案评测 history artifact 的原子持久化与状态机。"""

from __future__ import annotations

import json
import os
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


STATES = {"collecting", "collected", "scoring", "completed", "partial"}
TRANSITIONS = {
    "collecting": {"collecting", "collected", "partial"},
    "collected": {"collected", "scoring", "partial"},
    "scoring": {"scoring", "completed", "partial"},
    "partial": {"partial", "collecting", "scoring", "completed"},
    "completed": {"completed"},
}


def make_run_id(root: str | Path, *, now: datetime | None = None) -> str:
    root = Path(root)
    stamp = (now or datetime.now()).strftime("%Y%m%d-%H%M%S")
    try:
        sha = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        sha = "unknown"
    candidate = f"{stamp}-{sha}"
    if (root / candidate).exists():
        candidate += f"-{secrets.token_hex(3)}"
    return candidate


def atomic_write_json(path: str | Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{secrets.token_hex(6)}.tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temp, path)


def run_dir(history_dir: str | Path, run_id: str) -> Path:
    return Path(history_dir) / run_id


def item_path(history_dir: str | Path, run_id: str, item_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "-" for c in str(item_id)).strip("-") or "item"
    return run_dir(history_dir, run_id) / "items" / f"{safe}.json"


def create_run(history_dir: str | Path, run_id: str, manifest: dict[str, Any]) -> Path:
    directory = run_dir(history_dir, run_id)
    if directory.exists():
        raise FileExistsError(f"run 已存在: {run_id}")
    directory.joinpath("items").mkdir(parents=True)
    manifest = dict(manifest)
    manifest.setdefault("schema_version", 1)
    manifest.setdefault("run_id", run_id)
    manifest.setdefault("status", "collecting")
    manifest.setdefault("created_at", datetime.now(timezone.utc).isoformat())
    atomic_write_json(directory / "manifest.json", manifest)
    return directory


def load_manifest(history_dir: str | Path, run_id: str) -> dict[str, Any]:
    path = run_dir(history_dir, run_id) / "manifest.json"
    if not path.exists():
        raise FileNotFoundError(f"找不到评测 run: {run_id}")
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(history_dir: str | Path, run_id: str, **changes: Any) -> dict[str, Any]:
    manifest = load_manifest(history_dir, run_id)
    old = manifest.get("status", "collecting")
    new = changes.get("status", old)
    if new not in STATES or new not in TRANSITIONS.get(old, set()):
        raise ValueError(f"非法状态迁移: {old} -> {new}")
    manifest.update(changes)
    manifest["updated_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(run_dir(history_dir, run_id) / "manifest.json", manifest)
    return manifest


def write_item(history_dir: str | Path, run_id: str, item_id: str, item: dict[str, Any]) -> Path:
    path = item_path(history_dir, run_id, item_id)
    atomic_write_json(path, item)
    return path


def read_item(history_dir: str | Path, run_id: str, item_id: str) -> dict[str, Any] | None:
    path = item_path(history_dir, run_id, item_id)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def list_items(history_dir: str | Path, run_id: str) -> list[dict[str, Any]]:
    directory = run_dir(history_dir, run_id) / "items"
    if not directory.exists():
        return []
    return [json.loads(path.read_text(encoding="utf-8")) for path in sorted(directory.glob("*.json"))]


def assert_resume_compatible(manifest: dict[str, Any], expected: dict[str, Any]) -> None:
    for key in ("sample_fingerprint", "selected_ids", "generator_model", "judge_model", "embedding_model", "judge_prompt_hash"):
        if key in expected and manifest.get(key) != expected[key]:
            raise ValueError(f"resume provenance 不一致: {key}")
    if manifest.get("status") == "completed":
        raise ValueError("completed run 默认不可重复执行，请使用 --rescore")

