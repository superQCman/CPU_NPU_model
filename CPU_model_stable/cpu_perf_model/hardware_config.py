from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def materialize_hardware_profile(
    *,
    hardware_json_path: Path | None,
    output_dir: Path,
    filename: str = "hardware_profile_resolved.yaml",
) -> tuple[Path | None, dict[str, Any] | None]:
    if hardware_json_path is None:
        return None, None

    source_path = Path(hardware_json_path)
    suffix = source_path.suffix.lower()
    if suffix in {".yaml", ".yml"}:
        with source_path.open("r", encoding="utf-8") as handle:
            payload = yaml.safe_load(handle) or {}
    else:
        with source_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)

    output_dir.mkdir(parents=True, exist_ok=True)
    resolved_yaml_path = output_dir / filename
    resolved_yaml_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    snapshot_json_path = output_dir / "hardware_input_snapshot.json"
    snapshot_json_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return resolved_yaml_path, payload
