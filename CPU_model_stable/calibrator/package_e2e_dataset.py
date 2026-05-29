from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


PIPELINE_INDEX_CSV_NAME = "pipeline_artifact_index.csv"
PIPELINE_INDEX_JSON_NAME = "pipeline_artifact_index.json"
PORTABLE_ARTIFACT_ROOT_NAME = "portable_artifacts"
PROVENANCE_COLUMNS = (
    "original_sweep_dir",
    "original_shape_csv",
    "original_profile_dir",
    "original_timeline_csv",
    "original_effective_onnx_path",
    "original_log_dir",
)


def _json_dump(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_relative_to(path: Path, root: Path) -> bool:
    try:
        path.resolve().relative_to(root.resolve())
        return True
    except ValueError:
        return False


def _find_split_roots(input_root: Path) -> list[Path]:
    if (input_root / PIPELINE_INDEX_CSV_NAME).is_file():
        return [input_root]
    return sorted(path.parent for path in input_root.rglob(PIPELINE_INDEX_CSV_NAME))


def _copy_dataset_tree(input_root: Path, output_root: Path) -> None:
    for source_path in sorted(input_root.rglob("*")):
        if source_path.is_dir():
            continue
        rel = source_path.relative_to(input_root)
        if rel.parts and rel.parts[0] == PORTABLE_ARTIFACT_ROOT_NAME:
            continue
        if PORTABLE_ARTIFACT_ROOT_NAME in rel.parts:
            continue
        dest = output_root / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, dest)


def _path_text(value: Any) -> str:
    if pd.isna(value):
        return ""
    return str(value).strip()


def _source_path(value: Any, *, split_root: Path) -> Path | None:
    text = _path_text(value)
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = split_root / path
    return path.resolve()


def _safe_part(value: Any) -> str:
    text = str(value).strip() or "unknown"
    return "".join(ch if ch.isalnum() or ch in {"-", "_", "."} else "_" for ch in text)


def _copy_artifact(
    *,
    artifact_type: str,
    source_path: Path | None,
    source_split_root: Path,
    output_split_root: Path,
    row: pd.Series,
    file_records: list[dict[str, Any]],
    missing_records: list[dict[str, Any]],
) -> str:
    if source_path is None:
        return ""
    if not source_path.exists():
        missing_records.append(
            {
                "artifact_type": artifact_type,
                "source_name": _path_text(row.get("source_name")),
                "combo": _path_text(row.get("combo")),
                "original_path": str(source_path),
                "reason": "missing",
            }
        )
        return ""
    rel_dest = Path(PORTABLE_ARTIFACT_ROOT_NAME) / _safe_part(row.get("source_name")) / _safe_part(row.get("combo")) / artifact_type / source_path.name
    dest = output_split_root / rel_dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    if source_path.resolve() != dest.resolve():
        shutil.copy2(source_path, dest)
    file_records.append(
        {
            "artifact_type": artifact_type,
            "source_name": _path_text(row.get("source_name")),
            "combo": _path_text(row.get("combo")),
            "original_path": str(source_path),
            "relative_path": rel_dest.as_posix(),
            "size_bytes": int(dest.stat().st_size),
            "sha256": _sha256(dest),
        }
    )
    return rel_dest.as_posix()


def _package_split(
    *,
    source_split_root: Path,
    output_split_root: Path,
    include_effective_onnx: bool,
) -> dict[str, Any]:
    index_path = output_split_root / PIPELINE_INDEX_CSV_NAME
    if not index_path.exists():
        raise FileNotFoundError(f"Copied split is missing {PIPELINE_INDEX_CSV_NAME}: {output_split_root}")
    index_df = pd.read_csv(index_path, low_memory=False)
    file_records: list[dict[str, Any]] = []
    missing_records: list[dict[str, Any]] = []
    rewritten = index_df.copy()
    for column in PROVENANCE_COLUMNS:
        if column not in rewritten.columns:
            rewritten[column] = ""
    for idx, row in index_df.iterrows():
        original_shape = _source_path(row.get("shape_csv"), split_root=source_split_root)
        original_timeline = _source_path(row.get("timeline_csv"), split_root=source_split_root)
        original_effective = _source_path(row.get("effective_onnx_path"), split_root=source_split_root)
        rewritten.at[idx, "original_sweep_dir"] = _path_text(row.get("sweep_dir"))
        rewritten.at[idx, "original_shape_csv"] = str(original_shape or "")
        rewritten.at[idx, "original_profile_dir"] = _path_text(row.get("profile_dir"))
        rewritten.at[idx, "original_timeline_csv"] = str(original_timeline or "")
        rewritten.at[idx, "original_effective_onnx_path"] = str(original_effective or "")
        rewritten.at[idx, "original_log_dir"] = _path_text(row.get("log_dir"))

        shape_rel = _copy_artifact(
            artifact_type="op_shapes",
            source_path=original_shape,
            source_split_root=source_split_root,
            output_split_root=output_split_root,
            row=row,
            file_records=file_records,
            missing_records=missing_records,
        )
        timeline_rel = _copy_artifact(
            artifact_type="timeline",
            source_path=original_timeline,
            source_split_root=source_split_root,
            output_split_root=output_split_root,
            row=row,
            file_records=file_records,
            missing_records=missing_records,
        )
        effective_rel = ""
        if include_effective_onnx:
            effective_rel = _copy_artifact(
                artifact_type="effective_onnx",
                source_path=original_effective,
                source_split_root=source_split_root,
                output_split_root=output_split_root,
                row=row,
                file_records=file_records,
                missing_records=missing_records,
            )
        rewritten.at[idx, "sweep_dir"] = ""
        rewritten.at[idx, "shape_csv"] = shape_rel
        rewritten.at[idx, "profile_dir"] = str(Path(timeline_rel).parent.as_posix()) if timeline_rel else ""
        rewritten.at[idx, "timeline_csv"] = timeline_rel
        rewritten.at[idx, "effective_onnx_path"] = effective_rel
        rewritten.at[idx, "log_dir"] = ""
        rewritten.at[idx, "shape_csv_exists"] = bool(shape_rel and (output_split_root / shape_rel).exists())
        rewritten.at[idx, "timeline_csv_exists"] = bool(timeline_rel and (output_split_root / timeline_rel).exists())
        rewritten.at[idx, "effective_onnx_exists"] = bool(effective_rel and (output_split_root / effective_rel).exists())

    rewritten.to_csv(index_path, index=False)
    _json_dump(
        output_split_root / PIPELINE_INDEX_JSON_NAME,
        {
            "row_count": int(len(rewritten)),
            "portable": True,
            "path_base": "split_root",
            "rows": rewritten.to_dict(orient="records"),
        },
    )
    _rewrite_split_manifest(output_split_root)
    return {
        "source_split_root": str(source_split_root),
        "output_split_root": str(output_split_root),
        "pipeline_index_rows": int(len(rewritten)),
        "copied_files": int(len(file_records)),
        "missing_files": int(len(missing_records)),
        "files": file_records,
        "missing": missing_records,
    }


def _rewrite_split_manifest(output_split_root: Path) -> None:
    manifest_path = output_split_root / "split_manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    path_rewrites = {
        "input_dataset_root": ".",
        "scheduler_logs_root": "",
        "pipeline_artifact_index_csv": PIPELINE_INDEX_CSV_NAME,
        "pipeline_artifact_index_json": PIPELINE_INDEX_JSON_NAME,
        "pipeline_combo_manifest_csv": "pipeline_combo_manifest.csv",
    }
    for key, replacement in path_rewrites.items():
        value = str(manifest.get(key, "") or "").strip()
        if value and f"original_{key}" not in manifest:
            manifest[f"original_{key}"] = value
        if key in manifest:
            manifest[key] = replacement
    manifest["portable_dataset"] = True
    manifest["portable_artifact_root"] = PORTABLE_ARTIFACT_ROOT_NAME
    _json_dump(manifest_path, manifest)


def _rewrite_dataset_manifest(output_root: Path, split_summaries: list[dict[str, Any]]) -> None:
    manifest_path = output_root / "split_manifest.json"
    if not manifest_path.exists():
        return
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return
    for key in [
        "requested_source_root",
        "legacy_source_root",
        "legacy_calibrator_source_root",
        "phase2_source_root",
        "legacy_calibrator_phase2_source_root",
        "target_root",
        "output_root",
    ]:
        value = str(manifest.get(key, "") or "").strip()
        if value and f"original_{key}" not in manifest:
            manifest[f"original_{key}"] = value
        if key in manifest:
            manifest[key] = "."
    policies = manifest.get("policies")
    portable_rel_splits = [
        str(Path(item["output_split_root"]).resolve().relative_to(output_root.resolve()).as_posix())
        for item in split_summaries
        if "output_split_root" in item
    ]
    if isinstance(policies, dict):
        for _, policy_payload in policies.items():
            if not isinstance(policy_payload, dict):
                continue
            for key in ["source_input_dataset_root", "target_input_dataset_root"]:
                value = str(policy_payload.get(key, "") or "").strip()
                if value and f"original_{key}" not in policy_payload:
                    policy_payload[f"original_{key}"] = value
                if key in policy_payload:
                    policy_payload[key] = "."
            for key in ["source_output_root", "target_output_root"]:
                value = str(policy_payload.get(key, "") or "").strip()
                if value and f"original_{key}" not in policy_payload:
                    policy_payload[f"original_{key}"] = value
                if key in policy_payload:
                    try:
                        policy_payload[key] = Path(value).resolve().relative_to(output_root.resolve()).as_posix()
                    except ValueError:
                        policy_payload[key] = portable_rel_splits[0] if portable_rel_splits else ""
    manifest["portable_dataset"] = True
    manifest["portable_split_roots"] = portable_rel_splits
    _json_dump(manifest_path, manifest)


def package_dataset(
    *,
    input_root: Path,
    output_root: Path,
    include_effective_onnx: bool = False,
) -> dict[str, Any]:
    input_root = input_root.resolve()
    output_root = output_root.resolve()
    if input_root == output_root:
        raise ValueError("Refusing in-place packaging; choose a distinct output root.")
    if _is_relative_to(output_root, input_root):
        raise ValueError("Refusing to place output_root inside input_root.")
    if output_root.exists() and any(output_root.iterdir()):
        raise FileExistsError(f"Output root already exists and is not empty: {output_root}")
    split_roots = _find_split_roots(input_root)
    if not split_roots:
        raise FileNotFoundError(f"No {PIPELINE_INDEX_CSV_NAME} found under {input_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_dataset_tree(input_root, output_root)
    split_summaries: list[dict[str, Any]] = []
    for source_split in split_roots:
        rel_split = source_split.relative_to(input_root)
        output_split = output_root / rel_split
        split_summaries.append(
            _package_split(
                source_split_root=source_split,
                output_split_root=output_split,
                include_effective_onnx=include_effective_onnx,
            )
        )
    _rewrite_dataset_manifest(output_root, split_summaries)
    manifest = {
        "portable_dataset_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_root": str(input_root),
        "output_root": str(output_root),
        "include_effective_onnx": bool(include_effective_onnx),
        "path_base": "split_root",
        "split_count": int(len(split_summaries)),
        "splits": split_summaries,
    }
    _json_dump(output_root / "portable_manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Package an E2E split/dataset root into a self-contained portable directory.")
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--include-effective-onnx", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    manifest = package_dataset(
        input_root=args.input_root,
        output_root=args.output_root,
        include_effective_onnx=bool(args.include_effective_onnx),
    )
    print(json.dumps({k: v for k, v in manifest.items() if k != "splits"}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
