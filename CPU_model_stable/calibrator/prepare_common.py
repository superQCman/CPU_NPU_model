from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pandas as pd


TRAIN_ONLY_RATIOS = {"train": 1.0, "val": 0.0, "test": 0.0}


def _json_dump(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _contains_float_placeholder_shape(value: Any) -> bool:
    if pd.isna(value):
        return False
    text = str(value).strip()
    if not text:
        return False
    try:
        parsed = ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return False
    if not isinstance(parsed, list):
        return False
    for entry in parsed:
        if not isinstance(entry, dict):
            continue
        for dtype, dims in entry.items():
            if str(dtype).lower().startswith("float") and isinstance(dims, (list, tuple)) and len(dims) == 0:
                return True
    return False


def _shape_empty_mask(frame: pd.DataFrame) -> pd.Series:
    input_placeholder = frame["input_type_shape"].map(_contains_float_placeholder_shape)
    output_placeholder = frame["output_type_shape"].map(_contains_float_placeholder_shape)
    return input_placeholder | output_placeholder


def _patch_shape_backfill(output_root: Path) -> dict[str, Any]:
    dataset_full_path = output_root / "dataset_full.csv"
    if not dataset_full_path.exists():
        return {
            "patched_files": [],
            "rows_backfilled": 0,
            "empty_shape_rows_before": 0,
            "empty_shape_rows_after": 0,
        }

    dataset_full = pd.read_csv(dataset_full_path, low_memory=False)
    dataset_full["combo_op_key"] = dataset_full["combo"].astype(str) + "::opidx:" + dataset_full["op_idx"].astype(str)
    empty_before_mask = _shape_empty_mask(dataset_full)
    empty_before = int(empty_before_mask.sum())
    patched_files: list[str] = []

    if empty_before <= 0:
        return {
            "patched_files": patched_files,
            "rows_backfilled": 0,
            "empty_shape_rows_before": 0,
            "empty_shape_rows_after": 0,
        }

    reference = dataset_full.loc[
        ~empty_before_mask,
        ["combo_op_key", "input_type_shape", "output_type_shape"],
    ].drop_duplicates(subset=["combo_op_key"], keep="first")
    if reference["combo_op_key"].duplicated().any():
        duplicated = sorted(reference.loc[reference["combo_op_key"].duplicated(), "combo_op_key"].astype(str).unique().tolist())
        raise ValueError(f"Ambiguous combo/op reference keys for shape backfill: {duplicated[:10]}")

    reference = reference.set_index("combo_op_key")

    def _apply_shape_backfill(frame: pd.DataFrame) -> tuple[pd.DataFrame, int]:
        work = frame.copy()
        work["combo_op_key"] = work["combo"].astype(str) + "::opidx:" + work["op_idx"].astype(str)
        empty_mask = _shape_empty_mask(work)
        if not empty_mask.any():
            return work.drop(columns=["combo_op_key"]), 0
        missing_keys = [key for key in work.loc[empty_mask, "combo_op_key"].astype(str).tolist() if key not in reference.index]
        if missing_keys:
            raise ValueError(
                "Missing peer-thread shape reference for combo/op keys: "
                + ", ".join(sorted(dict.fromkeys(missing_keys))[:10])
            )
        refill = reference.reindex(work.loc[empty_mask, "combo_op_key"].astype(str))
        work.loc[empty_mask, "input_type_shape"] = refill["input_type_shape"].to_numpy()
        work.loc[empty_mask, "output_type_shape"] = refill["output_type_shape"].to_numpy()
        return work.drop(columns=["combo_op_key"]), int(empty_mask.sum())

    rows_backfilled = 0
    for split_name in ("dataset_full", "train", "val", "test"):
        csv_path = output_root / f"{split_name}.csv"
        if not csv_path.exists():
            continue
        frame = pd.read_csv(csv_path, low_memory=False)
        patched, patched_rows = _apply_shape_backfill(frame)
        patched.to_csv(csv_path, index=False)
        patched_files.append(csv_path.name)
        if split_name == "dataset_full":
            rows_backfilled = patched_rows

    patched_full = pd.read_csv(dataset_full_path, low_memory=False)
    empty_after = int(_shape_empty_mask(patched_full).sum())
    if empty_after != 0:
        raise ValueError(f"Shape backfill left {empty_after} rows with empty shapes in {dataset_full_path}")

    return {
        "patched_files": patched_files,
        "rows_backfilled": rows_backfilled,
        "empty_shape_rows_before": empty_before,
        "empty_shape_rows_after": empty_after,
    }
