from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_INPUT_DATA_ROOT = SCRIPT_DIR / "data"
DEFAULT_OUTPUT_DATA_ROOT = SCRIPT_DIR / "data_e2e_small_1_16"
SOURCE_PREFIX = "source_kunpeng920_dataset3_unseen_"
PIPELINE_INDEX_CSV = "pipeline_artifact_index.csv"
PIPELINE_INDEX_JSON = "pipeline_artifact_index.json"
PIPELINE_COMBO_MANIFEST_CSV = "pipeline_combo_manifest.csv"
E2E_TRAIN_CSV = "e2e_train.csv"
E2E_TEST_RAW_CSV = "e2e_test_raw.csv"
E2E_TRAIN_GROUPS_CSV = "e2e_train_case_combo_groups.csv"
E2E_TEST_RAW_GROUPS_CSV = "e2e_test_raw_case_combo_groups.csv"


def _json_dump(path: Path, payload: dict[str, Any] | list[Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _pipeline_key(frame: pd.DataFrame) -> pd.Series:
    return frame["source_name"].astype(str) + "::" + frame["combo"].astype(str)


def _stable_score(source_name: Any, combo: Any, *, seed: int, split: str) -> int:
    text = f"{seed}|{split}|{source_name}|{combo}"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:16], 16)


def _select_group_keys(groups: pd.DataFrame, *, divisor: int, seed: int, split: str) -> set[str]:
    if groups.empty:
        return set()
    work = groups.copy()
    work["pipeline_index_key"] = _pipeline_key(work)
    work["_score"] = [
        _stable_score(row.source_name, row.combo, seed=seed, split=split)
        for row in work.itertuples(index=False)
    ]
    target_total = max(1, int(round(len(work) / float(divisor))))
    if "case_id" not in work.columns:
        return set(
            work.sort_values("_score", kind="stable")
            .head(target_total)["pipeline_index_key"]
            .astype(str)
        )

    allocations: dict[str, int] = {}
    fractions: list[tuple[float, str]] = []
    grouped = list(work.groupby("case_id", sort=True))
    for case_id, case_group in grouped:
        quota = len(case_group) / float(divisor)
        take = int(math.floor(quota))
        if len(case_group) > 0:
            take = max(1, take)
        take = min(take, len(case_group))
        allocations[str(case_id)] = take
        fractions.append((quota - math.floor(quota), str(case_id)))

    current = sum(allocations.values())
    if current < target_total:
        for _, case_id in sorted(fractions, reverse=True):
            if current >= target_total:
                break
            case_size = int((work["case_id"].astype(str) == case_id).sum())
            if allocations[case_id] < case_size:
                allocations[case_id] += 1
                current += 1
    elif current > target_total:
        for _, case_id in sorted(fractions):
            if current <= target_total:
                break
            if allocations[case_id] > 1:
                allocations[case_id] -= 1
                current -= 1

    selected: set[str] = set()
    for case_id, case_group in grouped:
        take = allocations[str(case_id)]
        selected.update(
            case_group.sort_values("_score", kind="stable")
            .head(take)["pipeline_index_key"]
            .astype(str)
        )
    return selected


def _group_case_combo_rows(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=["source_name", "case_id", "combo", "row_count"])
    return (
        frame.groupby(["source_name", "case_id", "combo"], dropna=False)
        .size()
        .reset_index(name="row_count")
        .sort_values(["source_name", "combo"], kind="stable")
        .reset_index(drop=True)
    )


def _copy_base_files(input_root: Path, output_root: Path) -> None:
    skip = {
        E2E_TRAIN_CSV,
        E2E_TEST_RAW_CSV,
        E2E_TRAIN_GROUPS_CSV,
        E2E_TEST_RAW_GROUPS_CSV,
        PIPELINE_INDEX_CSV,
        PIPELINE_INDEX_JSON,
        PIPELINE_COMBO_MANIFEST_CSV,
        "split_manifest.json",
        "split_summary.csv",
    }
    for path in input_root.iterdir():
        if not path.is_file() or path.name in skip or path.suffix == ".pkl":
            continue
        shutil.copy2(path, output_root / path.name)


def _write_split_summary(output_root: Path, manifest: dict[str, Any]) -> None:
    rows = [
        {
            "split": "train",
            "rows": manifest.get("retained_train_rows"),
            "case_combo_groups": manifest.get("train_case_combo_groups"),
            "case_count": len(manifest.get("train_case_ids", [])),
            "ratio": manifest.get("retained_train_ratio"),
        },
        {
            "split": "test",
            "rows": manifest.get("retained_test_rows"),
            "case_combo_groups": manifest.get("test_case_combo_groups"),
            "case_count": len(manifest.get("test_case_ids", [])),
            "ratio": manifest.get("retained_test_ratio"),
        },
        {
            "split": "train_raw",
            "rows": manifest.get("raw_train_rows"),
            "case_combo_groups": manifest.get("train_raw_case_combo_groups"),
            "case_count": len(manifest.get("raw_train_case_ids", [])),
            "ratio": manifest.get("raw_train_ratio"),
        },
        {
            "split": "test_raw",
            "rows": manifest.get("raw_test_rows"),
            "case_combo_groups": manifest.get("test_raw_case_combo_groups"),
            "case_count": len(manifest.get("raw_test_case_ids", [])),
            "ratio": manifest.get("raw_test_ratio"),
        },
        {
            "split": "e2e_train",
            "rows": manifest.get("e2e_train_rows"),
            "case_combo_groups": manifest.get("e2e_train_case_combo_groups"),
            "case_count": len(manifest.get("raw_train_case_ids", [])),
            "ratio": manifest.get("e2e_train_combo_ratio"),
        },
        {
            "split": "e2e_test_raw",
            "rows": manifest.get("e2e_test_rows"),
            "case_combo_groups": manifest.get("e2e_test_case_combo_groups"),
            "case_count": len(manifest.get("raw_test_case_ids", [])),
            "ratio": manifest.get("e2e_test_ratio"),
        },
    ]
    pd.DataFrame(rows).to_csv(output_root / "split_summary.csv", index=False)


def build_small_split(input_root: Path, output_root: Path, *, divisor: int, seed: int, overwrite: bool) -> dict[str, Any]:
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(f"{output_root} already exists; pass --overwrite to replace it")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    _copy_base_files(input_root, output_root)

    train_groups = pd.read_csv(input_root / E2E_TRAIN_GROUPS_CSV)
    test_groups = pd.read_csv(input_root / E2E_TEST_RAW_GROUPS_CSV)
    train_keys = _select_group_keys(train_groups, divisor=divisor, seed=seed, split="e2e_train")
    test_keys = _select_group_keys(test_groups, divisor=divisor, seed=seed, split="e2e_test_raw")
    selected_keys = train_keys | test_keys

    e2e_train = pd.read_csv(input_root / E2E_TRAIN_CSV, low_memory=False)
    e2e_train_small = e2e_train.loc[_pipeline_key(e2e_train).isin(train_keys)].copy()
    e2e_train_small.to_csv(output_root / E2E_TRAIN_CSV, index=False)
    train_groups_small = _group_case_combo_rows(e2e_train_small)
    train_groups_small.to_csv(output_root / E2E_TRAIN_GROUPS_CSV, index=False)

    e2e_test = pd.read_csv(input_root / E2E_TEST_RAW_CSV, low_memory=False)
    e2e_test_small = e2e_test.loc[_pipeline_key(e2e_test).isin(test_keys)].copy()
    e2e_test_small.to_csv(output_root / E2E_TEST_RAW_CSV, index=False)
    test_groups_small = _group_case_combo_rows(e2e_test_small)
    test_groups_small.to_csv(output_root / E2E_TEST_RAW_GROUPS_CSV, index=False)

    combo_manifest = pd.read_csv(input_root / PIPELINE_COMBO_MANIFEST_CSV, low_memory=False)
    combo_manifest_small = combo_manifest.loc[combo_manifest["pipeline_index_key"].astype(str).isin(selected_keys)].copy()
    combo_manifest_small.to_csv(output_root / PIPELINE_COMBO_MANIFEST_CSV, index=False)

    artifact_index = pd.read_csv(input_root / PIPELINE_INDEX_CSV, low_memory=False)
    artifact_index_small = artifact_index.loc[artifact_index["pipeline_index_key"].astype(str).isin(selected_keys)].copy()
    artifact_index_small.to_csv(output_root / PIPELINE_INDEX_CSV, index=False)
    _json_dump(
        output_root / PIPELINE_INDEX_JSON,
        {
            "row_count": int(len(artifact_index_small)),
            "rows": artifact_index_small.to_dict(orient="records"),
        },
    )

    manifest = json.loads((input_root / "split_manifest.json").read_text(encoding="utf-8"))
    original_e2e_combo_count = int(manifest.get("e2e_total_combo_count", len(train_groups) + len(test_groups)))
    small_e2e_combo_count = int(len(train_groups_small) + len(test_groups_small))
    manifest.update(
        {
            "small_dataset_parent": str(input_root.resolve()),
            "small_dataset_divisor": int(divisor),
            "small_dataset_seed": int(seed),
            "small_dataset_combo_fraction": float(small_e2e_combo_count / max(original_e2e_combo_count, 1)),
            "pipeline_artifact_index_csv": str((output_root / PIPELINE_INDEX_CSV).resolve()),
            "pipeline_artifact_index_json": str((output_root / PIPELINE_INDEX_JSON).resolve()),
            "pipeline_artifact_index_rows": int(len(artifact_index_small)),
            "pipeline_combo_manifest_csv": str((output_root / PIPELINE_COMBO_MANIFEST_CSV).resolve()),
            "pipeline_combo_manifest_rows": int(len(combo_manifest_small)),
            "e2e_train_rows": int(len(e2e_train_small)),
            "e2e_train_case_combo_groups": int(len(train_groups_small)),
            "e2e_test_rows": int(len(e2e_test_small)),
            "e2e_test_case_combo_groups": int(len(test_groups_small)),
            "e2e_total_combo_count": small_e2e_combo_count,
            "e2e_train_combo_count": int(len(train_groups_small)),
            "e2e_selected_combo_count": int(len(test_groups_small)),
        }
    )
    total_small = max(small_e2e_combo_count, 1)
    manifest["e2e_train_combo_ratio"] = float(len(train_groups_small) / total_small)
    manifest["e2e_test_ratio"] = float(len(test_groups_small) / total_small)
    _json_dump(output_root / "split_manifest.json", manifest)
    _write_split_summary(output_root, manifest)

    return {
        "dataset_root": str(output_root),
        "parent_root": str(input_root),
        "e2e_train_combos": int(len(train_groups_small)),
        "e2e_test_combos": int(len(test_groups_small)),
        "e2e_train_rows": int(len(e2e_train_small)),
        "e2e_test_rows": int(len(e2e_test_small)),
        "pipeline_combo_manifest_rows": int(len(combo_manifest_small)),
        "pipeline_artifact_index_rows": int(len(artifact_index_small)),
    }


def _policy_roots(data_root: Path, policy: str) -> list[Path]:
    if policy == "all":
        return sorted(
            path
            for path in data_root.iterdir()
            if path.is_dir() and path.name.startswith(SOURCE_PREFIX) and "_small_" not in path.name
        )
    root = data_root / f"{SOURCE_PREFIX}{policy}"
    if not root.exists():
        raise FileNotFoundError(root)
    return [root]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build 1/N small E2E pipeline split roots from existing source splits.")
    parser.add_argument("--input-data-root", type=Path, default=DEFAULT_INPUT_DATA_ROOT)
    parser.add_argument("--output-data-root", type=Path, default=DEFAULT_OUTPUT_DATA_ROOT)
    parser.add_argument("--policy", default="all")
    parser.add_argument("--divisor", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260425)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.divisor <= 0:
        raise ValueError("--divisor must be positive")
    input_data_root = args.input_data_root.resolve()
    output_data_root = args.output_data_root.resolve()
    output_data_root.mkdir(parents=True, exist_ok=True)
    rows = []
    for input_root in _policy_roots(input_data_root, str(args.policy)):
        output_root = output_data_root / input_root.name
        summary = build_small_split(
            input_root.resolve(),
            output_root,
            divisor=int(args.divisor),
            seed=int(args.seed),
            overwrite=bool(args.overwrite),
        )
        rows.append(summary)
        print(
            "[small-e2e] "
            f"{input_root.name}: train_combos={summary['e2e_train_combos']} "
            f"test_combos={summary['e2e_test_combos']} "
            f"train_rows={summary['e2e_train_rows']} test_rows={summary['e2e_test_rows']}",
            flush=True,
        )
    pd.DataFrame(rows).to_csv(output_data_root / "small_split_summary.csv", index=False)
    print(f"[small-e2e] summary={output_data_root / 'small_split_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
